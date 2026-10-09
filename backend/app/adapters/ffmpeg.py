"""Thin wrapper around the ffmpeg / ffprobe command-line tools (plain functions).

Nothing outside this file builds ffmpeg command lines. Stages call
these functions and get plain Python objects back.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app.config import settings

ProgressFn = Callable[[float], None]  # receives 0.0 .. 1.0

# Subtitle codecs that are text and can be converted to .srt.
# Image-based ones (hdmv_pgs_subtitle, dvd_subtitle, dvb_subtitle) cannot.
TEXT_SUBTITLE_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}


class FFmpegError(RuntimeError):
    pass


@dataclass
class SubtitleTrack:
    index: int                 # absolute stream index in the file
    codec: str
    language: Optional[str]
    is_text: bool


@dataclass
class MediaInfo:
    duration_ms: int
    width: int
    height: int
    fps: float
    video_codec: str
    audio_tracks: int
    audio_codec: Optional[str]
    subtitle_tracks: list[SubtitleTrack] = field(default_factory=list)
    container: str = ""
    size_bytes: int = 0

    @property
    def has_audio(self) -> bool:
        return self.audio_tracks > 0

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_fps(rate: str | None) -> float:
    """'24000/1001' -> 23.976"""
    if not rate or rate in ("0/0", "0"):
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        return round(float(num) / float(den), 3) if float(den) else 0.0
    return round(float(rate), 3)


# ---------------------------------------------------------------- checks
def check_installed() -> None:
    for name, binary in (("ffmpeg", settings.ffmpeg_bin), ("ffprobe", settings.ffprobe_bin)):
        if shutil.which(binary) is None:
            raise FFmpegError(
                f"{name} not found (looked for '{binary}'). Install ffmpeg and make sure it is on PATH, "
                f"or set MOVIEQA_{name.upper()} to its full path."
            )

# ------------------------------------------------------------------ probe
def probe(path: Path) -> MediaInfo:
    cmd = [
        settings.ffprobe_bin, "-v", "error",
        "-print_format", "json", "-show_format", "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {proc.stderr.strip()}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    # Main video stream = first video stream that is not cover art.
    video = next(
        (s for s in streams
         if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")),
        None,
    )
    if video is None:
        raise FFmpegError(f"No video stream found in {path}")
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    subs = [s for s in streams if s.get("codec_type") == "subtitle"]

    duration_s = fmt.get("duration") or video.get("duration") or 0
    return MediaInfo(
        duration_ms=int(float(duration_s) * 1000),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        video_codec=video.get("codec_name", ""),
        audio_tracks=len(audio),
        audio_codec=audio[0].get("codec_name") if audio else None,
        subtitle_tracks=[
            SubtitleTrack(
                index=int(s["index"]),
                codec=s.get("codec_name", ""),
                language=(s.get("tags") or {}).get("language"),
                is_text=s.get("codec_name", "") in TEXT_SUBTITLE_CODECS,
            )
            for s in subs
        ],
        container=fmt.get("format_name", ""),
        size_bytes=int(fmt.get("size") or 0),
    )

# ------------------------------------------------------------------ outputs
def make_proxy(src: Path, dst: Path, *, height: int, crf: int, preset: str,
               source_height: int, duration_ms: int, on_progress: ProgressFn | None = None) -> None:
    """Low-res, video-only copy used for shot detection and frame grabs."""
    target_h = min(height, source_height) if source_height else height
    target_h -= target_h % 2                       # H.264 needs even dimensions
    args = [
        "-i", str(src),
        "-map", "0:V:0",                           # main video, skip cover art
        "-vf", f"scale=-2:{target_h}",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an", "-sn", "-dn",
        "-movflags", "+faststart",
    ]
    _run_to_file(args, dst, duration_ms, on_progress)

def extract_audio(src: Path, dst: Path, *, sample_rate: int,
                  duration_ms: int, on_progress: ProgressFn | None = None) -> None:
    """16 kHz mono 16-bit WAV — the input format Whisper expects."""
    args = [
        "-i", str(src),
        "-map", "0:a:0",
        "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
    ]
    _run_to_file(args, dst, duration_ms, on_progress)

def extract_subtitles(src: Path, dst: Path, stream_index: int) -> None:
    args = ["-i", str(src), "-map", f"0:{stream_index}", "-c:s", "srt"]
    _run_to_file(args, dst, duration_ms=0, on_progress=None)

# ------------------------------------------------------------------ internals
def _run_to_file(args: list[str], dst: Path, duration_ms: int,
                 on_progress: ProgressFn | None) -> None:
    """Run ffmpeg writing to a temp file, then atomically rename to dst.

    A crash or Ctrl+C can therefore never leave a half-written dst behind.
    """
    dst = Path(dst)
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")   # keep extension so ffmpeg picks the format
    cmd = [settings.ffmpeg_bin, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
           "-progress", "pipe:1", "-nostats", *args, str(tmp)]

    with tempfile.TemporaryFile(mode="w+") as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if on_progress and duration_ms > 0 and line.startswith("out_time_us="):
                    value = line.split("=", 1)[1].strip()
                    if value.isdigit():
                        on_progress(min(int(value) / 1000 / duration_ms, 1.0))
            proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            tmp.unlink(missing_ok=True)
            raise
        if proc.returncode != 0:
            err.seek(0)
            tmp.unlink(missing_ok=True)
            raise FFmpegError(f"ffmpeg failed ({' '.join(args[:4])} ...): {err.read().strip()[-2000:]}")

    os.replace(tmp, dst)
    if on_progress:
        on_progress(1.0)
