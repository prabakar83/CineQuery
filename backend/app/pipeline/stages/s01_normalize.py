"""Stage 1 — Normalize.

Input : data/videos/{video_id}/original.<ext>   (put there by cli.py or the upload API)
Output: data/videos/{video_id}/proxy.mp4         480p, video only  -> shot detection, frames
        data/videos/{video_id}/audio.wav         16 kHz mono       -> transcription
        data/videos/{video_id}/subtitles.srt     if the file had text subtitles
        data/videos/{video_id}/metadata.json     media info + stage status
"""
from __future__ import annotations

import logging
from typing import Optional

from app.adapters import ffmpeg
from app.adapters import local_storage as storage
from app.adapters.ffmpeg import SubtitleTrack
from app.config import settings
from app.pipeline.base import ProgressCallback, StageError, no_progress, track_stage

log = logging.getLogger(__name__)

NAME = "normalize"
PREFERRED_SUB_LANGS = ("en", "eng")


def is_done(video_id: str) -> bool:
    # The stage is done when metadata.json says so.
    # (If you delete proxy.mp4 / audio.wav by hand, re-run with --force.)
    return storage.get_stage_status(video_id, NAME) == "done"


def run(video_id: str, *, force: bool = False, progress: ProgressCallback = no_progress) -> None:
    if not force and is_done(video_id):
        log.info("[%s] %s already done, skipping", video_id, NAME)
        progress(NAME, 1.0, "already done")
        return

    # Find the original file that cli.py / the upload API placed in the video folder.
    # If it's missing, stop with a clear message.
    paths = storage.video_paths(video_id)
    original = paths.original()
    if original is None:
        raise StageError(f"No original file in {paths.root}. Run `python cli.py ingest <file>` first.")

    ffmpeg.check_installed()

    with track_stage(video_id, NAME):  # marks running -> done / failed in metadata.json
        # 1) Probe: read duration, resolution, fps, tracks ---------------------
        info = ffmpeg.probe(original)
        if info.duration_ms <= 0:
            raise StageError(f"Could not read a duration from {original.name}; the file may be damaged.")
        log.info("[%s] %s: %s, %dx%d @ %.3f fps, %d audio, %d subtitle tracks",
                 video_id, original.name, _fmt_ms(info.duration_ms), info.width, info.height,
                 info.fps, info.audio_tracks, len(info.subtitle_tracks))
        storage.update_metadata(video_id, original_file=original.name, media=info.to_dict())

        # 2) Proxy video (0% -> 80% of the stage) ------------------------------
        ffmpeg.make_proxy(
            original, paths.proxy,
            height=settings.proxy_height, crf=settings.proxy_crf, preset=settings.proxy_preset,
            source_height=info.height, duration_ms=info.duration_ms,
            on_progress=lambda f: progress(NAME, 0.80 * f, "creating 480p proxy"),
        )

        # 3) Audio (80% -> 95%) -------------------------------------------------
        if info.has_audio:
            ffmpeg.extract_audio(
                original, paths.audio,
                sample_rate=settings.audio_sample_rate, duration_ms=info.duration_ms,
                on_progress=lambda f: progress(NAME, 0.80 + 0.15 * f, "extracting audio"),
            )
        else:
            log.warning("[%s] no audio track — transcription will be skipped", video_id)

        # 4) Subtitles (best effort, never fails the stage) ---------------------
        subtitles_file = _extract_subtitles(video_id, original, info.subtitle_tracks, paths.subtitles, progress)

        # 5) Record outputs -----------------------------------------------------
        storage.update_metadata(
            video_id,
            outputs={
                "proxy": paths.proxy.name,
                "audio": paths.audio.name if info.has_audio else None,
                "subtitles": subtitles_file,
            },
        )

    progress(NAME, 1.0, "done")


# ---------------------------------------------------------------- helpers
def _pick_subtitle(tracks: list[SubtitleTrack]) -> Optional[SubtitleTrack]:
    """Prefer an English text track, else the first text track. Image subtitles can't become .srt."""
    text_tracks = [t for t in tracks if t.is_text]
    if not text_tracks:
        return None
    for t in text_tracks:
        if (t.language or "").lower() in PREFERRED_SUB_LANGS:
            return t
    return text_tracks[0]


def _extract_subtitles(video_id, original, tracks, dst, progress) -> Optional[str]:
    sub = _pick_subtitle(tracks)
    if sub is None:
        if tracks:
            log.info("[%s] only image-based subtitles found; skipping", video_id)
        return None
    progress(NAME, 0.96, "extracting subtitles")
    try:
        ffmpeg.extract_subtitles(original, dst, sub.index)
        return dst.name
    except Exception as exc:  # noqa: BLE001 — subtitles are a bonus
        log.warning("[%s] subtitle extraction failed: %s", video_id, exc)
        return None


def _fmt_ms(ms: int) -> str:
    s = ms // 1000
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
