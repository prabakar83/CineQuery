"""Stage 1 — Normalize.

Input : data/videos/{video_id}/original.<ext>   (put there by cli.py or the upload API)
Output: data/videos/{video_id}/proxy.mp4         480p, video only  -> shot detection, frames
        data/videos/{video_id}/audio.wav         16 kHz mono       -> transcription
        data/videos/{video_id}/subtitles.srt     if the file had text subtitles
        data/videos/{video_id}/metadata.json     media info + stage status
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from app.adapters.ffmpeg import FFmpeg, MediaInfo, SubtitleTrack
from app.adapters.local_storage import VideoStore
from app.config import settings
from app.pipeline.base import ProgressCallback, StageError, no_progress

log = logging.getLogger(__name__)

PREFERRED_SUB_LANGS = ("en", "eng")


def _pick_subtitle(tracks: list[SubtitleTrack]) -> Optional[SubtitleTrack]:
    text_tracks = [t for t in tracks if t.is_text]
    if not text_tracks:
        return None
    for t in text_tracks:
        if (t.language or "").lower() in PREFERRED_SUB_LANGS:
            return t
    return text_tracks[0]


class NormalizeStage:
    name = "normalize"

    def __init__(self, store: VideoStore | None = None, ffmpeg: FFmpeg | None = None,
                 progress: ProgressCallback = no_progress) -> None:
        # Initialize the store and ffmpeg objects
        # If no store or ffmpeg is provided, use the default ones
        # The progress callback is used to update the progress of the stage
        self.store = store or VideoStore()
        self.ffmpeg = ffmpeg or FFmpeg()
        self.progress = progress

    def is_done(self, video_id: str) -> bool:
        # Check if the stage is done by checking the status of the stage
        # and the outputs of the stage
        # If the stage is done, return True
        # If the stage is not done, return False
        meta = self.store.read_metadata(video_id)
        status = meta.get("stages", {}).get(self.name, {}).get("status")
        return status == "done"

    def run(self, video_id: str, chunk_id: Optional[str] = None, *, force: bool = False) -> None:
        if not force and self.is_done(video_id):
            log.info("[%s] %s already done, skipping", video_id, self.name)
            self.progress(self.name, 1.0, "already done")
            return

        # Get the paths for the video
        # If the original file is not found, raise an error
        # The original file is the file that was uploaded to the server
        paths = self.store.paths(video_id)
        original = paths.original()
        if original is None:
            raise StageError(f"No original file in {paths.root}. Run `python cli.py ingest <file>` first.")

        self.ffmpeg.check_installed()
        started = time.monotonic()
        self._set_status(video_id, "running")

        try:
            # 1) Probe -------------------------------------------------------
            info: MediaInfo = self.ffmpeg.probe(original)
            if info.duration_ms <= 0:
                raise StageError(f"Could not read a duration from {original.name}; the file may be damaged.")
            log.info("[%s] %s: %s, %dx%d @ %.3f fps, %d audio, %d subtitle tracks",
                     video_id, original.name, _fmt_ms(info.duration_ms), info.width, info.height,
                     info.fps, info.audio_tracks, len(info.subtitle_tracks))
            self.store.update_metadata(video_id, original_file=original.name, media=info.to_dict())

            # 2) Proxy video (0% -> 80% of the stage) --------------------------
            self.ffmpeg.make_proxy(
                original, paths.proxy,
                height=settings.proxy_height, crf=settings.proxy_crf, preset=settings.proxy_preset,
                source_height=info.height, duration_ms=info.duration_ms,
                on_progress=lambda f: self.progress(self.name, 0.80 * f, "creating 480p proxy"),
            )

            # 3) Audio (80% -> 95%) -------------------------------------------
            if info.has_audio:
                self.ffmpeg.extract_audio(
                    original, paths.audio,
                    sample_rate=settings.audio_sample_rate, duration_ms=info.duration_ms,
                    on_progress=lambda f: self.progress(self.name, 0.80 + 0.15 * f, "extracting audio"),
                )
            else:
                log.warning("[%s] no audio track — transcription will be skipped", video_id)

            # 4) Subtitles (best effort, never fails the stage) ----------------
            sub = _pick_subtitle(info.subtitle_tracks)
            subtitles_file = None
            if sub is not None:
                self.progress(self.name, 0.96, "extracting subtitles")
                try:
                    self.ffmpeg.extract_subtitles(original, paths.subtitles, sub.index)
                    subtitles_file = paths.subtitles.name
                except Exception as exc:  # noqa: BLE001 — subtitles are a bonus
                    log.warning("[%s] subtitle extraction failed: %s", video_id, exc)
            elif info.subtitle_tracks:
                log.info("[%s] only image-based subtitles found; skipping", video_id)

            # 5) Record outputs -------------------------------------------------
            self.store.update_metadata(
                video_id,
                outputs={
                    "proxy": paths.proxy.name,
                    "audio": paths.audio.name if info.has_audio else None,
                    "subtitles": subtitles_file,
                },
            )
        except BaseException as exc:
            self._set_status(video_id, "failed", error=str(exc) or exc.__class__.__name__)
            raise

        elapsed = round(time.monotonic() - started, 1)
        self._set_status(video_id, "done", seconds=elapsed)
        self.progress(self.name, 1.0, f"done in {elapsed}s")

    # ------------------------------------------------------------------
    def _set_status(self, video_id: str, status: str, **extra) -> None:
        meta = self.store.read_metadata(video_id)
        stages = meta.setdefault("stages", {})
        stages[self.name] = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **extra,
        }
        self.store.write_metadata(video_id, meta)


def _fmt_ms(ms: int) -> str:
    s = ms // 1000
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
