"""Local-disk storage for video files.

This is the ONLY place that knows the on-disk layout:

    data/videos/{video_id}/
        original.<ext>     untouched copy of the uploaded file
        proxy.mp4          480p, no audio
        audio.wav          16 kHz mono
        subtitles.srt      only if the movie had text subtitles
        metadata.json      media info + per-stage status

Later stages ask this class for paths instead of building them by hand,
so changing the layout (or moving to S3) touches one file.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class VideoPaths:
    root: Path

    @property
    def proxy(self) -> Path:
        return self.root / "proxy.mp4"

    @property
    def audio(self) -> Path:
        return self.root / "audio.wav"

    @property
    def subtitles(self) -> Path:
        return self.root / "subtitles.srt"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    def original(self) -> Path | None:
        """The original file, whatever its extension (original.mp4, original.mkv, ...)."""
        candidates = [p for p in self.root.glob("original.*") if ".tmp" not in p.suffixes]
        return candidates[0] if candidates else None


class VideoStore:
    def __init__(self, videos_dir: Path | None = None) -> None:
        self.videos_dir = videos_dir or settings.videos_dir

    # ---------- ids & paths ----------
    @staticmethod
    def new_video_id() -> str:
        return f"vid_{uuid.uuid4().hex[:8]}"

    def paths(self, video_id: str) -> VideoPaths:
        if not _VIDEO_ID_RE.match(video_id):
            raise ValueError(f"Invalid video_id: {video_id!r}")
        return VideoPaths(self.videos_dir / video_id)

    def exists(self, video_id: str) -> bool:
        return self.paths(video_id).root.is_dir()

    def list_ids(self) -> list[str]:
        if not self.videos_dir.is_dir():
            return []
        return sorted(p.name for p in self.videos_dir.iterdir() if p.is_dir())

    # ---------- import ----------
    def import_original(self, src: Path, video_id: str) -> Path:
        """Put the source file into the video folder as original.<ext>.

        Tries a hard link first (instant, no extra disk space when on the same
        drive) and falls back to a copy. Writes via a temp name so a
        half-copied file is never treated as complete.
        """
        src = Path(src).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Input file not found: {src}")

        paths = self.paths(video_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        existing = paths.original()
        if existing is not None:
            return existing

        ext = src.suffix.lower() or ".bin"
        dst = paths.root / f"original{ext}"
        tmp = paths.root / f"original.tmp{ext}"
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return dst

    # ---------- metadata.json ----------
    def read_metadata(self, video_id: str) -> dict[str, Any]:
        path = self.paths(video_id).metadata
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def write_metadata(self, video_id: str, data: dict[str, Any]) -> None:
        path = self.paths(video_id).metadata
        tmp = path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def update_metadata(self, video_id: str, **fields: Any) -> dict[str, Any]:
        data = self.read_metadata(video_id)
        data.update(fields)
        self.write_metadata(video_id, data)
        return data
