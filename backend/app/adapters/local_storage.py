"""Local-disk storage for video files (plain functions).

This is the ONLY place that knows the on-disk layout:

    data/videos/{video_id}/
        original.<ext>     untouched copy of the input file
        proxy.mp4          480p, no audio            (stage 1)
        audio.wav          16 kHz mono               (stage 1)
        subtitles.srt      only if text subtitles    (stage 1)
        chunks.json        time ranges per chunk     (stage 2)
        transcripts/       c000.json, c001.json ...  (stage 3)
        shots/             c000.json, c001.json ...  (stage 4)
        frames/            000304075.jpg ...         (stage 4 keyframes, named by ms)
        captions/          c000.json, c001.json ...  (stage 5)
        timeline.json      dialogue + on-screen text + visuals per shot (stage 6)
        scenes.json        scenes: description, characters, mood, tags (stage 7)
        metadata.json      media info + stage status

Stages ask these functions for paths instead of building them by hand,
so changing the layout (or moving to S3 / Postgres) touches one file.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import settings

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# ---------------------------------------------------------------- paths
@dataclass(frozen=True)
class VideoPaths:
    """All file locations for one video. Pure data, no behaviour except finding the original."""
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
    def chunks(self) -> Path:
        return self.root / "chunks.json"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def transcripts_dir(self) -> Path:
        return self.root / "transcripts"

    def transcript(self, chunk_id: str) -> Path:
        """transcripts/c000.json — the transcript of one chunk (stage 3)."""
        return self.transcripts_dir / f"{chunk_id}.json"

    @property
    def shots_dir(self) -> Path:
        return self.root / "shots"

    def shots(self, chunk_id: str) -> Path:
        """shots/c000.json — the shot list of one chunk (stage 4)."""
        return self.shots_dir / f"{chunk_id}.json"

    @property
    def captions_dir(self) -> Path:
        return self.root / "captions"

    def captions(self, chunk_id: str) -> Path:
        """captions/c000.json — keyframe descriptions of one chunk (stage 5)."""
        return self.captions_dir / f"{chunk_id}.json"

    @property
    def timeline(self) -> Path:
        """timeline.json — everything merged on one movie clock (stage 6)."""
        return self.root / "timeline.json"

    @property
    def scenes(self) -> Path:
        """scenes.json — scenes with LLM descriptions and tags (stage 7)."""
        return self.root / "scenes.json"

    def frame(self, ts_ms: int) -> Path:
        """frames/000304075.jpg — keyframe at that movie time (stage 4). Zero-padded so files sort by time."""
        return self.frames_dir / f"{ts_ms:09d}.jpg"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    def original(self) -> Optional[Path]:
        """The original file, whatever its extension (original.mp4, original.mkv, ...)."""
        candidates = [p for p in self.root.glob("original.*") if ".tmp" not in p.suffixes]
        return candidates[0] if candidates else None


def new_video_id() -> str:
    return f"vid_{uuid.uuid4().hex[:8]}"


def video_paths(video_id: str) -> VideoPaths:
    if not _VIDEO_ID_RE.match(video_id):
        raise ValueError(f"Invalid video_id: {video_id!r}")
    return VideoPaths(settings.videos_dir / video_id)


def video_exists(video_id: str) -> bool:
    return video_paths(video_id).root.is_dir()


def list_video_ids() -> list[str]:
    if not settings.videos_dir.is_dir():
        return []
    return sorted(p.name for p in settings.videos_dir.iterdir() if p.is_dir())


# ---------------------------------------------------------------- import
def import_original(src: Path, video_id: str) -> Path:
    """Put the source file into the video folder as original.<ext>.

    Tries a hard link first (instant, no extra disk space on the same drive)
    and falls back to a copy. Writes via a temp name so a half-copied file is
    never treated as complete.
    """
    src = Path(src).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Input file not found: {src}")

    paths = video_paths(video_id)
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


# ---------------------------------------------------------------- json helpers
def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic write: temp file + rename, so a crash never leaves half a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- metadata.json
def read_metadata(video_id: str) -> dict[str, Any]:
    return read_json(video_paths(video_id).metadata)


def write_metadata(video_id: str, data: dict[str, Any]) -> None:
    write_json(video_paths(video_id).metadata, data)


def update_metadata(video_id: str, **fields: Any) -> dict[str, Any]:
    data = read_metadata(video_id)
    data.update(fields)
    write_metadata(video_id, data)
    return data


def get_stage_status(video_id: str, stage: str) -> Optional[str]:
    return read_metadata(video_id).get("stages", {}).get(stage, {}).get("status")


def set_stage_status(video_id: str, stage: str, status: str, **extra: Any) -> None:
    """Record running / done / failed for a whole-video stage in metadata.json."""
    data = read_metadata(video_id)
    data.setdefault("stages", {})[stage] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    write_metadata(video_id, data)


# ---------------------------------------------------------------- chunks.json
def read_chunks(video_id: str) -> dict[str, Any]:
    return read_json(video_paths(video_id).chunks)


def write_chunks(video_id: str, data: dict[str, Any]) -> None:
    write_json(video_paths(video_id).chunks, data)


def set_chunk_stage_status(video_id: str, chunk_id: str, stage: str, status: str) -> None:
    """Record e.g. transcribe=done for one chunk (used by stages 3–5)."""
    data = read_chunks(video_id)
    for chunk in data.get("chunks", []):
        if chunk["id"] == chunk_id:
            chunk.setdefault("stages", {})[stage] = status
            write_chunks(video_id, data)
            return
    raise KeyError(f"Chunk {chunk_id!r} not found for video {video_id!r}")
