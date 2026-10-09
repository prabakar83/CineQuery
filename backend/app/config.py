"""Central settings for the backend.

Every value can be overridden with an environment variable, so the same code
runs on your laptop, in Docker, and on Kaggle without edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# backend/app/config.py -> parents[2] is the project root (Movie-QA/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    # Where all per-video folders live: data/videos/{video_id}/
    data_dir: Path = field(default_factory=lambda: _env_path("MOVIEQA_DATA_DIR", PROJECT_ROOT / "data"))

    # External binaries
    ffmpeg_bin: str = field(default_factory=lambda: os.environ.get("MOVIEQA_FFMPEG", "ffmpeg"))
    ffprobe_bin: str = field(default_factory=lambda: os.environ.get("MOVIEQA_FFPROBE", "ffprobe"))

    # Normalize stage
    proxy_height: int = int(os.environ.get("MOVIEQA_PROXY_HEIGHT", 480))
    proxy_crf: int = int(os.environ.get("MOVIEQA_PROXY_CRF", 28))          # higher = smaller/worse
    proxy_preset: str = os.environ.get("MOVIEQA_PROXY_PRESET", "veryfast")
    audio_sample_rate: int = 16_000                                        # what Whisper expects

    # Chunk stage
    chunk_ms: int = int(os.environ.get("MOVIEQA_CHUNK_MS", 5 * 60 * 1000))       # 5 minutes
    chunk_overlap_ms: int = int(os.environ.get("MOVIEQA_CHUNK_OVERLAP_MS", 5_000))  # 5 seconds each side
    min_last_chunk_ms: int = 60_000      # a shorter final chunk is merged into the previous one

    # Transcribe stage (faster-whisper)
    whisper_model: str = os.environ.get("MOVIEQA_WHISPER_MODEL", "small")      # tiny/base/small/medium/large-v3
    whisper_device: str = os.environ.get("MOVIEQA_WHISPER_DEVICE", "auto")     # auto -> cuda if available, else cpu
    whisper_compute_type: str = os.environ.get("MOVIEQA_WHISPER_COMPUTE", "auto")  # auto -> float16 on cuda, int8 on cpu
    whisper_beam_size: int = int(os.environ.get("MOVIEQA_WHISPER_BEAM", 5))
    whisper_language: Optional[str] = os.environ.get("MOVIEQA_WHISPER_LANGUAGE") or None  # None = auto-detect

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"


settings = Settings()
