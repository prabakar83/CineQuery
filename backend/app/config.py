"""Central settings for the backend.

Every value can be overridden with an environment variable, so the same code
runs on your laptop, in Docker, and on Kaggle without edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

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

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"


settings = Settings()
