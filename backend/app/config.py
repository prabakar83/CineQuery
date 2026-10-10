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
    whisper_segment_langid: bool = os.environ.get("MOVIEQA_WHISPER_SEGMENT_LANGID", "1") != "0"  # per-line language check

    # Shots stage (PySceneDetect + keyframes)
    shot_threshold: float = float(os.environ.get("MOVIEQA_SHOT_THRESHOLD", 20.0))     # lower = more cuts (27 missed cuts in dark scenes)
    min_shot_ms: int = int(os.environ.get("MOVIEQA_MIN_SHOT_MS", 500))                # shorter "shots" are merged
    keyframe_every_ms: int = int(os.environ.get("MOVIEQA_KEYFRAME_EVERY_MS", 8_000))  # extra frames in long shots
    max_keyframes_per_shot: int = int(os.environ.get("MOVIEQA_MAX_KEYFRAMES", 6))
    keyframe_jpeg_quality: int = 3                                                     # ffmpeg -q:v (2 best .. 31 worst)

    # Caption stage
    caption_engine: str = os.environ.get("MOVIEQA_CAPTION_ENGINE", "florence")      # "florence" (fast, 4 GB GPU) or "ollama"
    florence_model: str = os.environ.get("MOVIEQA_FLORENCE_MODEL", "florence-community/Florence-2-large")  # or ...-base (faster)
    florence_device: str = os.environ.get("MOVIEQA_FLORENCE_DEVICE", "auto")         # auto -> cuda if available
    florence_objects: bool = os.environ.get("MOVIEQA_FLORENCE_OBJECTS", "1") != "0"  # object labels; batched run incl. objects = 1.8 s/frame on RTX 3050
    florence_caption_beams: int = int(os.environ.get("MOVIEQA_FLORENCE_BEAMS", 1))   # 3 = slightly better, ~20% slower
    caption_batch_size: int = int(os.environ.get("MOVIEQA_CAPTION_BATCH", 4))        # frames per GPU call (Florence only)
    # Merge stage (thresholds for marking Whisper lines as unreliable, e.g. made-up languages)
    dialogue_min_lang_prob: float = float(os.environ.get("MOVIEQA_DIALOGUE_MIN_LANG_PROB", 0.5))   # main signal
    dialogue_subtitled_min_lang_prob: float = float(os.environ.get("MOVIEQA_DIALOGUE_SUBTITLED_MIN_LANG_PROB", 0.75))  # stricter while subtitles show
    dialogue_min_logprob: float = float(os.environ.get("MOVIEQA_DIALOGUE_MIN_LOGPROB", -1.2))      # backup signals
    dialogue_min_word_prob: float = float(os.environ.get("MOVIEQA_DIALOGUE_MIN_WORD_PROB", 0.35))
    ocr_same_text_ratio: float = 0.8      # neighbouring frames with >= this text similarity = same subtitle

    # Scenes stage (shot grouping + text LLM via Ollama)
    scene_llm_model: str = os.environ.get("MOVIEQA_SCENE_LLM", "qwen2.5:3b")          # ~2 GB, fits a 4 GB GPU
    scene_cut_threshold: float = float(os.environ.get("MOVIEQA_SCENE_CUT", 0.5))       # higher = fewer, longer scenes
    scene_min_ms: int = int(os.environ.get("MOVIEQA_SCENE_MIN_MS", 15_000))
    scene_max_ms: int = int(os.environ.get("MOVIEQA_SCENE_MAX_MS", 180_000))
    scene_max_visuals: int = 12              # visual notes sent to the LLM per scene
    scene_llm_ctx: int = 4096                # context window (tokens)

    # (ollama engine) vision model served by Ollama
    ollama_url: str = os.environ.get("MOVIEQA_OLLAMA_URL", "http://localhost:11434")
    vision_model: str = os.environ.get("MOVIEQA_VISION_MODEL", "qwen2.5vl:3b")        # 3b fits a 4 GB GPU
    vision_timeout_s: int = int(os.environ.get("MOVIEQA_VISION_TIMEOUT_S", 180))      # first call also loads the model
    vision_retries: int = int(os.environ.get("MOVIEQA_VISION_RETRIES", 2))            # per frame, on bad JSON / timeouts

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"


settings = Settings()
