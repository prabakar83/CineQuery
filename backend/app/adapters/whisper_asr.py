"""Speech-to-text with faster-whisper.

This is a CLASS (unlike ffmpeg/storage) because it holds a loaded model
(hundreds of MB in memory). Load it once, then call transcribe() for every chunk.
To try another ASR engine later, write a second class with the same
transcribe() method — the stage code doesn't change.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@dataclass
class Word:
    start_ms: int
    end_ms: int
    word: str
    prob: float


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Transcription:
    language: str
    language_prob: float
    segments: list[Segment]


class WhisperASR:
    def __init__(self, model_size: str = "small", device: str = "auto",
                 compute_type: str = "auto", beam_size: int = 5) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # clearer than a raw traceback
            raise RuntimeError(
                "faster-whisper is not installed in this Python. Activate the project venv "
                "(source .venv/bin/activate) and run: pip install -r requirements.txt"
            ) from exc

        self.device = _resolve_device(device)
        self.compute_type = compute_type if compute_type != "auto" else (
            "float16" if self.device == "cuda" else "int8")
        self.model_size = model_size
        self.beam_size = beam_size

        log.info("loading whisper '%s' on %s (%s) — first run downloads the model",
                 model_size, self.device, self.compute_type)
        self.model = WhisperModel(
            model_size, device=self.device, compute_type=self.compute_type,
            cpu_threads=os.cpu_count() or 4,
        )

    def transcribe(self, audio: np.ndarray, *, language: Optional[str] = None, offset_ms: int = 0,
                   on_progress: Optional[Callable[[float], None]] = None) -> Transcription:
        """Transcribe float32 16 kHz mono samples.

        offset_ms is added to every timestamp, so results come back in MOVIE time
        rather than time-within-this-slice.
        """
        if audio.size == 0:
            return Transcription(language or "", 0.0, [])

        segments_iter, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            vad_filter=True,                                   # skip silence/music -> fewer made-up lines
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=True,
            condition_on_previous_text=False,                  # avoids repetition loops on long audio
        )
        duration_s = max(info.duration, 1e-6)

        segments: list[Segment] = []
        for seg in segments_iter:                              # generator: work happens while iterating
            segments.append(Segment(
                start_ms=offset_ms + int(seg.start * 1000),
                end_ms=offset_ms + int(seg.end * 1000),
                text=seg.text.strip(),
                avg_logprob=round(seg.avg_logprob, 3),
                no_speech_prob=round(seg.no_speech_prob, 3),
                words=[
                    Word(offset_ms + int(w.start * 1000), offset_ms + int(w.end * 1000),
                         w.word.strip(), round(w.probability, 3))
                    for w in (seg.words or [])
                ],
            ))
            if on_progress:
                on_progress(min(seg.end / duration_s, 1.0))

        if on_progress:
            on_progress(1.0)
        return Transcription(info.language, round(info.language_probability, 3), segments)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"
