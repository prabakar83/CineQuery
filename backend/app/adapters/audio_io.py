"""Read slices of the 16 kHz mono WAV produced by stage 1 (plain functions).

Reading a slice straight from the WAV is instant and needs no temp files,
so chunks never have to be cut into separate audio files.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        return int(w.getnframes() * 1000 / w.getframerate())


def read_wav_slice(path: Path, start_ms: int, end_ms: int, expected_rate: int = 16_000) -> np.ndarray:
    """Return samples in [start_ms, end_ms) as float32 in -1.0..1.0 (the format Whisper takes)."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if rate != expected_rate or channels != 1 or width != 2:
            raise ValueError(
                f"{path.name} must be {expected_rate} Hz mono 16-bit, got {rate} Hz, "
                f"{channels} channel(s), {width * 8}-bit. Re-run normalize with --force."
            )
        total = w.getnframes()
        first = max(0, min(total, start_ms * rate // 1000))
        last = max(first, min(total, end_ms * rate // 1000))
        w.setpos(first)
        raw = w.readframes(last - first)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
