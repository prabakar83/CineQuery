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
    # Spoken-language check on this segment's audio (None if not run):
    lang: Optional[str] = None          # most likely language of the speech, e.g. "en", "ar"
    lang_prob: Optional[float] = None   # probability that it is the TRANSCRIPTION language

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Transcription:
    language: str
    language_prob: float
    segments: list[Segment]


class WhisperASR:
    def __init__(self, model_size: str = "small", device: str = "auto",
                 compute_type: str = "auto", beam_size: int = 5, segment_langid: bool = True) -> None:
        _register_nvidia_dlls()
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # harmless Windows cache warning
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # clearer than a raw traceback
            raise RuntimeError(
                f"faster-whisper could not be imported ({type(exc).__name__}: {exc}). "
                "Check with: python -c \"import faster_whisper\". Activate the project venv and run: "
                "pip install -r requirements.txt"
            ) from exc

        self.device = _resolve_device(device)
        self.compute_type = compute_type if compute_type != "auto" else (
            "float16" if self.device == "cuda" else "int8")
        self.model_size = model_size
        self.beam_size = beam_size
        self.segment_langid = segment_langid

        self._WhisperModel = WhisperModel
        self.model = self._load()

        # On a GPU, the CUDA libraries (cuBLAS / cuDNN) are only loaded when the model
        # first runs. Do a tiny test run now; if the libraries are missing, fall back to
        # CPU instead of crashing halfway through the movie.
        if self.device == "cuda":
            try:
                self._warm_up()
            except Exception as exc:  # noqa: BLE001 — ctranslate2 raises RuntimeError
                if device != "auto":
                    raise
                log.warning("GPU libraries not usable (%s) — falling back to CPU. "
                            "To use the GPU: pip install nvidia-cublas-cu12 \"nvidia-cudnn-cu12==9.*\"", exc)
                self.device, self.compute_type = "cpu", "int8"
                self.model = self._load()

    def _load(self):
        log.info("loading whisper '%s' on %s (%s) — first run downloads the model",
                 self.model_size, self.device, self.compute_type)
        return self._WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type,
            cpu_threads=os.cpu_count() or 4,
        )

    def _warm_up(self) -> None:
        segments, _ = self.model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en", beam_size=1)
        list(segments)   # the generator must be consumed for the model to actually run

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

        if self.segment_langid:
            self._check_segment_languages(audio, segments, offset_ms, language or info.language)

        if on_progress:
            on_progress(1.0)
        return Transcription(info.language, round(info.language_probability, 3), segments)

    def _check_segment_languages(self, audio: np.ndarray, segments: list[Segment],
                                 offset_ms: int, main_language: str) -> None:
        """Ask Whisper which language each segment's AUDIO is in.

        When Whisper is forced to write English, speech in another (or a made-up)
        language still comes out as English-looking words, often with decent
        confidence. Listening to the segment again with language detection catches it:
        e.g. the Fremen lines in Dune score low for "en".
        """
        for seg in segments:
            # Use at least ~2 s of audio around the segment: very short clips are hard to judge.
            mid = (seg.start_ms + seg.end_ms) / 2 - offset_ms
            half = max((seg.end_ms - seg.start_ms) / 2, 1000)
            a = max(0, int((mid - half) * SAMPLE_RATE / 1000))
            b = min(audio.size, int((mid + half) * SAMPLE_RATE / 1000))
            if b - a < SAMPLE_RATE // 4:
                continue
            try:
                top, _top_prob, all_probs = self.model.detect_language(audio=audio[a:b])
            except Exception as exc:  # noqa: BLE001 — optional extra, never fail transcription
                log.debug("language check failed: %s", exc)
                return
            probs = dict(all_probs or [])
            seg.lang = top
            seg.lang_prob = round(float(probs.get(main_language, 0.0)), 3)


def _register_nvidia_dlls() -> None:
    """Windows: let ctranslate2 find cuBLAS/cuDNN installed as pip packages (nvidia-*-cu12).

    Those packages put their DLLs in site-packages/nvidia/<lib>/bin, which Windows
    does not search by default. No-op on macOS/Linux or when they aren't installed.
    """
    if os.name != "nt":
        return
    import site
    import sys
    roots = [*site.getsitepackages(), site.getusersitepackages(), *sys.path]
    for root in roots:
        nvidia = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia):
            continue
        for lib in os.listdir(nvidia):
            bin_dir = os.path.join(nvidia, lib, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        return


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2
        has_gpu = ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return "cpu"
    if has_gpu and not _cuda_libs_available():
        log.warning("NVIDIA GPU found but cuBLAS 12 / cuDNN 9 are not installed — using CPU. "
                    "To use the GPU: pip install nvidia-cublas-cu12 \"nvidia-cudnn-cu12==9.*\"")
        return "cpu"
    return "cuda" if has_gpu else "cpu"


def _cuda_libs_available() -> bool:
    """Check the CUDA libraries faster-whisper needs can be loaded.

    Checked up front because a missing cuDNN can terminate the process outright
    instead of raising a Python error.
    """
    import ctypes
    if os.name == "nt":
        names = ["cublas64_12.dll", "cudnn_ops64_9.dll"]
        loader = ctypes.WinDLL
    else:
        names = ["libcublas.so.12", "libcudnn_ops.so.9"]
        loader = ctypes.CDLL
    for name in names:
        try:
            loader(name)
        except OSError:
            return False
    return True
