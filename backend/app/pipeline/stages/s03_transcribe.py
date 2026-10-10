"""Stage 3 — Transcribe (per chunk).

Input : data/videos/{id}/audio.wav     (stage 1)
        data/videos/{id}/chunks.json   (stage 2)
Output: data/videos/{id}/transcripts/c000.json, c001.json, ...  one file per chunk
        chunks.json  -> chunks[i].stages.transcribe = "done"
        metadata.json -> language, transcript summary, stages.transcribe

For each chunk we read only its padded time range from audio.wav, run Whisper,
and shift every timestamp by the chunk's start so everything is in MOVIE time.
Segments inside the overlap appear in two chunk files on purpose; load_transcript()
(and later the merge stage) keeps each one only in the chunk whose CORE range
contains its midpoint.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.adapters import local_storage as storage
from app.config import settings
from app.pipeline.base import ProgressCallback, no_progress, require_stage_done, track_stage

log = logging.getLogger(__name__)

NAME = "transcribe"

# Classic Whisper hallucination filter: "probably not speech" AND "low confidence".
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0
# Only pin the detected language once Whisper is reasonably sure.
LANGUAGE_PIN_MIN_PROB = 0.5


def is_done(video_id: str) -> bool:
    return storage.get_stage_status(video_id, NAME) == "done"


def run(video_id: str, *, force: bool = False, progress: ProgressCallback = no_progress) -> None:
    if not force and is_done(video_id):
        log.info("[%s] %s already done, skipping", video_id, NAME)
        progress(NAME, 1.0, "already done")
        return

    require_stage_done(video_id, "normalize")
    require_stage_done(video_id, "chunk")

    meta = storage.read_metadata(video_id)
    paths = storage.video_paths(video_id)

    with track_stage(video_id, NAME):
        # A video without sound has nothing to transcribe: finish cleanly.
        if not meta.get("outputs", {}).get("audio") or not paths.audio.is_file():
            log.warning("[%s] no audio.wav — skipping transcription", video_id)
            storage.update_metadata(video_id, transcript={"skipped": "no audio"})
            progress(NAME, 1.0, "no audio, skipped")
            return

        chunks = storage.read_chunks(video_id)["chunks"]
        todo = [c for c in chunks if force or not _chunk_done(paths, c)]
        log.info("[%s] %d/%d chunk(s) to transcribe", video_id, len(todo), len(chunks))

        if todo:
            # Imported here, not at the top: these need numpy/faster-whisper (the .venv),
            # while the other CLI commands still work with plain system Python.
            from app.adapters import audio_io
            from app.adapters.whisper_asr import WhisperASR
            progress(NAME, 0.0, f"loading whisper {settings.whisper_model}")
            asr = WhisperASR(settings.whisper_model, settings.whisper_device,
                             settings.whisper_compute_type, settings.whisper_beam_size,
                             segment_langid=settings.whisper_segment_langid)

            language: Optional[str] = settings.whisper_language or meta.get("language")
            for n, chunk in enumerate(todo):
                label = f"chunk {chunk['index'] + 1}/{len(chunks)}"
                audio = audio_io.read_wav_slice(paths.audio, chunk["start_ms"], chunk["end_ms"],
                                                expected_rate=settings.audio_sample_rate)
                result = asr.transcribe(
                    audio, language=language, offset_ms=chunk["start_ms"],
                    on_progress=lambda f, n=n, label=label: progress(NAME, (n + f) / len(todo), label),
                )

                # Pin the language after the first confident detection, so all chunks agree.
                if language is None and result.segments and result.language_prob >= LANGUAGE_PIN_MIN_PROB:
                    language = result.language
                    storage.update_metadata(video_id, language=language)
                    log.info("[%s] language detected: %s (p=%.2f)", video_id, language, result.language_prob)

                kept = [s for s in result.segments if s.text and not _looks_hallucinated(s)]
                storage.write_json(paths.transcript(chunk["id"]), {
                    "chunk_id": chunk["id"],
                    "start_ms": chunk["start_ms"],
                    "end_ms": chunk["end_ms"],
                    "language": result.language,
                    "language_prob": result.language_prob,
                    "model": settings.whisper_model,
                    "dropped_segments": len(result.segments) - len(kept),
                    "segments": [s.to_dict() for s in kept],
                })
                storage.set_chunk_stage_status(video_id, chunk["id"], NAME, "done")
                log.info("[%s] %s: %d segment(s)", video_id, chunk["id"], len(kept))

        # Summary for metadata.json (de-duplicated across chunk overlaps)
        segments = load_transcript(video_id)
        storage.update_metadata(video_id, transcript={
            "model": settings.whisper_model,
            "language": storage.read_metadata(video_id).get("language"),
            "segments": len(segments),
            "words": sum(len(s["words"]) for s in segments),
        })

    progress(NAME, 1.0, "done")


# ---------------------------------------------------------------- reading results
def load_transcript(video_id: str) -> list[dict[str, Any]]:
    """All segments of the movie in time order, with overlap duplicates removed.

    Rule: a segment belongs to the chunk whose CORE range contains its midpoint.
    """
    paths = storage.video_paths(video_id)
    chunks = storage.read_chunks(video_id).get("chunks", [])
    result: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        data = storage.read_json(paths.transcript(chunk["id"]))
        for seg in data.get("segments", []):
            mid = (seg["start_ms"] + seg["end_ms"]) // 2
            in_core = chunk["core_start_ms"] <= mid < chunk["core_end_ms"]
            if in_core or (is_last and mid >= chunk["core_end_ms"]):   # last chunk owns the very end
                result.append(seg)
    result.sort(key=lambda s: s["start_ms"])
    return result


# ---------------------------------------------------------------- helpers
def _chunk_done(paths: storage.VideoPaths, chunk: dict[str, Any]) -> bool:
    return chunk.get("stages", {}).get(NAME) == "done" and paths.transcript(chunk["id"]).is_file()


def _looks_hallucinated(seg) -> bool:
    return seg.no_speech_prob > NO_SPEECH_THRESHOLD and seg.avg_logprob < LOGPROB_THRESHOLD
