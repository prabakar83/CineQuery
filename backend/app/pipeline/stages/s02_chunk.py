"""Stage 2 — Chunk.

Splits the movie's timeline into fixed windows (default 5 min) with a small
overlap (default 5 s each side). No files are cut: later stages read only
their slice of audio.wav / proxy.mp4 using these time ranges.

Input : metadata.json -> media.duration_ms   (written by stage 1)
Output: data/videos/{video_id}/chunks.json

Each chunk has two ranges:
    core   [core_start_ms, core_end_ms)  the part this chunk OWNS; cores tile the
                                         timeline end to end with no gaps
    padded [start_ms, end_ms)            core +/- overlap; what is actually processed,
                                         so a sentence crossing a boundary isn't cut
The merge stage later keeps a result only if its midpoint falls in the core
range of the chunk that produced it, so overlaps are never counted twice.
"""
from __future__ import annotations

import logging
from typing import Any

from app.adapters import local_storage as storage
from app.config import settings
from app.pipeline.base import ProgressCallback, StageError, no_progress, require_stage_done, track_stage

log = logging.getLogger(__name__)

NAME = "chunk"


def plan_chunks(duration_ms: int, chunk_ms: int, overlap_ms: int, min_last_chunk_ms: int) -> list[dict[str, Any]]:
    """Pure function: compute chunk ranges. No I/O, easy to unit-test."""
    if duration_ms <= 0:
        raise ValueError("duration_ms must be > 0")
    if chunk_ms <= 0 or overlap_ms < 0:
        raise ValueError("chunk_ms must be > 0 and overlap_ms >= 0")

    # 1) Core ranges that tile the timeline
    cores: list[list[int]] = []
    start = 0
    while start < duration_ms:
        end = min(start + chunk_ms, duration_ms)
        cores.append([start, end])
        start = end

    # 2) Merge a too-short last chunk into the previous one
    if len(cores) > 1 and cores[-1][1] - cores[-1][0] < min_last_chunk_ms:
        cores[-2][1] = cores[-1][1]
        cores.pop()

    # 3) Add overlap padding, clamped to the movie's bounds
    return [
        {
            "id": f"c{i:03d}",
            "index": i,
            "core_start_ms": core_start,
            "core_end_ms": core_end,
            "start_ms": max(0, core_start - overlap_ms),
            "end_ms": min(duration_ms, core_end + overlap_ms),
            "stages": {},
        }
        for i, (core_start, core_end) in enumerate(cores)
    ]


def is_done(video_id: str) -> bool:
    return storage.get_stage_status(video_id, NAME) == "done"


def run(video_id: str, *, force: bool = False, progress: ProgressCallback = no_progress) -> None:
    if not force and is_done(video_id):
        log.info("[%s] %s already done, skipping", video_id, NAME)
        progress(NAME, 1.0, "already done")
        return

    require_stage_done(video_id, "normalize")
    duration_ms = storage.read_metadata(video_id).get("media", {}).get("duration_ms", 0)
    if duration_ms <= 0:
        raise StageError(f"No duration in metadata.json for {video_id}. Re-run normalize with --force.")

    with track_stage(video_id, NAME):
        chunks = plan_chunks(
            duration_ms,
            chunk_ms=settings.chunk_ms,
            overlap_ms=settings.chunk_overlap_ms,
            min_last_chunk_ms=settings.min_last_chunk_ms,
        )

        # On a forced re-run, keep per-chunk progress for chunks whose range didn't change,
        # so finished transcription/caption work isn't thrown away.
        previous = {c["id"]: c for c in storage.read_chunks(video_id).get("chunks", [])}
        for c in chunks:
            old = previous.get(c["id"])
            if old and old["start_ms"] == c["start_ms"] and old["end_ms"] == c["end_ms"]:
                c["stages"] = old.get("stages", {})

        storage.write_chunks(video_id, {
            "duration_ms": duration_ms,
            "chunk_ms": settings.chunk_ms,
            "overlap_ms": settings.chunk_overlap_ms,
            "count": len(chunks),
            "chunks": chunks,
        })
        log.info("[%s] %d chunk(s) of up to %ds with %ds overlap",
                 video_id, len(chunks), settings.chunk_ms // 1000, settings.chunk_overlap_ms // 1000)

    progress(NAME, 1.0, f"{len(chunks)} chunk(s)")
