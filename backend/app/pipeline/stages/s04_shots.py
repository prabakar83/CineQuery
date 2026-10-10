"""Stage 4 — Shots + keyframes (per chunk).

Input : data/videos/{id}/proxy.mp4     (stage 1)
        data/videos/{id}/chunks.json   (stage 2)
Output: data/videos/{id}/shots/c000.json, ...   shot list per chunk
        data/videos/{id}/frames/000304075.jpg    keyframe images, named by movie ms
        chunks.json   -> chunks[i].stages.shots = "done"
        metadata.json -> shots summary, stages.shots

Per chunk: detect camera cuts in the chunk's padded range, keep the shots whose
midpoint lies in the chunk's CORE range (so overlap shots are kept once), pick
keyframe times (middle of the shot, plus extra frames spread across long shots),
and save those frames as JPGs.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from app.adapters import ffmpeg
from app.adapters import local_storage as storage
from app.config import settings
from app.pipeline.base import ProgressCallback, no_progress, require_stage_done, track_stage

log = logging.getLogger(__name__)

NAME = "shots"


def keyframe_times(start_ms: int, end_ms: int, every_ms: int, max_frames: int) -> list[int]:
    """Pure function: evenly spaced frame times inside a shot.

    1 frame for a short shot (its middle); a long shot gets one frame per `every_ms`,
    capped at `max_frames`. Each frame sits in the middle of its slice of the shot.
    """
    duration = end_ms - start_ms
    if duration <= 0:
        return []
    n = max(1, min(max_frames, math.ceil(duration / every_ms)))
    return [start_ms + int(duration * (i + 0.5) / n) for i in range(n)]


def is_done(video_id: str) -> bool:
    return storage.get_stage_status(video_id, NAME) == "done"


def run(video_id: str, *, force: bool = False, progress: ProgressCallback = no_progress) -> None:
    if not force and is_done(video_id):
        log.info("[%s] %s already done, skipping", video_id, NAME)
        progress(NAME, 1.0, "already done")
        return

    require_stage_done(video_id, "normalize")
    require_stage_done(video_id, "chunk")
    paths = storage.video_paths(video_id)
    ffmpeg.check_installed()

    with track_stage(video_id, NAME):
        # Imported here (needs the .venv); other CLI commands still work with system Python.
        from app.adapters.scenedetect import detect_shots

        chunk_data = storage.read_chunks(video_id)
        chunks = chunk_data["chunks"]
        duration_ms = chunk_data["duration_ms"]
        todo = [c for c in chunks if force or not _chunk_done(paths, c)]
        log.info("[%s] %d/%d chunk(s) to scan for shots", video_id, len(todo), len(chunks))
        paths.frames_dir.mkdir(parents=True, exist_ok=True)

        for n, chunk in enumerate(todo):
            label = f"chunk {chunk['index'] + 1}/{len(chunks)}"
            base = n / len(todo)
            step = 1 / len(todo)

            # 1) Detect cuts in the padded range (first ~40% of this chunk's progress)
            progress(NAME, base, f"{label}: detecting cuts")
            raw = detect_shots(paths.proxy, chunk["start_ms"], chunk["end_ms"],
                               threshold=settings.shot_threshold, min_shot_ms=settings.min_shot_ms)

            # 2) Keep shots owned by this chunk (midpoint in core range; last chunk owns the end)
            is_last = chunk["index"] == len(chunks) - 1
            shots: list[dict[str, Any]] = []
            for start, end in raw:
                mid = (start + end) // 2
                if not (chunk["core_start_ms"] <= mid < chunk["core_end_ms"] or (is_last and mid >= chunk["core_end_ms"])):
                    continue
                shots.append({
                    "start_ms": start,
                    "end_ms": end,
                    # True when the shot touches the padded edge (not the movie's edge):
                    # it may really continue into the neighbour chunk; the merge stage joins these.
                    "clipped_start": start <= chunk["start_ms"] and chunk["start_ms"] > 0,
                    "clipped_end": end >= chunk["end_ms"] and chunk["end_ms"] < duration_ms,
                    "keyframes": [
                        {"ts_ms": ts, "file": f"frames/{paths.frame(ts).name}"}
                        for ts in keyframe_times(start, end, settings.keyframe_every_ms,
                                                 settings.max_keyframes_per_shot)
                    ],
                })

            # 3) Extract keyframes (remaining ~60%); existing files are reused on resume
            frames = [kf["ts_ms"] for s in shots for kf in s["keyframes"]]
            for i, ts in enumerate(frames):
                dst = paths.frame(ts)
                if not dst.is_file():
                    ffmpeg.extract_frame(paths.proxy, ts, dst, quality=settings.keyframe_jpeg_quality)
                progress(NAME, base + step * (0.4 + 0.6 * (i + 1) / max(len(frames), 1)),
                         f"{label}: frame {i + 1}/{len(frames)}")

            # On a re-run, delete this chunk's old keyframes that the new shot list no longer uses.
            removed = _remove_stale_frames(paths, chunk, keep=set(frames))
            if removed:
                log.info("[%s] %s: removed %d old keyframe(s)", video_id, chunk["id"], removed)

            storage.write_json(paths.shots(chunk["id"]), {
                "chunk_id": chunk["id"],
                "start_ms": chunk["start_ms"],
                "end_ms": chunk["end_ms"],
                "detector": {"type": "content", "threshold": settings.shot_threshold,
                             "min_shot_ms": settings.min_shot_ms},
                "keyframe_every_ms": settings.keyframe_every_ms,
                "shots": shots,
            })
            storage.set_chunk_stage_status(video_id, chunk["id"], NAME, "done")
            log.info("[%s] %s: %d shot(s), %d keyframe(s)", video_id, chunk["id"], len(shots), len(frames))

        all_shots = load_shots(video_id)
        storage.update_metadata(video_id, shots={
            "shots": len(all_shots),
            "keyframes": sum(len(s["keyframes"]) for s in all_shots),
            "avg_shot_ms": int(sum(s["end_ms"] - s["start_ms"] for s in all_shots) / max(len(all_shots), 1)),
        })

    progress(NAME, 1.0, "done")


# ---------------------------------------------------------------- reading results
def load_shots(video_id: str) -> list[dict[str, Any]]:
    """All shots of the movie in time order (each chunk file already holds only the shots it owns)."""
    paths = storage.video_paths(video_id)
    shots: list[dict[str, Any]] = []
    for chunk in storage.read_chunks(video_id).get("chunks", []):
        shots.extend(storage.read_json(paths.shots(chunk["id"])).get("shots", []))
    shots.sort(key=lambda s: s["start_ms"])
    return shots


def _remove_stale_frames(paths: storage.VideoPaths, chunk: dict[str, Any], keep: set[int]) -> int:
    """Delete frames/*.jpg owned by this chunk (time inside its core range) that aren't in `keep`."""
    if not paths.frames_dir.is_dir():
        return 0
    removed = 0
    for f in paths.frames_dir.glob("*.jpg"):
        if not f.stem.isdigit():
            continue
        ts = int(f.stem)
        if chunk["core_start_ms"] <= ts < chunk["core_end_ms"] and ts not in keep:
            f.unlink(missing_ok=True)
            removed += 1
    return removed


def _chunk_done(paths: storage.VideoPaths, chunk: dict[str, Any]) -> bool:
    return chunk.get("stages", {}).get(NAME) == "done" and paths.shots(chunk["id"]).is_file()
