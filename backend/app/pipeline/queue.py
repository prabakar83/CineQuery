"""Background processing for the API: one worker thread runs the pipeline, one video at a time.

Why one at a time: the GPU (4 GB) can hold only one model, and every stage after normalize
uses it. Videos wait in a FIFO queue.

State lives in two places:
    metadata.json -> "job": {"status": queued|running|done|failed, "error", "updated_at"}   (survives restarts)
    memory        -> live progress {stage, fraction, message, overall} for the progress stream

On API start, videos whose job was queued/running (e.g. the server was stopped mid-way) are
queued again; finished stages are skipped, so processing resumes where it stopped.

Plain module-level functions + a little module state (like a singleton); no class needed.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from app.adapters import local_storage as storage
from app.pipeline import runner

log = logging.getLogger(__name__)

_jobs: "queue.Queue[str]" = queue.Queue()
_queued: set[str] = set()                       # ids waiting or running (avoid double-queueing)
_progress: dict[str, dict[str, Any]] = {}       # video_id -> live progress
_lock = threading.Lock()
_worker: Optional[threading.Thread] = None


# =============================================================== public
def start() -> None:
    """Start the worker thread (once) and re-queue unfinished videos."""
    global _worker
    if _worker and _worker.is_alive():
        return
    _worker = threading.Thread(target=_work_loop, name="pipeline-worker", daemon=True)
    _worker.start()
    for vid in storage.list_video_ids():
        if storage.read_metadata(vid).get("job", {}).get("status") in ("queued", "running"):
            log.info("resuming %s", vid)
            enqueue(vid)


def enqueue(video_id: str) -> bool:
    """Queue a video for processing. False if it is already queued or running."""
    with _lock:
        if video_id in _queued:
            return False
        _queued.add(video_id)
        _progress[video_id] = {"stage": None, "fraction": 0.0, "message": "waiting in queue",
                               "overall": 0.0, "updated": time.time()}
    _set_job(video_id, "queued")
    _jobs.put(video_id)
    return True


def get_progress(video_id: str) -> dict[str, Any]:
    """{status, stage, fraction, message, overall, queue_position, stages{name: status}, error?}."""
    meta = storage.read_metadata(video_id)
    job = meta.get("job", {})
    with _lock:
        live = dict(_progress.get(video_id, {}))
        waiting = list(_jobs.queue)
    status = job.get("status") or ("done" if _all_done(meta) else "new")
    out = {
        "video_id": video_id,
        "status": status,
        "stage": live.get("stage"),
        "fraction": round(live.get("fraction", 1.0 if status == "done" else 0.0), 3),
        "message": live.get("message", ""),
        "overall": round(live.get("overall", 1.0 if status == "done" else 0.0), 3),
        "queue_position": waiting.index(video_id) + 1 if video_id in waiting else None,
        "stages": {name: meta.get("stages", {}).get(name, {}).get("status", "pending") for name in runner.STAGE_NAMES},
    }
    if job.get("error"):
        out["error"] = job["error"]
    return out


def is_ready(video_id: str) -> bool:
    """True when every stage has finished (questions can be asked)."""
    return _all_done(storage.read_metadata(video_id))


# =============================================================== worker
def _work_loop() -> None:
    while True:
        video_id = _jobs.get()
        try:
            _process(video_id)
        finally:
            with _lock:
                _queued.discard(video_id)
            _jobs.task_done()


def _process(video_id: str) -> None:
    log.info("processing %s", video_id)
    _set_job(video_id, "running")

    def progress(stage: str, fraction: float, message: str) -> None:
        with _lock:
            _progress[video_id] = {"stage": stage, "fraction": fraction, "message": message,
                                   "overall": runner.overall_fraction(stage, fraction), "updated": time.time()}

    try:
        runner.run_pipeline(video_id, progress=progress)
    except Exception as exc:  # noqa: BLE001 — record any failure; the worker must keep running
        log.exception("processing %s failed", video_id)
        _set_job(video_id, "failed", error=str(exc) or exc.__class__.__name__)
        with _lock:
            _progress.setdefault(video_id, {})["message"] = f"failed: {exc}"
        return
    _set_job(video_id, "done")
    with _lock:
        _progress[video_id] = {"stage": runner.STAGE_NAMES[-1], "fraction": 1.0, "message": "ready",
                               "overall": 1.0, "updated": time.time()}
    log.info("processing %s done", video_id)


# =============================================================== helpers
def _set_job(video_id: str, status: str, **extra: Any) -> None:
    storage.update_metadata(video_id, job={
        "status": status, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra})


def _all_done(meta: dict[str, Any]) -> bool:
    stages = meta.get("stages", {})
    return all(stages.get(name, {}).get("status") == "done" for name in runner.STAGE_NAMES)
