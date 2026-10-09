"""Shared pieces for pipeline stages.

Every stage is a plain Python module that provides:

    NAME: str                                   e.g. "normalize"
    is_done(video_id) -> bool                   True if run() can be skipped
    run(video_id, *, force=False, progress=no_progress) -> None

The runner (cli.py for now) just loops over a list of these modules.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator, Protocol

from app.adapters import local_storage as storage

# progress(stage_name, fraction 0..1, message)
ProgressCallback = Callable[[str, float, str], None]


def no_progress(stage: str, fraction: float, message: str) -> None:  # default: stay silent
    pass


class StageError(RuntimeError):
    """Raised when a stage cannot complete. The message should tell the user what to fix."""


class StageModule(Protocol):
    """What a stage module looks like (for type checkers; modules match it structurally)."""
    NAME: str

    def is_done(self, video_id: str) -> bool: ...
    def run(self, video_id: str, *, force: bool = False, progress: ProgressCallback = no_progress) -> None: ...


@contextmanager
def track_stage(video_id: str, stage: str) -> Iterator[None]:
    """Mark a stage running -> done (with seconds) or failed (with error) in metadata.json.

    Usage inside a stage's run():
        with track_stage(video_id, NAME):
            ...do the work...
    """
    started = time.monotonic()
    storage.set_stage_status(video_id, stage, "running")
    try:
        yield
    except BaseException as exc:
        storage.set_stage_status(video_id, stage, "failed", error=str(exc) or exc.__class__.__name__)
        raise
    storage.set_stage_status(video_id, stage, "done", seconds=round(time.monotonic() - started, 1))


def require_stage_done(video_id: str, stage: str) -> None:
    """Fail early with a clear message if an earlier stage hasn't finished."""
    if storage.get_stage_status(video_id, stage) != "done":
        raise StageError(f"Stage '{stage}' has not finished for {video_id}. Run it first.")
