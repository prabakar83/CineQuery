"""The contract every pipeline stage follows."""
from __future__ import annotations

from typing import Callable, Optional, Protocol

# progress(stage_name, fraction 0..1, message)
ProgressCallback = Callable[[str, float, str], None]


def no_progress(stage: str, fraction: float, message: str) -> None:  # default: stay silent
    pass


class StageError(RuntimeError):
    """Raised when a stage cannot complete. The message should tell the user what to fix."""


class Stage(Protocol):
    name: str

    def is_done(self, video_id: str) -> bool:
        """True if this stage's outputs already exist and are valid (so run() can be skipped)."""
        ...

    def run(self, video_id: str, chunk_id: Optional[str] = None, *, force: bool = False) -> None:
        """Read inputs from disk/DB, write outputs to disk/DB, mark itself done."""
        ...
