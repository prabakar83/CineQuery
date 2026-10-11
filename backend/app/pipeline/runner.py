"""The pipeline: which stages run, in which order. Used by the CLI and the API worker."""
from __future__ import annotations

from app.pipeline.base import ProgressCallback, no_progress
from app.pipeline.stages import (s01_normalize, s02_chunk, s03_transcribe, s04_shots, s05_caption,
                                 s06_merge, s07_scenes, s08_embed, s09_summarize)

# Stages run in this order. New stages get appended here.
PIPELINE = [s01_normalize, s02_chunk, s03_transcribe, s04_shots, s05_caption,
            s06_merge, s07_scenes, s08_embed, s09_summarize]
STAGES_BY_NAME = {stage.NAME: stage for stage in PIPELINE}
STAGE_NAMES = [stage.NAME for stage in PIPELINE]


def run_pipeline(video_id: str, *, force: bool = False, progress: ProgressCallback = no_progress) -> None:
    """Run every stage in order. Finished stages are skipped, so this also resumes after a crash."""
    for stage in PIPELINE:
        stage.run(video_id, force=force, progress=progress)


def overall_fraction(stage: str, fraction: float) -> float:
    """Progress of the whole pipeline (0..1) from the current stage and its own progress."""
    if stage not in STAGE_NAMES:
        return 0.0
    return (STAGE_NAMES.index(stage) + max(0.0, min(fraction, 1.0))) / len(STAGE_NAMES)
