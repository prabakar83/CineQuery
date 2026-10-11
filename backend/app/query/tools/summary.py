"""get_movie_summary / get_chapter_summaries: whole-movie questions (genre, plot, ending, "why").

Reads summaries.json from stage 9. Used by `cli.py profile` / `cli.py chapters` now,
and later by the question-answering agent as tools.
"""
from __future__ import annotations

from typing import Any

from app.pipeline.stages import s09_summarize


def get_movie_summary(video_id: str) -> dict[str, Any]:
    """{video_type, genres[], logline, plot, characters[], themes[], setting, tone, ending, tag_stats}."""
    data = s09_summarize.load_summaries(video_id)
    if not data:
        raise RuntimeError(f"No summaries yet. Run: python cli.py summarize {video_id}")
    return data.get("profile", {})


def get_chapter_summaries(video_id: str, start_ms: int | None = None, end_ms: int | None = None) -> list[dict[str, Any]]:
    """Chapters in time order, optionally only those overlapping [start_ms, end_ms)."""
    data = s09_summarize.load_summaries(video_id)
    if not data:
        raise RuntimeError(f"No summaries yet. Run: python cli.py summarize {video_id}")
    out = []
    for c in data.get("chapters", []):
        if start_ms is not None and c["end_ms"] <= start_ms:
            continue
        if end_ms is not None and c["start_ms"] >= end_ms:
            continue
        out.append(c)
    return out
