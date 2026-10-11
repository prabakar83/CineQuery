"""get_range: everything known about a time window (scenes, dialogue, on-screen text).

Answers "what happens between 1:42:50 and 1:50:00?" and gives context around a moment.
"""
from __future__ import annotations

from typing import Any

from app.pipeline.stages import s06_merge, s07_scenes

MAX_LINES = 40


def get_range(video_id: str, start_ms: int, end_ms: int) -> dict[str, Any]:
    """{start_ms, end_ms, scenes[], dialogue[], on_screen_text[]} overlapping [start_ms, end_ms)."""
    if end_ms <= start_ms:
        end_ms = start_ms + 1000
    overlaps = lambda x: x["start_ms"] < end_ms and x["end_ms"] > start_ms
    scenes = [{"scene": s["scene"], "start_ms": s["start_ms"], "end_ms": s["end_ms"],
               "title": s.get("title", ""), "description": s.get("description", ""),
               "mood": s.get("mood", ""), "characters": s.get("characters", []),
               "tags": [k for k, v in s.get("tags", {}).items() if v >= 0.5]}
              for s in s07_scenes.load_scenes(video_id) if overlaps(s) and "error" not in s]
    tl = s06_merge.load_timeline(video_id)
    dialogue = [{"start_ms": d["start_ms"], "end_ms": d["end_ms"],
                 "text": d["text"] if d.get("reliable") else "[speech in another language]",
                 "reliable": bool(d.get("reliable"))}
                for d in tl.get("dialogue", []) if overlaps(d)]
    texts = [{"start_ms": t["start_ms"], "end_ms": t["end_ms"], "text": t["text"]}
             for t in tl.get("on_screen_text", []) if overlaps(t)]
    return {"start_ms": start_ms, "end_ms": end_ms, "scenes": scenes,
            "dialogue": _thin(dialogue), "on_screen_text": _thin(texts)}


def _thin(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep at most MAX_LINES, spread evenly over the window."""
    if len(items) <= MAX_LINES:
        return items
    step = len(items) / MAX_LINES
    return [items[int(i * step)] for i in range(MAX_LINES)]
