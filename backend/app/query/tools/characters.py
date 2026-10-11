"""get_characters: who is in the video, with roles and where they appear.

Combines the movie profile (stage 9: main characters + roles) with the per-scene
character lists (stage 7) and name mentions in dialogue / on-screen text.
"""
from __future__ import annotations

import re
from typing import Any

from app.pipeline.stages import s06_merge, s07_scenes, s09_summarize


def get_characters(video_id: str) -> dict[str, Any]:
    """{main: [{name, role, description}], appearances: [{name, scenes, first_ms, last_ms, mentions}]}."""
    profile = s09_summarize.load_summaries(video_id).get("profile", {})
    scenes = [s for s in s07_scenes.load_scenes(video_id) if "error" not in s]
    tl = s06_merge.load_timeline(video_id)

    seen: dict[str, dict[str, Any]] = {}
    for sc in scenes:
        for name in sc.get("characters", []):
            key = name.lower()
            e = seen.setdefault(key, {"name": name, "scenes": [], "first_ms": sc["start_ms"], "last_ms": sc["end_ms"]})
            e["scenes"].append(sc["scene"])
            e["last_ms"] = max(e["last_ms"], sc["end_ms"])

    lines = [d["text"] for d in tl.get("dialogue", []) if d.get("reliable")] + \
            [t["text"] for t in tl.get("on_screen_text", [])]
    for e in seen.values():
        pat = re.compile(r"\b" + re.escape(e["name"]) + r"\b", re.IGNORECASE)
        e["mentions"] = sum(1 for ln in lines if pat.search(ln))

    named = sorted(seen.values(), key=lambda e: (-len(e["scenes"]), e["first_ms"]))
    return {"main": profile.get("characters", []), "appearances": named[:20]}
