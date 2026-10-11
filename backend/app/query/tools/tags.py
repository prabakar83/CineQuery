"""find_scenes_by_tag: fast path for common categories ("show all romance scenes").

Uses the tag scores from stage 8 (LLM confidence blended with how typical the scene is
for that tag within this movie). Falls back to the raw LLM confidences if stage 8 hasn't run.
"""
from __future__ import annotations

from typing import Any

from app.pipeline.stages import s07_scenes, s08_embed


def find_scenes_by_tag(video_id: str, tag: str, min_score: float = 0.5) -> list[dict[str, Any]]:
    """Scenes with score >= min_score, best first: [{scene, start_ms, end_ms, title, score, llm_confidence, evidence}]."""
    if tag not in s07_scenes.TAGS:
        raise ValueError(f"Unknown tag {tag!r}. Tags: {', '.join(s07_scenes.TAGS)}")
    index, _ = s08_embed.load_index(video_id)
    blended = index.get("tag_scores", {})
    out = []
    for sc in s07_scenes.load_scenes(video_id):
        if "error" in sc or sc.get("graphic"):
            continue
        llm = float(sc.get("tags", {}).get(tag, 0.0))
        score = blended.get(str(sc["scene"]), {}).get(tag, llm)
        if score >= min_score:
            out.append({"scene": sc["scene"], "start_ms": sc["start_ms"], "end_ms": sc["end_ms"],
                        "title": sc.get("title", ""), "score": round(score, 3), "llm_confidence": llm,
                        "evidence": sc.get("tag_evidence", {}).get(tag, "")})
    out.sort(key=lambda r: -r["score"])
    return out
