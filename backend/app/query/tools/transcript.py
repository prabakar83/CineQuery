"""search_dialogue: find exact or near-exact quotes in speech and on-screen text.

"Who says 'Lisan al-Gaib'?", "When does someone say 'paradise'?". Fuzzy, so small
speech-recognition errors still match. Unreliable (other-language) speech is skipped.
"""
from __future__ import annotations

import difflib
import re
from typing import Any

from app.pipeline.stages import s06_merge

_WORD = re.compile(r"[\w'-]+")


def search_dialogue(video_id: str, text: str, top_k: int = 8, min_score: float = 0.65) -> list[dict[str, Any]]:
    """Best-matching lines: [{score, source ('says'|'on screen'), start_ms, end_ms, text}]."""
    tl = s06_merge.load_timeline(video_id)
    lines = [{"source": "says", **d} for d in tl.get("dialogue", []) if d.get("reliable")]
    lines += [{"source": "on screen", **t} for t in tl.get("on_screen_text", [])]
    q = text.lower().strip()
    out = []
    for ln in lines:
        s = match_score(q, ln["text"].lower())
        if s >= min_score:
            out.append({"score": round(s, 2), "source": ln["source"], "start_ms": ln["start_ms"],
                        "end_ms": ln["end_ms"], "text": ln["text"]})
    out.sort(key=lambda r: (-r["score"], r["start_ms"]))
    return out[:top_k]


def match_score(query: str, line: str) -> float:
    """1.0 if the query appears literally; else the best of word overlap and fuzzy similarity."""
    if not query:
        return 0.0
    if query in line:
        return 1.0
    qw = set(_WORD.findall(query))
    lw = set(_WORD.findall(line))
    overlap = len(qw & lw) / len(qw) if qw else 0.0
    # fuzzy: compare the query with every window of the line of the same length
    n = len(query)
    best = 0.0
    for i in range(0, max(len(line) - n, 0) + 1, max(n // 4, 1)):
        best = max(best, difflib.SequenceMatcher(None, query, line[i:i + n]).ratio())
    return max(overlap, best)
