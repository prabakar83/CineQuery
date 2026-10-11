"""Rule-based router: obvious questions get their tool results up front, no LLM planning needed.

    "what happens between 1:42:50 and 1:50:00"  -> get_range(1:42:50, 1:50:00)
    "what is happening at 0:45"                  -> get_range(0:30, 1:00)
    "what genre is this" / plot / ending / about -> get_movie_summary
    "who is in it" / characters / names          -> get_characters

Returns a list of planned tool calls [{"action": ..., args}]. The agent runs them first
and can still call more tools afterwards.
"""
from __future__ import annotations

import re
from typing import Any

from app.domain.timecode import find_times

AROUND_MS = 15_000   # a single timestamp -> +-15 s window

_SUMMARY = re.compile(r"\b(genre|plot|story|about|ending|end|ends|theme|themes|summary|summari[sz]e|"
                      r"tone|setting|kind of (movie|film|video)|logline)\b", re.I)
_PEOPLE = re.compile(r"\b(who|characters?|names?|people|cast|protagonist|hero|villain|antagonist)\b", re.I)


def route(question: str) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    times = find_times(question)
    if len(times) >= 2:
        start, end = sorted(times[:2])
        plan.append({"action": "get_range", "start_ms": start, "end_ms": end})
    elif len(times) == 1:
        plan.append({"action": "get_range", "start_ms": max(times[0] - AROUND_MS, 0), "end_ms": times[0] + AROUND_MS})
    if _SUMMARY.search(question) and not times:
        plan.append({"action": "get_movie_summary"})
    if _PEOPLE.search(question) and not times:
        plan.append({"action": "get_characters"})
    return plan
