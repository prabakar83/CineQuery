"""Movie time helpers (pure, no I/O). All project times are integer milliseconds."""
from __future__ import annotations

import re
from typing import Optional

TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\b")


def parse_time(value: Optional[str]) -> Optional[int]:
    """'1:42:50' / '1:42' / '95' / '95.5' -> milliseconds. None/'' -> None. Bad input -> None."""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        seconds = 0.0
        for part in value.split(":"):
            seconds = seconds * 60 + float(part)
    except ValueError:
        return None
    return int(seconds * 1000)


def fmt(ms: int) -> str:
    """Milliseconds -> 'h:mm:ss' (or 'm:ss' under an hour)."""
    s = max(int(ms), 0) // 1000
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def find_times(text: str) -> list[int]:
    """All timestamps written in a text, in order, as milliseconds."""
    return [t for t in (parse_time(m) for m in TIME_RE.findall(text)) if t is not None]
