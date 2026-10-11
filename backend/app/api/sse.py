"""Server-Sent Events helpers: the browser opens one HTTP response and the server keeps writing
"event: name\ndata: {json}\n\n" blocks to it (EventSource / fetch streaming on the frontend)."""
from __future__ import annotations

import json
from typing import Any

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def ping() -> str:
    """Comment line: keeps proxies/browsers from closing an idle stream."""
    return ": ping\n\n"
