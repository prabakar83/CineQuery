"""Live processing progress as Server-Sent Events.

    GET /api/videos/{id}/events
    -> event: progress   data: {status, stage, fraction, message, overall, stages{...}}   (on every change)
    -> event: done | failed  (last message, then the stream closes)
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.deps import video_id_param
from app.api.sse import SSE_HEADERS, ping, sse
from app.pipeline import queue

router = APIRouter(tags=["progress"])

POLL_S = 1.0
PING_S = 15.0


@router.get("/videos/{video_id}/events")
async def progress_events(request: Request, video_id: str = Depends(video_id_param)) -> StreamingResponse:
    return StreamingResponse(_events(request, video_id), media_type="text/event-stream", headers=SSE_HEADERS)


async def _events(request: Request, video_id: str) -> AsyncIterator[str]:
    last, last_sent = None, time.monotonic()
    while not await request.is_disconnected():
        p = await asyncio.to_thread(queue.get_progress, video_id)
        key = (p["status"], p["stage"], p["fraction"], p["message"], p["queue_position"])
        if key != last:
            last, last_sent = key, time.monotonic()
            yield sse("progress", p)
            if p["status"] in ("done", "failed"):
                yield sse(p["status"], p)
                return
        elif time.monotonic() - last_sent > PING_S:
            last_sent = time.monotonic()
            yield ping()
        await asyncio.sleep(POLL_S)
