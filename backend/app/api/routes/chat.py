"""Ask questions about a video; the answer streams back while the model writes it.

    POST /api/videos/{id}/ask        body {"question": "..."}     -> Server-Sent Events:
        event: status   {"message": "..."}                          waiting for the GPU / thinking
        event: step     {"action", "args", "source", "seconds"}     each tool call (search, range, ...)
        event: token    {"text": "..."}                             next piece of the answer
        event: reset    {}                                          discard tokens so far (a retry follows)
        event: answer   {answer, sources[], unverified_times[], model, seconds, tool_calls[]}   final
        event: error    {"message": "..."}

    POST /api/videos/{id}/ask/sync   same body -> one JSON AskResult (simple clients, tests)

Only one question runs at a time (one GPU); others wait. A question keeps running to the end even
if the browser disconnects (the model call can't be interrupted midway).
"""
from __future__ import annotations

import asyncio
import queue as stdqueue
import threading
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.deps import video_id_param
from app.api.schemas import AskRequest, AskResult
from app.api.sse import SSE_HEADERS, ping, sse
from app.pipeline import queue as jobs
from app.query.agent import ask

router = APIRouter(tags=["chat"])

_gpu = threading.Lock()          # one question at a time
_END = object()


@router.post("/videos/{video_id}/ask")
async def ask_stream(body: AskRequest, request: Request, video_id: str = Depends(video_id_param)) -> StreamingResponse:
    _require_ready(video_id)
    return StreamingResponse(_ask_events(request, video_id, body.question.strip()),
                             media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/videos/{video_id}/ask/sync", response_model=AskResult)
def ask_sync(body: AskRequest, video_id: str = Depends(video_id_param)) -> dict[str, Any]:
    _require_ready(video_id)
    with _gpu:
        return _result(ask(video_id, body.question.strip()))


def _require_ready(video_id: str) -> None:
    if not jobs.is_ready(video_id):
        p = jobs.get_progress(video_id)
        raise HTTPException(409, f"Video is not ready yet (status: {p['status']}, stage: {p['stage']}).")


async def _ask_events(request: Request, video_id: str, question: str) -> AsyncIterator[str]:
    events: "stdqueue.Queue[Any]" = stdqueue.Queue()
    put = lambda name, data: events.put((name, data))

    def on_step(step: dict[str, Any]) -> None:
        call = step["call"]
        put("step", {"action": call["action"], "args": {k: v for k, v in call.items() if k != "action"},
                     "source": step["source"], "seconds": step["seconds"]})

    def work() -> None:
        try:
            if not _gpu.acquire(blocking=False):
                put("status", {"message": "waiting for another question to finish"})
                _gpu.acquire()
            try:
                put("status", {"message": "thinking"})
                result = ask(video_id, question, on_step=on_step,
                             on_token=lambda t: put("token", {"text": t}),
                             on_reset=lambda: put("reset", {}))
                put("answer", _result(result))
            finally:
                _gpu.release()
        except Exception as exc:  # noqa: BLE001 — report any failure to the client
            put("error", {"message": str(exc) or exc.__class__.__name__})
        finally:
            events.put(_END)

    threading.Thread(target=work, name="ask", daemon=True).start()
    while True:
        try:
            item = await asyncio.to_thread(events.get, True, 15.0)
        except stdqueue.Empty:
            if await request.is_disconnected():
                return
            yield ping()
            continue
        if item is _END:
            return
        yield sse(*item)


def _result(r: dict[str, Any]) -> dict[str, Any]:
    return {"answer": r["answer"], "sources": r.get("sources", []), "unverified_times": r.get("unverified_times", []),
            "model": r["model"], "seconds": r["seconds"],
            "tool_calls": [{"action": s["call"]["action"],
                            "args": {k: v for k, v in s["call"].items() if k != "action"},
                            "source": s["source"], "seconds": s["seconds"]} for s in r.get("steps", [])]}
