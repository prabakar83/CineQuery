"""Read endpoints: list videos, details, the indexed data (scenes, chapters, profile), and the video file."""
from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.adapters import local_storage as storage
from app.api.deps import video_id_param
from app.api.schemas import Progress, VideoSummary
from app.pipeline import queue
from app.pipeline.stages import s06_merge, s07_scenes, s09_summarize

router = APIRouter(tags=["videos"])

_RANGE = re.compile(r"bytes=(\d*)-(\d*)")
CHUNK = 1024 * 1024
_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
          ".mkv": "video/x-matroska", ".avi": "video/x-msvideo"}


@router.get("/videos", response_model=list[VideoSummary])
def list_videos() -> list[VideoSummary]:
    out = []
    for vid in storage.list_video_ids():
        meta = storage.read_metadata(vid)
        prog = queue.get_progress(vid)
        out.append(VideoSummary(video_id=vid, title=meta.get("title") or meta.get("original_file") or vid,
                                status=prog["status"], duration_ms=meta.get("media", {}).get("duration_ms"),
                                created_at=meta.get("created_at"), overall=prog["overall"]))
    return sorted(out, key=lambda v: v.created_at or "", reverse=True)


@router.get("/videos/{video_id}")
def get_video(video_id: str = Depends(video_id_param)) -> dict[str, Any]:
    """Metadata (media info, stage results) + processing status."""
    meta = storage.read_metadata(video_id)
    return {"video_id": video_id, "title": meta.get("title") or meta.get("original_file") or video_id,
            "progress": queue.get_progress(video_id), "ready": queue.is_ready(video_id),
            "media": meta.get("media", {}), "summary": {k: meta.get(k) for k in
            ("transcript", "shots", "captions", "timeline", "scenes", "embed", "summarize") if k in meta}}


@router.get("/videos/{video_id}/progress", response_model=Progress)
def get_progress(video_id: str = Depends(video_id_param)) -> dict[str, Any]:
    """One snapshot. For live updates use /events (Server-Sent Events)."""
    return queue.get_progress(video_id)


@router.post("/videos/{video_id}/process", response_model=Progress, status_code=202)
def process_video(video_id: str = Depends(video_id_param)) -> dict[str, Any]:
    """(Re)start processing, e.g. after a failure. Finished stages are skipped."""
    queue.enqueue(video_id)
    return queue.get_progress(video_id)


@router.get("/videos/{video_id}/scenes")
def get_scenes(video_id: str = Depends(video_id_param)) -> list[dict[str, Any]]:
    return [{k: sc.get(k) for k in ("scene", "start_ms", "end_ms", "title", "description", "characters",
                                    "location", "time_of_day", "mood", "events", "graphic", "tag_evidence")}
            | {"tags": {t: v for t, v in sc.get("tags", {}).items() if v > 0}}
            for sc in s07_scenes.load_scenes(video_id) if "error" not in sc]


@router.get("/videos/{video_id}/chapters")
def get_chapters(video_id: str = Depends(video_id_param)) -> list[dict[str, Any]]:
    return s09_summarize.load_summaries(video_id).get("chapters", [])


@router.get("/videos/{video_id}/profile")
def get_profile(video_id: str = Depends(video_id_param)) -> dict[str, Any]:
    return s09_summarize.load_summaries(video_id).get("profile", {})


@router.get("/videos/{video_id}/transcript")
def get_transcript(video_id: str = Depends(video_id_param)) -> dict[str, Any]:
    """Dialogue + on-screen text on the movie clock (for a transcript panel / subtitles)."""
    tl = s06_merge.load_timeline(video_id)
    return {"dialogue": [{k: d.get(k) for k in ("id", "start_ms", "end_ms", "text", "reliable", "subtitled")}
                         for d in tl.get("dialogue", [])],
            "on_screen_text": [{k: t.get(k) for k in ("id", "start_ms", "end_ms", "text")}
                               for t in tl.get("on_screen_text", [])]}


@router.get("/videos/{video_id}/stream")
def stream_video(request: Request, video_id: str = Depends(video_id_param)) -> StreamingResponse:
    """The original file with HTTP Range support, so the browser player can seek."""
    path = storage.video_paths(video_id).original()
    if path is None:
        raise HTTPException(404, "Video file missing")
    size = path.stat().st_size
    ctype = _TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {"Accept-Ranges": "bytes", "Content-Type": ctype}

    m = _RANGE.fullmatch(request.headers.get("range", "").strip())
    if not m or (not m.group(1) and not m.group(2)):
        headers["Content-Length"] = str(size)
        return StreamingResponse(_read(path, 0, size - 1), headers=headers, media_type=ctype)
    if m.group(1):
        start = int(m.group(1))
        end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    else:                                           # "bytes=-500" = last 500 bytes
        start, end = max(size - int(m.group(2)), 0), size - 1
    if start >= size or start > end:
        raise HTTPException(416, "Range not satisfiable", headers={"Content-Range": f"bytes */{size}"})
    headers |= {"Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(end - start + 1)}
    return StreamingResponse(_read(path, start, end), status_code=206, headers=headers, media_type=ctype)


def _read(path: Path, start: int, end: int) -> Iterator[bytes]:
    with open(path, "rb") as f:
        f.seek(start)
        left = end - start + 1
        while left > 0:
            data = f.read(min(CHUNK, left))
            if not data:
                break
            left -= len(data)
            yield data
