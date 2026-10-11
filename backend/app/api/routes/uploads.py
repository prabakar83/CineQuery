"""Add videos: upload the file bytes, or register a file that is already on this computer.

Upload = the raw file as the request body (not multipart), so a multi-GB movie streams straight
to disk without a temp copy:
    POST /api/videos?filename=movie.mp4      body: the file bytes (Content-Type: application/octet-stream)
The frontend sends it with fetch/XHR (XHR gives an upload progress bar).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.adapters import local_storage as storage
from app.api.schemas import UploadResponse
from app.config import settings
from app.pipeline import queue

log = logging.getLogger(__name__)
router = APIRouter(tags=["videos"])

CHUNK = 1024 * 1024


@router.post("/videos", response_model=UploadResponse, status_code=201)
async def upload_video(request: Request, filename: str = Query(..., min_length=1, max_length=255)) -> UploadResponse:
    ext = _check_extension(filename)
    video_id = storage.new_video_id()
    paths = storage.video_paths(video_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    tmp = paths.root / f"original.tmp{ext}"
    limit = int(settings.max_upload_gb * 1024 ** 3)
    size = 0
    try:
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File larger than {settings.max_upload_gb} GB")
                f.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty upload")
        os.replace(tmp, paths.root / f"original{ext}")
    except BaseException:
        tmp.unlink(missing_ok=True)
        if not any(paths.root.iterdir()):
            paths.root.rmdir()
        raise
    _register(video_id, Path(filename).name)
    log.info("uploaded %s (%s, %.1f MB)", video_id, filename, size / CHUNK)
    return UploadResponse(video_id=video_id, title=Path(filename).name, status="queued")


class ImportRequest(BaseModel):
    path: str          # a file on the machine running the API, e.g. "D:\\Movies\\film.mkv"


@router.post("/videos/import", response_model=UploadResponse, status_code=201)
def import_local_video(body: ImportRequest) -> UploadResponse:
    """Register a file already on this computer (hard link when possible: instant, no extra space)."""
    src = Path(body.path).expanduser()
    if not src.is_file():
        raise HTTPException(400, f"File not found on the server: {src}")
    _check_extension(src.name)
    video_id = storage.new_video_id()
    storage.import_original(src, video_id)
    _register(video_id, src.name)
    return UploadResponse(video_id=video_id, title=src.name, status="queued")


def _check_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in settings.upload_extensions:
        raise HTTPException(415, f"Unsupported file type {ext or '(none)'}. Allowed: {', '.join(settings.upload_extensions)}")
    return ext


def _register(video_id: str, title: str) -> None:
    storage.update_metadata(video_id, title=title,
                            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    queue.enqueue(video_id)
