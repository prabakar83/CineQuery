"""Shared request checks (FastAPI dependencies)."""
from __future__ import annotations

from fastapi import HTTPException, Path

from app.adapters import local_storage as storage


def video_id_param(video_id: str = Path(..., pattern=r"^[A-Za-z0-9_-]{1,64}$")) -> str:
    """The {video_id} path parameter, validated, 404 if the video doesn't exist."""
    if not storage.video_exists(video_id):
        raise HTTPException(status_code=404, detail=f"Unknown video {video_id}")
    return video_id
