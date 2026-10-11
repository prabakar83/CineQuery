"""Request / response models (shown in the OpenAPI docs at /docs; the frontend client is generated from them)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class VideoSummary(BaseModel):
    video_id: str
    title: str
    status: str                       # new | queued | running | done | failed
    duration_ms: Optional[int] = None
    created_at: Optional[str] = None
    overall: float = 0.0              # processing progress 0..1


class UploadResponse(BaseModel):
    video_id: str
    title: str
    status: str


class Progress(BaseModel):
    video_id: str
    status: str
    stage: Optional[str] = None
    fraction: float = 0.0
    message: str = ""
    overall: float = 0.0
    queue_position: Optional[int] = None
    stages: dict[str, str] = {}
    error: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class Source(BaseModel):
    label: str
    start_ms: int
    end_ms: int


class AskResult(BaseModel):
    answer: str
    sources: list[Source] = []
    unverified_times: list[str] = []
    model: str
    seconds: float
    tool_calls: list[dict[str, Any]] = []
