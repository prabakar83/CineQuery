"""FastAPI application: the HTTP API the frontend talks to.

Run from backend/:
    uvicorn app.main:app --port 8000
Docs (try every endpoint in the browser): http://localhost:8000/docs

Don't use --reload while a video is processing: a reload restarts the server and stops the job
(it resumes on the next start, finished stages are kept).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, progress, uploads, videos
from app.config import settings
from app.pipeline import queue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
for noisy in ("httpx", "httpcore", "huggingface_hub", "faster_whisper", "pyscenedetect", "urllib3", "transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    queue.start()            # background worker + resume unfinished videos
    yield


app = FastAPI(title="CineQuery API", version="0.1.0", lifespan=lifespan,
              description="Upload a video, follow its processing, then ask anything about it.")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"])

for r in (uploads.router, videos.router, progress.router, chat.router):
    app.include_router(r, prefix="/api")


@app.get("/api/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
