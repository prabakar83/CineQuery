"""Describe images with a local vision model served by Ollama (http://localhost:11434).

A CLASS because it holds the connection settings and model name, and to make
swapping engines easy: any class with describe(image) -> dict works for stage 5.

Uses only the Python standard library (urllib), so no extra package is needed.
Ollama's "structured outputs" feature forces the reply to match FRAME_SCHEMA,
so we always get valid JSON in our exact shape.
"""
from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """The model is missing, or a request failed (may succeed on retry)."""


class OllamaUnreachable(OllamaError):
    """Ollama is not running / connection refused (retrying won't help)."""


FRAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "on_screen_text": {"type": "string"},
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "action": {"type": "string"},
                    "expression": {"type": "string"},
                },
                "required": ["description", "action", "expression"],
            },
        },
        "setting": {"type": "string"},
        "time_of_day": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "shot_type": {"type": "string"},
        "mood": {"type": "string"},
    },
    "required": ["caption", "on_screen_text", "people", "setting", "time_of_day",
                 "objects", "shot_type", "mood"],
}

FRAME_PROMPT = """You are analysing one still frame from a movie. Reply ONLY with JSON matching the schema.

Fields:
- caption: 1-2 sentences describing what is happening in the frame.
- on_screen_text: ALL text visible in the image (subtitles, signs, titles, screens), copied exactly as written. Empty string if there is none. Never invent text.
- people: one entry per visible person (max 6). description = visible appearance (age, gender if clear, hair, clothing, distinctive features). action = what they are doing. expression = facial expression / emotion. Do NOT guess names or identify real actors.
- setting: where this takes place (indoor/outdoor, type of place, notable features, lighting).
- time_of_day: day, night, dusk, dawn, or unknown.
- objects: important visible objects (weapons, vehicles, props), max 8.
- shot_type: one of extreme close-up, close-up, medium shot, wide shot, extreme wide shot, aerial, over-the-shoulder, insert.
- mood: the emotional tone in 1-3 words (e.g. tense, romantic, joyful, eerie, violent, calm).

Describe only what is visible. If something is unclear, say "unclear" rather than guessing."""


class OllamaVision:
    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 timeout_s: int = 180, retries: int = 2) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries

    # ---------------------------------------------------------------- setup checks
    def check_ready(self) -> None:
        """Fail early with a clear message if Ollama isn't running or the model isn't pulled."""
        try:
            tags = self._request("GET", "/api/tags", timeout=10)
        except OllamaError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}. Start the Ollama app (or run `ollama serve`) "
                f"and try again. ({exc})"
            ) from exc
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise OllamaError(f"Model '{self.model}' is not downloaded. Run: ollama pull {self.model}")

    # ---------------------------------------------------------------- main call
    def describe(self, image: Path) -> dict[str, Any]:
        """Return the structured description of one image (see FRAME_SCHEMA)."""
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": FRAME_PROMPT,
                "images": [base64.b64encode(Path(image).read_bytes()).decode("ascii")],
            }],
            "format": FRAME_SCHEMA,          # structured output: reply must match the schema
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096},
            "keep_alive": "5m",              # keep the model loaded between frames
        }
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            try:
                reply = self._request("POST", "/api/chat", payload, timeout=self.timeout_s)
                content = reply.get("message", {}).get("content", "")
                return _normalize(json.loads(content))
            except OllamaUnreachable:
                raise   # Ollama went away: retrying won't help
            except (json.JSONDecodeError, OllamaError, TimeoutError) as exc:
                last_error = exc
                log.warning("frame %s: attempt %d failed (%s)", Path(image).name, attempt, exc)
                time.sleep(1)
        raise OllamaError(f"Could not describe {Path(image).name}: {last_error}")

    def unload(self) -> None:
        """Free the GPU memory now instead of waiting for Ollama's idle timeout."""
        try:
            self._request("POST", "/api/generate", {"model": self.model, "keep_alive": 0}, timeout=30)
        except OllamaError as exc:
            log.debug("unload failed: %s", exc)

    # ---------------------------------------------------------------- HTTP
    def _request(self, method: str, path: str, body: dict | None = None, *, timeout: float) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise OllamaError(f"HTTP {exc.code} from Ollama: {detail}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise OllamaError(f"timed out after {timeout}s") from exc
            raise OllamaUnreachable(str(exc.reason)) from exc


def _normalize(d: dict[str, Any]) -> dict[str, Any]:
    """Make sure every field exists with the right type, whatever the model returned."""
    def s(v: Any) -> str:
        return v.strip() if isinstance(v, str) else ""

    people = []
    for p in d.get("people") or []:
        if isinstance(p, dict):
            people.append({"description": s(p.get("description")), "action": s(p.get("action")),
                           "expression": s(p.get("expression"))})
    return {
        "caption": s(d.get("caption")),
        "on_screen_text": s(d.get("on_screen_text")),
        "people": people[:6],
        "setting": s(d.get("setting")),
        "time_of_day": s(d.get("time_of_day")) or "unknown",
        "objects": [s(o) for o in (d.get("objects") or []) if isinstance(o, str) and o.strip()][:8],
        "shot_type": s(d.get("shot_type")),
        "mood": s(d.get("mood")),
    }
