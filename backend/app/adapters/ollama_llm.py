"""Text LLM served by Ollama (http://localhost:11434), e.g. qwen2.5:3b.

A CLASS because it holds the connection settings and model name; any class with
generate_json(system, prompt, schema) -> dict can replace it (another local model,
or a cloud API later). Standard library only (urllib).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from app.adapters.ollama_vision import OllamaError, OllamaUnreachable

log = logging.getLogger(__name__)


class OllamaLLM:
    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 timeout_s: int = 180, retries: int = 2, num_ctx: int = 4096) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries
        self.num_ctx = num_ctx

    def check_ready(self) -> None:
        try:
            tags = self._request("GET", "/api/tags", timeout=10)
        except OllamaError as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}. Start the Ollama app and try again. ({exc})") from exc
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise OllamaError(f"Model '{self.model}' is not downloaded. Run: ollama pull {self.model}")

    def generate_json(self, system: str, prompt: str, schema: dict[str, Any],
                      temperature: float = 0.2) -> dict[str, Any]:
        """Ask the model for JSON matching `schema` (Ollama structured outputs)."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "format": schema,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
            "keep_alive": "5m",
        }
        last: Optional[Exception] = None
        for attempt in range(1, self.retries + 2):
            try:
                reply = self._request("POST", "/api/chat", payload, timeout=self.timeout_s)
                return json.loads(reply.get("message", {}).get("content", ""))
            except OllamaUnreachable:
                raise
            except (json.JSONDecodeError, OllamaError) as exc:
                last = exc
                log.warning("LLM attempt %d failed (%s)", attempt, exc)
                time.sleep(1)
        raise OllamaError(f"LLM failed after {self.retries + 1} attempts: {last}")

    def unload(self) -> None:
        try:
            self._request("POST", "/api/generate", {"model": self.model, "keep_alive": 0}, timeout=30)
        except OllamaError as exc:
            log.debug("unload failed: %s", exc)

    def _request(self, method: str, path: str, body: dict | None = None, *, timeout: float) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            raise OllamaError(f"HTTP {exc.code} from Ollama: {exc.read().decode('utf-8', 'replace')[:500]}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise OllamaError(f"timed out after {timeout}s") from exc
            raise OllamaUnreachable(str(exc.reason)) from exc
