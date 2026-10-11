"""Text embeddings served by Ollama (default model: nomic-embed-text, ~270 MB).

An embedding turns a text into a vector (768 numbers for nomic-embed-text) so that texts
with similar MEANING get vectors pointing in a similar direction. Comparing two vectors
with cosine similarity (-1 .. 1, higher = closer in meaning) is how we search by meaning.

A CLASS because it holds the model name and connection settings; anything with
embed_documents(texts) / embed_query(text) can replace it (bge-small via
sentence-transformers, a cloud API, ...). Standard library + numpy only.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

import numpy as np

from app.adapters.ollama_vision import OllamaError, OllamaUnreachable

log = logging.getLogger(__name__)


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434",
                 timeout_s: int = 120, batch_size: int = 32, retries: int = 2) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.batch_size = batch_size
        self.retries = retries
        # nomic-embed-text was trained with task prefixes; using them noticeably improves search.
        nomic = "nomic" in model.lower()
        self.doc_prefix = "search_document: " if nomic else ""
        self.query_prefix = "search_query: " if nomic else ""

    # ------------------------------------------------------------ public
    def check_ready(self) -> None:
        try:
            tags = self._request("GET", "/api/tags", timeout=10)
        except OllamaError as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}. Start the Ollama app and try again. ({exc})") from exc
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise OllamaError(f"Embedding model '{self.model}' is not downloaded. Run: ollama pull {self.model}")

    def embed_documents(self, texts: list[str],
                        on_progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        """Texts to search IN -> float32 matrix (len(texts), dim), each row length 1."""
        rows: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [self.doc_prefix + t for t in texts[i:i + self.batch_size]]
            rows.append(self._embed(batch))
            if on_progress:
                on_progress(min(i + self.batch_size, len(texts)), len(texts))
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(rows)

    def embed_query(self, text: str) -> np.ndarray:
        """The question to search WITH -> float32 vector (dim,), length 1."""
        return self._embed([self.query_prefix + text])[0]

    def unload(self) -> None:
        try:
            self._request("POST", "/api/embed", {"model": self.model, "input": [], "keep_alive": 0}, timeout=30)
        except OllamaError as exc:
            log.debug("unload failed: %s", exc)

    # ------------------------------------------------------------ internals
    def _embed(self, batch: list[str]) -> np.ndarray:
        last: Optional[Exception] = None
        for attempt in range(1, self.retries + 2):
            try:
                reply = self._request("POST", "/api/embed",
                                      {"model": self.model, "input": batch, "truncate": True},
                                      timeout=self.timeout_s)
                vecs = np.asarray(reply.get("embeddings", []), dtype=np.float32)
                if vecs.shape[0] != len(batch):
                    raise OllamaError(f"expected {len(batch)} embeddings, got {vecs.shape[0]}")
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs / np.maximum(norms, 1e-12)     # unit length -> dot product = cosine similarity
            except OllamaUnreachable:
                raise
            except OllamaError as exc:
                last = exc
                log.warning("embed attempt %d failed (%s)", attempt, exc)
                time.sleep(1)
        raise OllamaError(f"Embedding failed after {self.retries + 1} attempts: {last}")

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
