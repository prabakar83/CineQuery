"""search_moments: find moments by meaning (+ a small exact-word bonus).

Used by `cli.py search` now, and later by the question-answering agent as a tool.

score = cosine similarity (meaning)  +  up to 0.15 if the query's words appear literally
(helps names and rare words such as "Lisan al-Gaib" that embeddings handle poorly).
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from app.adapters.embeddings import OllamaEmbedder
from app.config import settings
from app.pipeline.stages import s08_embed

KEYWORD_BONUS = 0.15
_WORD = re.compile(r"[\w'-]+")


def search_moments(video_id: str, query: str, *, top_k: int = 10,
                   kinds: Optional[Iterable[str]] = None,
                   embedder: Optional[OllamaEmbedder] = None) -> list[dict[str, Any]]:
    """Best-matching items, highest score first: [{score, similarity, kind, scene, start_ms, end_ms, text, ref}]."""
    index, vectors = s08_embed.load_index(video_id)
    if not index:
        raise RuntimeError(f"No search index yet. Run: python cli.py embed {video_id}")

    embedder = embedder or OllamaEmbedder(index.get("model", settings.embed_model), settings.ollama_url)
    q = embedder.embed_query(query)
    if q.shape[0] != vectors.shape[1]:
        raise RuntimeError("The index was built with a different embedding model. Re-run: python cli.py embed --force")

    sims = vectors @ q
    words = _keywords(query)
    wanted = set(kinds) if kinds else None
    results = []
    for i, item in enumerate(index["items"]):
        if wanted and item["kind"] not in wanted:
            continue
        bonus = KEYWORD_BONUS * keyword_overlap(words, item["text"]) if words else 0.0
        results.append({**item, "similarity": round(float(sims[i]), 3), "score": round(float(sims[i]) + bonus, 3)})
    results.sort(key=lambda r: -r["score"])
    return results[:top_k]


def keyword_overlap(words: set[str], text: str) -> float:
    """Share of the query's keywords that appear in the text (0..1)."""
    found = {w.lower() for w in _WORD.findall(text)}
    return sum(1 for w in words if w in found) / len(words)


def _keywords(query: str) -> set[str]:
    stop = {"the", "and", "for", "with", "when", "what", "where", "who", "does", "did", "are", "was",
            "scene", "scenes", "about", "that", "this", "there", "from", "into", "show", "find", "any"}
    return {w.lower() for w in _WORD.findall(query) if len(w) >= 3 and w.lower() not in stop}
