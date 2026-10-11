"""Stage 8 — Embed: make the movie searchable by meaning.

Input : scenes.json (stage 7), timeline.json (stage 6)
Output: data/videos/{id}/embeddings/index.json   one entry per searchable item + tag scores
        data/videos/{id}/embeddings/vectors.npy  row i = vector of items[i] (float32, unit length)
        metadata.json -> embed summary, stages.embed

What gets embedded (each becomes one "item"):
    scene     title + description + key moments + characters + location + mood + tags
    dialogue  every RELIABLE spoken line (gibberish from other languages is skipped)
    text      every on-screen text (subtitles carry the meaning of foreign dialogue)

Tag scores: the LLM's tag confidences are absolute ("religious_ritual 0.9" on 7 of 8
scenes). We also embed each tag's definition and rank the scenes of THIS movie by how close
they are to it. tag score = 0.6 * LLM confidence + 0.4 * rank (0 = least like the tag in
this movie, 1 = most like it). So the most typical scenes for a tag come first.

Runs in seconds (a 2-hour movie has ~200 scenes and ~2,000 lines). No resume needed:
re-running embeds everything again.
"""
from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.adapters import local_storage as storage
from app.adapters.embeddings import OllamaEmbedder
from app.adapters.ollama_vision import OllamaError, OllamaUnreachable
from app.config import settings
from app.pipeline.base import ProgressCallback, StageError, no_progress, require_stage_done, track_stage
from app.pipeline.stages import s06_merge, s07_scenes

log = logging.getLogger(__name__)

NAME = "embed"


# =============================================================== stage
def is_done(video_id: str) -> bool:
    meta = storage.read_metadata(video_id)
    paths = storage.video_paths(video_id)
    return (meta.get("stages", {}).get(NAME, {}).get("status") == "done"
            and paths.embeddings_index.exists() and paths.embeddings_vectors.exists()
            and meta.get(NAME, {}).get("inputs") == _inputs(meta))   # re-run if scenes / merge / model changed


def run(video_id: str, *, force: bool = False, progress: ProgressCallback = no_progress) -> None:
    if not force and is_done(video_id):
        log.info("[%s] %s already done, skipping", video_id, NAME)
        progress(NAME, 1.0, "already done")
        return

    require_stage_done(video_id, "scenes")
    scenes = s07_scenes.load_scenes(video_id)
    timeline = s06_merge.load_timeline(video_id)
    if not scenes:
        raise StageError("scenes.json has no scenes. Re-run: python cli.py scenes <id> --force")

    embedder = OllamaEmbedder(settings.embed_model, settings.ollama_url, batch_size=settings.embed_batch_size)
    try:
        embedder.check_ready()
    except OllamaError as exc:
        raise StageError(str(exc)) from exc

    with track_stage(video_id, NAME):
        items = build_items(scenes, timeline)
        if not items:
            raise StageError("Nothing to embed (no scenes, reliable dialogue or on-screen text).")
        progress(NAME, 0.0, f"embedding {len(items)} items")
        try:
            vectors = embedder.embed_documents(
                [it["text"] for it in items],
                on_progress=lambda done, total: progress(NAME, 0.9 * done / total, f"{done}/{total} items"))
            progress(NAME, 0.9, "tag scores")
            tag_scores = compute_tag_scores(scenes, items, vectors, embedder)
        except OllamaUnreachable as exc:
            raise StageError(f"Lost connection to Ollama ({exc}). Start Ollama and re-run.") from exc
        except OllamaError as exc:
            raise StageError(str(exc)) from exc
        finally:
            embedder.unload()

        meta = storage.read_metadata(video_id)
        counts = {k: sum(1 for it in items if it["kind"] == k) for k in ("scene", "dialogue", "text")}
        paths = storage.video_paths(video_id)
        _save_vectors(paths.embeddings_vectors, vectors)
        storage.write_json(paths.embeddings_index, {
            "video_id": video_id,
            "model": settings.embed_model,
            "dim": int(vectors.shape[1]),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counts": counts,
            "tag_llm_weight": settings.tag_llm_weight,
            "items": items,
            "tag_scores": tag_scores,
        })
        storage.update_metadata(video_id, embed={
            "model": settings.embed_model, "dim": int(vectors.shape[1]), "items": len(items), **counts,
            "inputs": _inputs(meta),
        })
    progress(NAME, 1.0, "done")


# =============================================================== items
def build_items(scenes: list[dict[str, Any]], timeline: dict[str, Any]) -> list[dict[str, Any]]:
    """Everything worth searching, as {id, kind, scene, start_ms, end_ms, text, ref}."""
    items: list[dict[str, Any]] = []
    story = [sc for sc in scenes if "error" not in sc]

    for sc in story:
        items.append({"id": f"scene:{sc['scene']}", "kind": "scene", "scene": sc["scene"],
                      "start_ms": sc["start_ms"], "end_ms": sc["end_ms"],
                      "text": scene_text(sc), "ref": sc["scene"]})

    for d in timeline.get("dialogue", []):
        if d.get("reliable") and d.get("text", "").strip():
            items.append({"id": f"dialogue:{d['id']}", "kind": "dialogue",
                          "scene": scene_at(scenes, d["start_ms"], d["end_ms"]),
                          "start_ms": d["start_ms"], "end_ms": d["end_ms"],
                          "text": d["text"].strip(), "ref": d["id"]})

    for t in timeline.get("on_screen_text", []):
        if t.get("text", "").strip():
            items.append({"id": f"text:{t['id']}", "kind": "text",
                          "scene": scene_at(scenes, t["start_ms"], t["end_ms"]),
                          "start_ms": t["start_ms"], "end_ms": t["end_ms"],
                          "text": t["text"].strip(), "ref": t["id"]})
    return items


def scene_text(sc: dict[str, Any]) -> str:
    """One paragraph that says what the scene is about (what gets embedded for a scene)."""
    parts = [f"{sc.get('title', '')}.", sc.get("description", "")]
    if sc.get("events"):
        parts.append("Key moments: " + "; ".join(sc["events"]) + ".")
    if sc.get("characters"):
        parts.append("Characters: " + "; ".join(sc["characters"]) + ".")
    if sc.get("location"):
        parts.append(f"Location: {sc['location']} ({sc.get('time_of_day', 'unknown')}).")
    if sc.get("mood"):
        parts.append(f"Mood: {sc['mood']}.")
    tags = [k.replace("_", " ") for k, v in sorted(sc.get("tags", {}).items(), key=lambda kv: -kv[1]) if v >= 0.5]
    if tags:
        parts.append("Tags: " + ", ".join(tags) + ".")
    return " ".join(p for p in parts if p and p != ".")


def scene_at(scenes: list[dict[str, Any]], start_ms: int, end_ms: int) -> int | None:
    """Number of the scene that contains the item's midpoint."""
    mid = (start_ms + end_ms) // 2
    for sc in scenes:
        if sc["start_ms"] <= mid < sc["end_ms"]:
            return sc["scene"]
    return scenes[-1]["scene"] if scenes and mid >= scenes[-1]["end_ms"] else None


# =============================================================== tag scores
def compute_tag_scores(scenes: list[dict[str, Any]], items: list[dict[str, Any]],
                       vectors: np.ndarray, embedder) -> dict[str, dict[str, float]]:
    """{scene number (str): {tag: score 0..1}} for story scenes (not title cards)."""
    rows = {it["scene"]: i for i, it in enumerate(items) if it["kind"] == "scene"}
    story = [sc for sc in scenes if "error" not in sc and not sc.get("graphic") and sc["scene"] in rows]
    if not story:
        return {}
    scene_vecs = vectors[[rows[sc["scene"]] for sc in story]]
    tag_vecs = np.vstack([embedder.embed_query(tag_query(t)) for t in s07_scenes.TAGS])
    sims = scene_vecs @ tag_vecs.T                               # (scenes, tags) cosine similarity
    ranks = rank_columns(sims)
    w = settings.tag_llm_weight
    out: dict[str, dict[str, float]] = {}
    for i, sc in enumerate(story):
        out[str(sc["scene"])] = {
            t: round(w * float(sc.get("tags", {}).get(t, 0.0)) + (1 - w) * float(ranks[i, j]), 3)
            for j, t in enumerate(s07_scenes.TAGS)
        }
    return out


def tag_query(tag: str) -> str:
    return f"a movie scene with {tag.replace('_', ' ')}: {s07_scenes.TAG_DEFINITIONS.get(tag, '')}"


def rank_columns(m: np.ndarray) -> np.ndarray:
    """Per column: 0 for the lowest value .. 1 for the highest (0.5 everywhere if only one row)."""
    n = m.shape[0]
    if n == 1:
        return np.full_like(m, 0.5)
    order = m.argsort(axis=0).argsort(axis=0)                     # 0 .. n-1 within each column
    return order.astype(np.float32) / (n - 1)


# =============================================================== loading / saving
def load_index(video_id: str) -> tuple[dict[str, Any], np.ndarray]:
    """(index.json content, vectors matrix). Empty if stage 8 hasn't run."""
    paths = storage.video_paths(video_id)
    if not paths.embeddings_index.exists() or not paths.embeddings_vectors.exists():
        return {}, np.zeros((0, 0), dtype=np.float32)
    return storage.read_json(paths.embeddings_index), np.load(paths.embeddings_vectors)


def _save_vectors(path, vectors: np.ndarray) -> None:
    """Atomic write, like write_json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.save(buf, vectors.astype(np.float32))
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)


def _inputs(meta: dict[str, Any]) -> dict[str, Any]:
    stages = meta.get("stages", {})
    return {"scenes": stages.get("scenes", {}).get("updated_at"),
            "merge": stages.get("merge", {}).get("updated_at"),
            "model": settings.embed_model}
