"""Stage 5 — Caption keyframes with a local vision model (per chunk).

Input : data/videos/{id}/shots/c000.json   keyframe list   (stage 4)
        data/videos/{id}/frames/*.jpg       keyframe images (stage 4)
Output: data/videos/{id}/captions/c000.json one structured description per keyframe
        chunks.json   -> chunks[i].stages.caption = "done"
        metadata.json -> captions summary, stages.caption

Engine (config.caption_engine):
    "florence" (default)  Florence-2 in-process on the GPU: caption + OCR + objects, ~0.5 s/frame
    "ollama"              Qwen2.5-VL via Ollama: richer JSON (people, mood...), needs a bigger GPU
Saves after EVERY frame, so an interruption loses at most one frame; re-running
continues where it stopped. A frame that fails is recorded with an "error" and
skipped instead of failing the whole movie; the next run retries only those frames.
Switching engine/model re-captions automatically (results are tagged with the model).
"""
from __future__ import annotations

import gc
import logging
import time
from typing import Any

from app.adapters import local_storage as storage
from app.adapters.ollama_vision import OllamaError, OllamaUnreachable, OllamaVision
from app.config import settings
from app.pipeline.base import ProgressCallback, StageError, no_progress, require_stage_done, track_stage

log = logging.getLogger(__name__)

NAME = "caption"


def is_done(video_id: str) -> bool:
    # Done only if finished with the CURRENTLY configured model; switching model re-captions.
    meta = storage.read_metadata(video_id)
    captions = meta.get("captions", {})
    return (meta.get("stages", {}).get(NAME, {}).get("status") == "done"
            and captions.get("model") == _model_name()
            and not captions.get("errors"))          # failed frames are retried on the next run


def run(video_id: str, *, force: bool = False, progress: ProgressCallback = no_progress) -> None:
    if not force and is_done(video_id):
        log.info("[%s] %s already done, skipping", video_id, NAME)
        progress(NAME, 1.0, "already done")
        return

    require_stage_done(video_id, "shots")
    paths = storage.video_paths(video_id)

    # Free GPU memory held by earlier stages in this process (e.g. Whisper) — a 4 GB GPU
    # can hold only one model at a time.
    gc.collect()

    vision = _make_engine()
    model_name = _model_name()

    with track_stage(video_id, NAME):
        chunks = storage.read_chunks(video_id)["chunks"]
        todo = [c for c in chunks if force or not _chunk_done(paths, c, model_name)]
        total_frames = sum(len(_keyframes(paths, c)) for c in todo)
        log.info("[%s] %d chunk(s), %d keyframe(s) to caption with %s",
                 video_id, len(todo), total_frames, model_name)

        done_frames = 0
        seconds: list[float] = []
        try:
            for chunk in todo:
                keyframes = _keyframes(paths, chunk)
                out_path = paths.captions(chunk["id"])

                # Resume: keep frames already captioned by the SAME model (unless --force)
                previous = storage.read_json(out_path)
                existing = {} if force or previous.get("model") != model_name else {
                    f["ts_ms"]: f for f in previous.get("frames", []) if "error" not in f
                }
                results: dict[int, dict[str, Any]] = dict(existing)
                pending = [kf for kf in keyframes if kf["ts_ms"] not in existing]
                done_frames += len(keyframes) - len(pending)
                batch_size = max(1, settings.caption_batch_size) if hasattr(vision, "describe_batch") else 1

                for i in range(0, len(pending), batch_size):
                    batch = pending[i:i + batch_size]
                    started = time.monotonic()
                    for kf, desc in zip(batch, _describe(vision, paths, batch, video_id)):
                        results[kf["ts_ms"]] = {"ts_ms": kf["ts_ms"], "file": kf["file"], **desc}
                    seconds.append((time.monotonic() - started) / len(batch))
                    done_frames += len(batch)

                    frames = [results[kf["ts_ms"]] for kf in keyframes if kf["ts_ms"] in results]
                    _save(out_path, chunk, frames, model_name, complete=False)   # save after every batch
                    progress(NAME, done_frames / max(total_frames, 1),
                             f"chunk {chunk['index'] + 1}/{len(chunks)}: frame {done_frames}/{total_frames}")

                frames = [results[kf["ts_ms"]] for kf in keyframes if kf["ts_ms"] in results]
                _save(out_path, chunk, frames, model_name, complete=True)
                storage.set_chunk_stage_status(video_id, chunk["id"], NAME, "done")
                log.info("[%s] %s: %d frame(s) captioned", video_id, chunk["id"], len(frames))
        finally:
            vision.unload()   # give the GPU back for the next stage

        all_frames = load_captions(video_id)
        storage.update_metadata(video_id, captions={
            "engine": settings.caption_engine,
            "model": model_name,
            "frames": len(all_frames),
            "with_text": sum(1 for f in all_frames if f.get("on_screen_text")),
            "errors": sum(1 for f in all_frames if "error" in f),
            "avg_seconds_per_frame": round(sum(seconds) / len(seconds), 2) if seconds else None,
        })

    progress(NAME, 1.0, "done")


# ---------------------------------------------------------------- reading results
def load_captions(video_id: str) -> list[dict[str, Any]]:
    """All captioned keyframes of the movie in time order."""
    paths = storage.video_paths(video_id)
    frames: list[dict[str, Any]] = []
    for chunk in storage.read_chunks(video_id).get("chunks", []):
        frames.extend(storage.read_json(paths.captions(chunk["id"])).get("frames", []))
    frames.sort(key=lambda f: f["ts_ms"])
    return frames


# ---------------------------------------------------------------- helpers
def _keyframes(paths: storage.VideoPaths, chunk: dict[str, Any]) -> list[dict[str, Any]]:
    shots = storage.read_json(paths.shots(chunk["id"])).get("shots", [])
    return [kf for s in shots for kf in s["keyframes"]]


def _describe(vision, paths, batch: list[dict[str, Any]], video_id: str) -> list[dict[str, Any]]:
    """Describe a batch of keyframes. If the batch fails, retry one by one so a single
    bad frame only marks itself as an error."""
    images = [paths.root / kf["file"] for kf in batch]
    if len(batch) > 1:
        try:
            return vision.describe_batch(images)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] batch of %d failed (%s); retrying frames one by one", video_id, len(batch), exc)
    out = []
    for kf, image in zip(batch, images):
        try:
            out.append(vision.describe(image))
        except OllamaUnreachable as exc:
            raise StageError(f"Lost connection to Ollama ({exc}). Start Ollama and re-run; "
                             "finished frames are kept.") from exc
        except Exception as exc:  # noqa: BLE001 — one bad frame must not stop the movie
            log.warning("[%s] %s: %s", video_id, kf["file"], exc)
            out.append({"error": str(exc)})
    return out


def _model_name() -> str:
    return settings.florence_model if settings.caption_engine == "florence" else settings.vision_model


def _make_engine():
    """Create the captioning engine chosen in config. Both have describe(image) and unload()."""
    if settings.caption_engine == "florence":
        from app.adapters.florence_vision import FlorenceVision   # imports torch: only when needed
        return FlorenceVision(settings.florence_model, settings.florence_device, settings.florence_objects,
                              caption_beams=settings.florence_caption_beams)
    if settings.caption_engine == "ollama":
        vision = OllamaVision(settings.vision_model, settings.ollama_url,
                              settings.vision_timeout_s, settings.vision_retries)
        try:
            vision.check_ready()
        except OllamaError as exc:
            raise StageError(str(exc)) from exc
        return vision
    raise StageError(f"Unknown caption engine {settings.caption_engine!r} (use 'florence' or 'ollama').")


def _save(path, chunk: dict[str, Any], frames: list[dict[str, Any]], model: str, *, complete: bool) -> None:
    storage.write_json(path, {
        "chunk_id": chunk["id"],
        "model": model,
        "complete": complete,
        "frames": frames,
    })


def _chunk_done(paths: storage.VideoPaths, chunk: dict[str, Any], model: str) -> bool:
    """Done = marked done AND complete AND made by the current model (switching models re-captions)."""
    data = storage.read_json(paths.captions(chunk["id"]))
    return (chunk.get("stages", {}).get(NAME) == "done"
            and data.get("complete") is True and data.get("model") == model
            and not any("error" in f for f in data.get("frames", [])))
