"""Stage 7 — Scenes: group shots into scenes, then describe each scene with a local LLM.

Input : timeline.json (stage 6), keyframe images (stage 4)
Output: data/videos/{id}/scenes.json
        metadata.json -> scenes summary, stages.scenes

Part 1 — segmentation (code, no AI). Between every two neighbouring shots we score
"how likely is a scene change here?":
    look      colour-histogram distance between the shots' keyframes (0 same .. 1 different)
    dialogue  a line spanning the cut (or < 2 s gap) pulls the shots together;
              5+ s of silence around the cut pushes them apart
    text      on-screen text spanning the cut pulls them together
Cuts are made where the score passes config.scene_cut_threshold, respecting a minimum
and maximum scene length. Title cards / credits (graphic frames) become their own scenes.

Part 2 — description (LLM via Ollama, default qwen2.5:3b). Each scene's dialogue,
on-screen text and visual notes go to the model, which returns structured JSON:
title, description, characters, location, time of day, mood, events, tag confidences.
Foreign/unreliable speech is replaced by "[speech in another language]"; visual notes
are marked as possibly wrong. Evidence ids (dialogue / text) are attached by code, so
answers can later quote the exact lines.
"""
from __future__ import annotations

import difflib
import gc
import logging
import time
from typing import Any, Optional

from app.adapters import local_storage as storage
from app.adapters.ollama_llm import OllamaLLM
from app.adapters.ollama_vision import OllamaError, OllamaUnreachable
from app.config import settings
from app.pipeline.base import ProgressCallback, StageError, no_progress, require_stage_done, track_stage
from app.pipeline.stages import s06_merge

log = logging.getLogger(__name__)

NAME = "scenes"

TAGS = [
    "dialogue", "action", "fight", "chase", "battle", "romance", "comedy", "sad", "suspense",
    "horror", "violence", "celebration", "religious_ritual", "speech_to_crowd", "travel",
    "nature_landscape", "dream_or_vision", "death", "betrayal", "reunion",
]

SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "characters": {"type": "array", "items": {"type": "string"}},
        "location": {"type": "string"},
        "time_of_day": {"type": "string"},
        "mood": {"type": "string"},
        "events": {"type": "array", "items": {"type": "string"}},
        # Only the tags that apply (small models rate every tag ~1.0 when asked to score all 20)
        "tags": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {"tag": {"type": "string", "enum": TAGS},
                               "evidence": {"type": "string"},
                               "confidence": {"type": "number"}},
                "required": ["tag", "evidence", "confidence"],
            },
        },
    },
    "required": ["title", "description", "characters", "location", "time_of_day", "mood", "events", "tags"],
}

TAG_GUIDE = "\n".join(f"  - {t}: {d}" for t, d in {
    "dialogue": "people talk to each other for most of the scene",
    "speech_to_crowd": "one person addresses a group or crowd",
    "suspense": "tension, threat or something about to happen",
    "sad": "grief, crying, loss",
    "religious_ritual": "prayer, ceremony, prophecy, worship",
    "action": "fast physical movement: running, explosions, vehicles, combat. NOT standing, talking or walking",
    "fight": "people physically hit or attack each other",
    "chase": "someone pursues someone",
    "battle": "armies or groups in combat",
    "violence": "injury, killing or blood is shown",
    "horror": "monsters, gore or terror is SHOWN (a dark or tense mood alone is not horror)",
    "romance": "affection between two people: kissing, embracing, love talk",
    "comedy": "jokes, laughter",
    "celebration": "cheering, party, festivity",
    "travel": "journey, characters moving across places",
    "nature_landscape": "landscape or nature shots dominate",
    "dream_or_vision": "dream, vision or hallucination",
    "death": "a character dies or a body is shown",
    "betrayal": "someone is betrayed or accused of betrayal",
    "reunion": "people meet again after a separation",
}.items())

SYSTEM_PROMPT = f"""You describe one scene of a movie from automatically extracted notes.
How to use the notes:
- DIALOGUE comes from speech recognition. "[speech in another language]" means the audio was not in the main language.
- ON-SCREEN TEXT is read from the picture. Text that looks like a subtitle is what a character is saying; use it as dialogue, especially when the speech is in another language.
- VISUAL NOTES come from an automatic image model and MAY CONTAIN ERRORS (it sometimes invents microphones, stages, concerts, phones). Trust a visual detail only if several notes agree or the dialogue supports it.
- Never invent names. If a name is spoken or shown on screen you may use it; otherwise describe people ("young man in a dark robe").
Reply ONLY with JSON:
- title: 2-6 words.
- description: 3-5 sentences: what happens, who is involved, what is said, how it ends.
- characters: people present, by name if known, else short descriptions (max 6).
- location: where it takes place. time_of_day: day, night, dusk, dawn, or unknown.
- mood: 1-3 words. events: 1-5 short key moments in order.
- tags: pick ONLY the 0-4 tags that clearly fit, from the list below. For each give "evidence": a short quote
  or note from the scene notes that proves it. No evidence -> do not add the tag. An empty list is fine.
  confidence: 0.9 = shown clearly, 0.6 = likely, 0.3 = possible.
  Tag meanings (be strict):
{TAG_GUIDE}"""


def is_done(video_id: str) -> bool:
    meta = storage.read_metadata(video_id)
    return (meta.get("stages", {}).get(NAME, {}).get("status") == "done"
            and meta.get("scenes", {}).get("inputs") == _inputs(meta)
            and not meta.get("scenes", {}).get("errors"))          # failed scenes are retried


def run(video_id: str, *, force: bool = False, progress: ProgressCallback = no_progress) -> None:
    if not force and is_done(video_id):
        log.info("[%s] %s already done, skipping", video_id, NAME)
        progress(NAME, 1.0, "already done")
        return

    require_stage_done(video_id, "merge")
    paths = storage.video_paths(video_id)
    timeline = s06_merge.load_timeline(video_id)
    if not timeline.get("shots"):
        raise StageError("timeline.json has no shots. Re-run merge with --force.")

    gc.collect()   # free GPU memory from earlier stages in this process
    llm = OllamaLLM(settings.scene_llm_model, settings.ollama_url, settings.vision_timeout_s,
                    settings.vision_retries, settings.scene_llm_ctx)
    try:
        llm.check_ready()
    except OllamaError as exc:
        raise StageError(str(exc)) from exc

    with track_stage(video_id, NAME):
        # ---- Part 1: segmentation
        progress(NAME, 0.0, "finding scene boundaries")
        shots = timeline["shots"]
        scores = boundary_scores(shots, paths)
        groups = segment(shots, scores)
        log.info("[%s] %d shots -> %d scenes", video_id, len(shots), len(groups))

        # ---- Part 2: describe (resume: keep scenes with the same range + model)
        previous = storage.read_json(paths.scenes)
        done_before = {}
        if not force and previous.get("model") == settings.scene_llm_model \
                and previous.get("prompt_version") == PROMPT_VERSION:
            done_before = {(s["start_ms"], s["end_ms"]): s for s in previous.get("scenes", []) if "error" not in s}

        scenes: list[dict[str, Any]] = []
        seconds: list[float] = []
        try:
            for n, (first, last) in enumerate(groups, 1):
                scene_shots = shots[first:last + 1]
                base = _scene_base(n, scene_shots, scores, first)
                key = (base["start_ms"], base["end_ms"])
                if key in done_before:
                    scenes.append({**done_before[key], "scene": n})
                else:
                    started = time.monotonic()
                    scenes.append({**base, **describe_scene(llm, base, scene_shots, timeline)})
                    seconds.append(time.monotonic() - started)
                _save(paths, video_id, timeline, scenes, complete=False)
                progress(NAME, n / len(groups), f"scene {n}/{len(groups)}")
        finally:
            llm.unload()

        _save(paths, video_id, timeline, scenes, complete=True)
        storage.update_metadata(video_id, scenes={
            "model": settings.scene_llm_model,
            "scenes": len(scenes),
            "errors": sum(1 for s in scenes if "error" in s),
            "avg_scene_ms": int(sum(s["end_ms"] - s["start_ms"] for s in scenes) / max(len(scenes), 1)),
            "avg_seconds_per_scene": round(sum(seconds) / len(seconds), 1) if seconds else None,
            "inputs": _inputs(storage.read_metadata(video_id)),
        })

    progress(NAME, 1.0, "done")


def load_scenes(video_id: str) -> list[dict[str, Any]]:
    return storage.read_json(storage.video_paths(video_id).scenes).get("scenes", [])


# =============================================================== Part 1: segmentation
def _shot_histogram(shot: dict[str, Any], paths: storage.VideoPaths):
    """Average HSV colour histogram of a shot's keyframes (None if no images)."""
    import cv2
    hists = []
    for v in shot.get("visual", []):
        img = cv2.imread(str(paths.root / v["file"]))
        if img is None:
            continue
        hsv = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 4], [0, 180, 0, 256, 0, 256])
        hists.append(cv2.normalize(h, h).flatten())
    if not hists:
        return None
    avg = sum(hists) / len(hists)
    return avg


def boundary_scores(shots: list[dict[str, Any]], paths: storage.VideoPaths) -> list[float]:
    """scores[i] = how likely a scene change is between shot i and shot i+1 (higher = more likely)."""
    import cv2
    hists = [_shot_histogram(s, paths) for s in shots]
    scores = []
    for i in range(len(shots) - 1):
        a, b = shots[i], shots[i + 1]
        cut = a["end_ms"]

        if hists[i] is not None and hists[i + 1] is not None:
            look = float(cv2.compareHist(hists[i], hists[i + 1], cv2.HISTCMP_BHATTACHARYYA))
        else:
            look = 0.5

        a_ids = {d["id"] for d in a["dialogue"]}
        spans = any(d["id"] in a_ids for d in b["dialogue"])          # one line across the cut
        last_a = max((d["end_ms"] for d in a["dialogue"]), default=None)
        first_b = min((d["start_ms"] for d in b["dialogue"]), default=None)
        talk_continues = spans or (last_a is not None and first_b is not None and first_b - last_a < 2000)
        near = [d for d in a["dialogue"] + b["dialogue"] if abs((d["start_ms"] + d["end_ms"]) / 2 - cut) < 5000]
        silence = not near

        t_ids = {t["id"] for t in a["on_screen_text"]}
        text_spans = any(t["id"] in t_ids for t in b["on_screen_text"])

        score = look + (0.3 if silence else 0.0) - (0.4 if talk_continues else 0.0) - (0.3 if text_spans else 0.0)
        if _is_graphic(a) != _is_graphic(b):
            score = 2.0                                              # always cut around title cards / credits
        scores.append(round(score, 3))
    return scores


def _is_graphic(shot: dict[str, Any]) -> bool:
    vis = shot.get("visual", [])
    return bool(vis) and all(v.get("graphic") for v in vis)


def segment(shots: list[dict[str, Any]], scores: list[float]) -> list[tuple[int, int]]:
    """Greedy grouping into (first_shot_index, last_shot_index) pairs."""
    groups: list[tuple[int, int]] = []
    start = 0
    for i, score in enumerate(scores):
        length = shots[i]["end_ms"] - shots[start]["start_ms"]
        next_len = shots[i + 1]["end_ms"] - shots[start]["start_ms"]
        forced = score >= 2.0
        if forced or (score >= settings.scene_cut_threshold and length >= settings.scene_min_ms) \
                or next_len > settings.scene_max_ms:
            groups.append((start, i))
            start = i + 1
    groups.append((start, len(shots) - 1))

    # Merge a too-short trailing scene into the previous one (unless it's a graphic scene)
    if len(groups) > 1:
        f, l = groups[-1]
        if shots[l]["end_ms"] - shots[f]["start_ms"] < settings.scene_min_ms // 2 and not _is_graphic(shots[f]):
            pf, _ = groups[-2]
            if not _is_graphic(shots[pf]):
                groups[-2:] = [(pf, l)]
    return groups


# =============================================================== Part 2: describe
def _scene_base(n: int, scene_shots: list[dict[str, Any]], scores: list[float], first: int) -> dict[str, Any]:
    dialogue = _unique([d for s in scene_shots for d in s["dialogue"]])
    texts = _unique([t for s in scene_shots for t in s["on_screen_text"]])
    return {
        "scene": n,
        "start_ms": scene_shots[0]["start_ms"],
        "end_ms": scene_shots[-1]["end_ms"],
        "shots": [scene_shots[0]["shot"], scene_shots[-1]["shot"]],
        "boundary_score": scores[first - 1] if first > 0 else None,   # score of the cut that started it
        "graphic": all(_is_graphic(s) for s in scene_shots),
        "evidence": {
            "dialogue_ids": [d["id"] for d in dialogue if d["reliable"]],
            "unreliable_dialogue_ids": [d["id"] for d in dialogue if not d["reliable"]],
            "text_ids": [t["id"] for t in texts],
            "frames": [v["file"] for s in scene_shots for v in s["visual"]],
        },
    }


def build_prompt(base: dict[str, Any], scene_shots: list[dict[str, Any]]) -> str:
    fmt = lambda ms: f"{ms // 60000}:{ms // 1000 % 60:02d}"
    dialogue = _unique([d for s in scene_shots for d in s["dialogue"]])
    texts = _unique([t for s in scene_shots for t in s["on_screen_text"]])
    visuals = [v for s in scene_shots for v in s["visual"] if not v.get("graphic")]

    lines = [f"SCENE {base['scene']}: {fmt(base['start_ms'])} to {fmt(base['end_ms'])} "
             f"({(base['end_ms'] - base['start_ms']) // 1000} s, {len(scene_shots)} shots)", ""]
    lines.append("DIALOGUE:")
    if dialogue:
        for d in dialogue:
            said = f'"{d["text"]}"' if d["reliable"] else "[speech in another language]"
            lines.append(f"[{fmt(d['start_ms'])}] {said}")
    else:
        lines.append("(no dialogue)")

    lines += ["", "ON-SCREEN TEXT:"]
    lines += [f"[{fmt(t['start_ms'])}] \"{t['text']}\"" for t in texts] or ["(none)"]

    lines += ["", "VISUAL NOTES (may contain errors):"]
    kept: list[dict[str, Any]] = []
    for v in visuals:   # drop near-duplicate descriptions
        if v["description"] and not any(
                difflib.SequenceMatcher(None, v["description"], k["description"]).ratio() > 0.8 for k in kept):
            kept.append(v)
    if len(kept) > settings.scene_max_visuals:   # spread evenly over the scene
        step = len(kept) / settings.scene_max_visuals
        kept = [kept[int(i * step)] for i in range(settings.scene_max_visuals)]
    lines += [f"[{fmt(v['ts_ms'])}] {v['description'][:300]}" for v in kept] or ["(none)"]

    objects = sorted({o for v in visuals for o in v.get("objects", [])})
    if objects:
        lines += ["", "OBJECTS SEEN: " + ", ".join(objects[:25])]
    return "\n".join(lines)


def describe_scene(llm, base: dict[str, Any], scene_shots: list[dict[str, Any]],
                   timeline: dict[str, Any]) -> dict[str, Any]:
    if base["graphic"]:   # title card / credits: no story content, no LLM needed
        texts = _unique([t for s in scene_shots for t in s["on_screen_text"]])
        return {"title": "Title card / credits", "description": " / ".join(t["text"] for t in texts),
                "characters": [], "location": "", "time_of_day": "unknown", "mood": "",
                "events": [], "tags": {t: 0.0 for t in TAGS}}
    try:
        raw = llm.generate_json(SYSTEM_PROMPT, build_prompt(base, scene_shots), SCENE_SCHEMA)
        return _normalize(raw)
    except OllamaUnreachable as exc:
        raise StageError(f"Lost connection to Ollama ({exc}). Start Ollama and re-run; finished scenes are kept.") from exc
    except Exception as exc:  # noqa: BLE001 — one bad scene must not stop the movie
        log.warning("scene %d: %s", base["scene"], exc)
        return {"error": str(exc)}


def _normalize(d: dict[str, Any]) -> dict[str, Any]:
    s = lambda v: v.strip() if isinstance(v, str) else ""
    lst = lambda v, n: [s(x) for x in (v or []) if isinstance(x, str) and s(x)][:n]
    picked: dict[str, float] = {}
    reasons: dict[str, str] = {}
    raw_tags = d.get("tags") or []
    if isinstance(raw_tags, dict):                       # tolerate the old {tag: confidence} shape
        raw_tags = [{"tag": k, "confidence": v} for k, v in raw_tags.items()]
    for item in raw_tags:
        if isinstance(item, dict) and item.get("tag") in TAGS:
            if "evidence" in item and not s(item.get("evidence")):
                continue                                   # a tag without evidence is a guess: drop it
            try:
                conf = min(max(float(item.get("confidence") or 0.0), 0.0), 1.0)
            except (TypeError, ValueError):
                continue
            if conf > picked.get(item["tag"], -1):
                picked[item["tag"]] = conf
                reasons[item["tag"]] = s(item.get("evidence"))
    return {
        "title": s(d.get("title")),
        "description": s(d.get("description")),
        "characters": lst(d.get("characters"), 6),
        "location": s(d.get("location")),
        "time_of_day": s(d.get("time_of_day")) or "unknown",
        "mood": s(d.get("mood")),
        "events": lst(d.get("events"), 5),
        "tags": {t: round(picked.get(t, 0.0), 2) for t in TAGS},   # full dict: every tag, 0.0 if not picked
        "tag_evidence": reasons,                                    # why each picked tag applies
    }


# =============================================================== helpers
def _unique(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x["start_ms"]):
        if it["id"] not in seen:
            seen.add(it["id"])
            out.append(it)
    return out


def _save(paths, video_id, timeline, scenes, *, complete: bool) -> None:
    storage.write_json(paths.scenes, {
        "video_id": video_id,
        "model": settings.scene_llm_model,
        "prompt_version": PROMPT_VERSION,
        "complete": complete,
        "segmentation": {"cut_threshold": settings.scene_cut_threshold,
                         "min_ms": settings.scene_min_ms, "max_ms": settings.scene_max_ms},
        "tags_vocabulary": TAGS,
        "scenes": scenes,
    })


PROMPT_VERSION = 3   # bump when the prompt/schema changes so scenes are re-described


def _inputs(meta: dict[str, Any]) -> dict[str, Any]:
    return {"merge": meta.get("stages", {}).get("merge", {}).get("updated_at"),
            "model": settings.scene_llm_model, "prompt_version": PROMPT_VERSION}
