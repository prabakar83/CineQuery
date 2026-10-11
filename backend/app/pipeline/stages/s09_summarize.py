"""Stage 9 — Summarize: chapter summaries and a whole-movie profile.

Input : scenes.json (stage 7), timeline.json (stage 6)
Output: data/videos/{id}/summaries.json
        metadata.json -> summarize summary, stages.summarize

Why bottom-up: a 2-hour movie (~200 scenes) doesn't fit in one LLM call on a small model.
    1. Chapters: consecutive scenes grouped into ~10-minute chapters (scene boundaries kept).
       The LLM summarizes each chapter from its scene descriptions + the key quoted lines.
    2. Profile: one more call reads all chapter summaries (+ tag statistics as hard evidence)
       and returns genre(s) with evidence, logline, plot, main characters, themes, setting,
       tone and ending.
About 13 LLM calls for a 2-hour movie. Saves after every chapter; failed calls are retried
on the next run, finished chapters are kept.
"""
from __future__ import annotations

import gc
import logging
import time
from typing import Any

from app.adapters import local_storage as storage
from app.adapters.ollama_llm import OllamaLLM
from app.adapters.ollama_vision import OllamaError, OllamaUnreachable
from app.config import settings
from app.pipeline.base import ProgressCallback, StageError, no_progress, require_stage_done, track_stage
from app.pipeline.stages import s06_merge, s07_scenes

log = logging.getLogger(__name__)

NAME = "summarize"
PROMPT_VERSION = 2   # bump when prompts/schemas change so summaries are rewritten

GENRES = [
    "action", "adventure", "animation", "comedy", "crime", "documentary", "drama", "family", "fantasy",
    "history", "horror", "music", "mystery", "romance", "science_fiction", "thriller", "war", "western",
]

CHAPTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "events": {"type": "array", "items": {"type": "string"}},
        "characters": {"type": "array", "items": {"type": "string"}},
        "locations": {"type": "array", "items": {"type": "string"}},
        "mood": {"type": "string"},
    },
    "required": ["title", "summary", "events", "characters", "locations", "mood"],
}

PROFILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_type": {"type": "string", "enum": ["feature_film", "trailer", "clip_or_excerpt", "episode", "other"]},
        "genres": {"type": "array", "maxItems": 3, "items": {
            "type": "object",
            "properties": {"genre": {"type": "string", "enum": GENRES},
                           "evidence": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["genre", "evidence", "confidence"]}},
        "logline": {"type": "string"},
        "plot": {"type": "string"},
        "characters": {"type": "array", "maxItems": 10, "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "description": {"type": "string"}},
            "required": ["name", "role", "description"]}},
        "themes": {"type": "array", "items": {"type": "string"}},
        "setting": {"type": "string"},
        "tone": {"type": "string"},
        "ending": {"type": "string"},
    },
    "required": ["video_type", "genres", "logline", "plot", "characters", "themes", "setting", "tone", "ending"],
}

CHAPTER_SYSTEM = """You summarize one chapter of a movie from automatically written scene notes.
- SCENES are short descriptions written by another model from speech, subtitles and images; small visual details may be wrong.
- QUOTES are lines spoken (speech recognition) or shown as subtitles; they are the most reliable evidence.
- Never invent names: use a name only if it appears in the notes; otherwise describe the person.
Reply ONLY with JSON:
- title: 2-6 words.
- summary: 4-8 sentences telling what happens in order, who is involved and why it matters.
- events: 3-8 key moments in order, each one short sentence.
- characters: people who matter in this chapter (max 8). locations: places (max 5). mood: 1-3 words."""

PROFILE_SYSTEM = f"""You write a profile of a whole video from its chapter summaries.
- Base everything on the chapters and quotes given. Never invent names, places or events.
- If the video is short or reads like a trailer or excerpt, say so in video_type and describe only what is shown.
- TAG STATISTICS are counted by code (share of scenes with that tag); use them as evidence for genres.
Reply ONLY with JSON:
- video_type: feature_film, trailer, clip_or_excerpt, episode or other.
- genres: 1-3 from this list: {", ".join(GENRES)}. Each with "evidence" (what in the chapters shows it)
  and confidence (0.9 clear, 0.6 likely, 0.3 possible).
- logline: one sentence. plot: one paragraph per chapter or less, in order.
- characters: main people (max 10): name (or short description if unnamed), role (e.g. protagonist,
  leader, antagonist, mentor), one-sentence description. A crowd or group is not a character.
  Call someone an antagonist only if the chapters show them opposing someone; if the role is unclear, write "unclear".
  The same person may be called by several names (a title, a nickname); do not list them twice if the notes link them.
- themes: 2-5 short phrases. setting: where and when. tone: 1-3 words.
- ending: how the video ends. Use FINAL MOMENTS (the last scenes, in order), not the beginning."""


# =============================================================== stage
def is_done(video_id: str) -> bool:
    meta = storage.read_metadata(video_id)
    info = meta.get(NAME, {})
    return (meta.get("stages", {}).get(NAME, {}).get("status") == "done"
            and storage.video_paths(video_id).summaries.exists()
            and info.get("inputs") == _inputs(meta)
            and not info.get("errors"))                     # failed chapters / profile are retried


def run(video_id: str, *, force: bool = False, progress: ProgressCallback = no_progress) -> None:
    if not force and is_done(video_id):
        log.info("[%s] %s already done, skipping", video_id, NAME)
        progress(NAME, 1.0, "already done")
        return

    require_stage_done(video_id, "scenes")
    paths = storage.video_paths(video_id)
    scenes = s07_scenes.load_scenes(video_id)
    timeline = s06_merge.load_timeline(video_id)
    if not scenes:
        raise StageError("scenes.json has no scenes. Re-run: python cli.py scenes <id> --force")

    gc.collect()
    llm = OllamaLLM(settings.summary_llm_model, settings.ollama_url, settings.summary_timeout_s,
                    settings.vision_retries, settings.summary_llm_ctx)
    try:
        llm.check_ready()
    except OllamaError as exc:
        raise StageError(str(exc)) from exc

    meta = storage.read_metadata(video_id)
    with track_stage(video_id, NAME):
        groups = plan_chapters(scenes, settings.chapter_target_ms)
        log.info("[%s] %d scenes -> %d chapters", video_id, len(scenes), len(groups))

        previous = storage.read_json(paths.summaries)
        reuse = {}
        if not force and previous.get("inputs") == _inputs(meta):
            reuse = {(c["start_ms"], c["end_ms"]): c for c in previous.get("chapters", []) if "error" not in c}

        steps = len(groups) + 1
        chapters: list[dict[str, Any]] = []
        seconds: list[float] = []
        profile: dict[str, Any] = {}
        try:
            for n, (first, last) in enumerate(groups, 1):
                ch_scenes = scenes[first:last + 1]
                base = {"chapter": n, "start_ms": ch_scenes[0]["start_ms"], "end_ms": ch_scenes[-1]["end_ms"],
                        "scenes": [ch_scenes[0]["scene"], ch_scenes[-1]["scene"]]}
                key = (base["start_ms"], base["end_ms"])
                if key in reuse:
                    chapters.append({**reuse[key], "chapter": n})
                else:
                    started = time.monotonic()
                    chapters.append({**base, **summarize_chapter(llm, base, ch_scenes, timeline)})
                    seconds.append(time.monotonic() - started)
                _save(paths, video_id, meta, chapters, profile, complete=False)
                progress(NAME, n / steps, f"chapter {n}/{len(groups)}")

            progress(NAME, len(groups) / steps, "movie profile")
            started = time.monotonic()
            profile = build_profile(llm, chapters, scenes, timeline)
            seconds.append(time.monotonic() - started)
        finally:
            llm.unload()

        _save(paths, video_id, meta, chapters, profile, complete=True)
        errors = sum(1 for c in chapters if "error" in c) + (1 if "error" in profile else 0)
        storage.update_metadata(video_id, summarize={
            "model": settings.summary_llm_model,
            "chapters": len(chapters),
            "errors": errors,
            "genres": [g["genre"] for g in profile.get("genres", [])],
            "video_type": profile.get("video_type"),
            "avg_seconds_per_call": round(sum(seconds) / len(seconds), 1) if seconds else None,
            "inputs": _inputs(meta),
        })
    progress(NAME, 1.0, "done")


def load_summaries(video_id: str) -> dict[str, Any]:
    return storage.read_json(storage.video_paths(video_id).summaries)


# =============================================================== 1. chapters
def plan_chapters(scenes: list[dict[str, Any]], target_ms: int) -> list[tuple[int, int]]:
    """Group consecutive scenes into ~target_ms chapters: (first_index, last_index) pairs."""
    groups: list[tuple[int, int]] = []
    start = 0
    for i, sc in enumerate(scenes):
        if sc["end_ms"] - scenes[start]["start_ms"] >= target_ms and i < len(scenes) - 1:
            groups.append((start, i))
            start = i + 1
    groups.append((start, len(scenes) - 1))
    # A short last chapter (< 1/3 of the target) is folded into the previous one
    if len(groups) > 1:
        f, l = groups[-1]
        if scenes[l]["end_ms"] - scenes[f]["start_ms"] < target_ms // 3:
            groups[-2:] = [(groups[-2][0], l)]
    return groups


def build_chapter_prompt(base: dict[str, Any], ch_scenes: list[dict[str, Any]], timeline: dict[str, Any]) -> str:
    lines = [f"CHAPTER {base['chapter']}: {_fmt(base['start_ms'])} to {_fmt(base['end_ms'])} "
             f"({len(ch_scenes)} scenes)", "", "SCENES:"]
    for sc in ch_scenes:
        head = f"[{_fmt(sc['start_ms'])}-{_fmt(sc['end_ms'])}]"
        if "error" in sc:
            lines.append(f"{head} (no description)")
        elif sc.get("graphic"):
            lines.append(f"{head} Title card / credits: {sc.get('description') or '(no text)'}")
        else:
            tags = ", ".join(k for k, v in sc.get("tags", {}).items() if v >= 0.5)
            extra = f" Mood: {sc.get('mood', '')}." + (f" Tags: {tags}." if tags else "")
            lines.append(f"{head} {sc.get('title', '')}. {sc.get('description', '')}{extra}")

    lines += ["", "QUOTES:"]
    quotes = chapter_quotes(timeline, base["start_ms"], base["end_ms"], settings.chapter_max_quotes)
    lines += [f"[{_fmt(q['start_ms'])}] {q['label']}: \"{q['text'][:150]}\"" for q in quotes] or ["(none)"]
    return "\n".join(lines)


def chapter_quotes(timeline: dict[str, Any], start_ms: int, end_ms: int, limit: int) -> list[dict[str, Any]]:
    """Reliable dialogue + on-screen text inside the range, in time order, spread evenly if too many."""
    inside = lambda x: start_ms <= (x["start_ms"] + x["end_ms"]) // 2 < end_ms
    items = [{**d, "label": "says"} for d in timeline.get("dialogue", []) if d.get("reliable") and inside(d)]
    items += [{**t, "label": "on screen"} for t in timeline.get("on_screen_text", []) if inside(t)]
    items.sort(key=lambda x: x["start_ms"])
    if len(items) > limit:
        step = len(items) / limit
        items = [items[int(i * step)] for i in range(limit)]
    return items


def summarize_chapter(llm, base: dict[str, Any], ch_scenes: list[dict[str, Any]],
                      timeline: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = llm.generate_json(CHAPTER_SYSTEM, build_chapter_prompt(base, ch_scenes, timeline), CHAPTER_SCHEMA)
    except OllamaUnreachable as exc:
        raise StageError(f"Lost connection to Ollama ({exc}). Start Ollama and re-run; finished chapters are kept.") from exc
    except Exception as exc:  # noqa: BLE001 — one bad chapter must not stop the movie
        log.warning("chapter %d: %s", base["chapter"], exc)
        return {"error": str(exc)}
    return {
        "title": _s(raw.get("title")),
        "summary": _s(raw.get("summary")),
        "events": _list(raw.get("events"), 8),
        "characters": _list(raw.get("characters"), 8),
        "locations": _list(raw.get("locations"), 5),
        "mood": _s(raw.get("mood")),
    }


# =============================================================== 2. profile
def tag_stats(scenes: list[dict[str, Any]]) -> dict[str, float]:
    """Share of story scenes (0..1) with each tag >= 0.5, only tags that occur, highest first."""
    story = [sc for sc in scenes if "error" not in sc and not sc.get("graphic")]
    if not story:
        return {}
    counts = {t: sum(1 for sc in story if sc.get("tags", {}).get(t, 0.0) >= 0.5) for t in s07_scenes.TAGS}
    return {t: round(c / len(story), 2) for t, c in sorted(counts.items(), key=lambda kv: -kv[1]) if c}


def build_profile_prompt(chapters: list[dict[str, Any]], scenes: list[dict[str, Any]],
                         timeline: dict[str, Any]) -> str:
    duration = max((sc["end_ms"] for sc in scenes), default=0)
    lines = [f"VIDEO LENGTH: {_fmt(duration)} ({len(scenes)} scenes, {len(chapters)} chapters)", "", "CHAPTERS:"]
    for c in chapters:
        if "error" in c:
            lines.append(f"Chapter {c['chapter']} [{_fmt(c['start_ms'])}-{_fmt(c['end_ms'])}]: (summary failed)")
            continue
        lines.append(f"Chapter {c['chapter']} [{_fmt(c['start_ms'])}-{_fmt(c['end_ms'])}] {c['title']}: {c['summary']}")
        if c.get("characters"):
            lines.append(f"  Characters: {'; '.join(c['characters'])}")
    stats = tag_stats(scenes)
    lines += ["", "TAG STATISTICS (share of scenes):"]
    lines += [f"  {t}: {int(v * 100)}%" for t, v in stats.items()] or ["  (no tags)"]
    final = [sc for sc in scenes if "error" not in sc and not sc.get("graphic")][-3:]
    if final:
        lines += ["", "FINAL MOMENTS (last scenes, in order):"]
        lines += [f"  [{_fmt(sc['start_ms'])}] {sc.get('title', '')}: {sc.get('description', '')}" for sc in final]
    titles = [t["text"] for sc in scenes if sc.get("graphic") for t in timeline.get("on_screen_text", [])
              if sc["start_ms"] <= t["start_ms"] < sc["end_ms"]]
    if titles:
        lines += ["", "TEXT ON TITLE CARDS: " + " / ".join(titles[:10])]
    return "\n".join(lines)


def build_profile(llm, chapters: list[dict[str, Any]], scenes: list[dict[str, Any]],
                  timeline: dict[str, Any]) -> dict[str, Any]:
    stats = tag_stats(scenes)
    if all("error" in c for c in chapters):
        return {"error": "all chapter summaries failed", "tag_stats": stats}
    try:
        raw = llm.generate_json(PROFILE_SYSTEM, build_profile_prompt(chapters, scenes, timeline), PROFILE_SCHEMA)
    except OllamaUnreachable as exc:
        raise StageError(f"Lost connection to Ollama ({exc}). Start Ollama and re-run; finished chapters are kept.") from exc
    except Exception as exc:  # noqa: BLE001
        log.warning("profile: %s", exc)
        return {"error": str(exc), "tag_stats": stats}

    genres = []
    for g in raw.get("genres") or []:
        if isinstance(g, dict) and g.get("genre") in GENRES and _s(g.get("evidence")):
            try:
                conf = min(max(float(g.get("confidence") or 0.0), 0.0), 1.0)
            except (TypeError, ValueError):
                continue
            if g["genre"] not in {x["genre"] for x in genres}:
                genres.append({"genre": g["genre"], "confidence": round(conf, 2), "evidence": _s(g["evidence"])})
    characters = [{"name": _s(c.get("name")), "role": _s(c.get("role")), "description": _s(c.get("description"))}
                  for c in raw.get("characters") or [] if isinstance(c, dict) and _s(c.get("name"))][:10]
    return {
        "video_type": raw.get("video_type") if raw.get("video_type") in PROFILE_SCHEMA["properties"]["video_type"]["enum"] else "other",
        "genres": sorted(genres, key=lambda g: -g["confidence"]),
        "logline": _s(raw.get("logline")),
        "plot": _s(raw.get("plot")),
        "characters": characters,
        "themes": _list(raw.get("themes"), 5),
        "setting": _s(raw.get("setting")),
        "tone": _s(raw.get("tone")),
        "ending": _s(raw.get("ending")),
        "tag_stats": stats,                      # hard evidence counted by code, for genre questions
    }


# =============================================================== helpers
def _s(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def _list(v: Any, n: int) -> list[str]:
    return [_s(x) for x in (v or []) if isinstance(x, str) and _s(x)][:n]


def _fmt(ms: int) -> str:
    s = ms // 1000
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _save(paths, video_id: str, meta: dict[str, Any], chapters, profile, *, complete: bool) -> None:
    storage.write_json(paths.summaries, {
        "video_id": video_id,
        "model": settings.summary_llm_model,
        "prompt_version": PROMPT_VERSION,
        "complete": complete,
        "chapter_target_ms": settings.chapter_target_ms,
        "inputs": _inputs(meta),
        "chapters": chapters,
        "profile": profile,
    })


def _inputs(meta: dict[str, Any]) -> dict[str, Any]:
    return {"scenes": meta.get("stages", {}).get("scenes", {}).get("updated_at"),
            "model": settings.summary_llm_model, "prompt_version": PROMPT_VERSION,
            "chapter_target_ms": settings.chapter_target_ms}
