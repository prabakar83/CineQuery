"""Stage 6 — Merge everything onto one movie timeline (pure Python, no AI).

Input : transcripts/*.json (stage 3), shots/*.json (stage 4), captions/*.json (stage 5),
        subtitles.srt if the file had embedded subtitles (stage 1)
Output: data/videos/{id}/timeline.json
        metadata.json -> timeline summary, stages.merge

This stage does NOT rewrite or interpret anything. It lines the sources up on the
same clock (movie milliseconds), using shots as containers:

  1. Shots     chunk-edge shots are joined back together; overlaps trimmed
  2. Dialogue  Whisper lines (overlap duplicates removed), each marked reliable or not:
               other_language (per-line language check, e.g. Fremen in Dune) or
               low_confidence; "subtitled" if on-screen text was showing at the time;
               embedded subtitles (.srt) are added as reliable dialogue
  3. On-screen text  the same subtitle read on neighbouring frames becomes ONE entry
               with a start and end time
  4. Visuals   frame descriptions with boilerplate sentences removed; title/credit
               graphics flagged
  5. Assign    every item goes into each shot it overlaps

Turning a group of shots into one meaningful scene description is stage 7's job.
"""
from __future__ import annotations

import difflib
import logging
import re
from typing import Any, Optional

from app.adapters import local_storage as storage
from app.config import settings
from app.pipeline.base import ProgressCallback, no_progress, require_stage_done, track_stage
from app.pipeline.stages import s03_transcribe, s04_shots, s05_caption

log = logging.getLogger(__name__)

NAME = "merge"
UPSTREAM = ("transcribe", "shots", "caption")


def is_done(video_id: str) -> bool:
    """Done only if finished AND no upstream stage has been re-run since (then merge again)."""
    meta = storage.read_metadata(video_id)
    if meta.get("stages", {}).get(NAME, {}).get("status") != "done":
        return False
    return meta.get("timeline", {}).get("inputs") == _upstream_versions(meta)


def run(video_id: str, *, force: bool = False, progress: ProgressCallback = no_progress) -> None:
    if not force and is_done(video_id):
        log.info("[%s] %s already done, skipping", video_id, NAME)
        progress(NAME, 1.0, "already done")
        return

    for stage in UPSTREAM:
        require_stage_done(video_id, stage)
    paths = storage.video_paths(video_id)
    meta = storage.read_metadata(video_id)
    duration_ms = meta.get("media", {}).get("duration_ms", 0)

    with track_stage(video_id, NAME):
        progress(NAME, 0.1, "loading stage outputs")
        shots = merge_shots(s04_shots.load_shots(video_id), duration_ms)
        dialogue = build_dialogue(s03_transcribe.load_transcript(video_id))
        if paths.subtitles.is_file():
            dialogue = sorted(dialogue + parse_srt(paths.subtitles.read_text(encoding="utf-8", errors="replace")),
                              key=lambda d: d["start_ms"])
        _number(dialogue, "d")

        progress(NAME, 0.5, "collapsing on-screen text")
        frames = s05_caption.load_captions(video_id)
        on_screen = build_on_screen_text(frames, shots)
        _number(on_screen, "t")
        visuals = [build_visual(f) for f in frames if "error" not in f]

        for d in dialogue:   # spoken while text was on screen = probably subtitled speech
            d["subtitled"] = any(t["start_ms"] < d["end_ms"] and d["start_ms"] < t["end_ms"] for t in on_screen)
            # Subtitled AND only moderately English-sounding -> foreign speech (Dune: "Hagaari,
            # imashneri yakalaha..." scored p(en)=0.67 but was subtitled). Clearly English
            # subtitled lines (p >= 0.75, e.g. "This is my father's ducal signet" 0.96) stay reliable.
            lp = d.get("lang_prob")
            if d["reliable"] and d["subtitled"] and lp is not None and lp < settings.dialogue_subtitled_min_lang_prob:
                d["reliable"], d["reason"] = False, "other_language"

        progress(NAME, 0.8, "assigning to shots")
        timeline_shots = assign_to_shots(shots, dialogue, on_screen, visuals)

        stats = {
            "shots": len(timeline_shots),
            "dialogue_lines": len(dialogue),
            "dialogue_unreliable": sum(1 for d in dialogue if not d["reliable"]),
            "dialogue_other_language": sum(1 for d in dialogue if d["reason"] == "other_language"),
            "dialogue_subtitled": sum(1 for d in dialogue if d["subtitled"]),
            "on_screen_texts": len(on_screen),
            "visual_frames": len(visuals),
            "graphic_frames": sum(1 for v in visuals if v["graphic"]),
            "shots_without_dialogue": sum(1 for s in timeline_shots if not s["dialogue"]),
        }
        storage.write_json(paths.timeline, {
            "video_id": video_id,
            "duration_ms": duration_ms,
            "language": meta.get("language"),
            "stats": stats,
            "dialogue": dialogue,
            "on_screen_text": on_screen,
            "shots": timeline_shots,
        })
        storage.update_metadata(video_id, timeline={**stats, "inputs": _upstream_versions(storage.read_metadata(video_id))})
        log.info("[%s] timeline: %d shots, %d dialogue lines (%d unreliable), %d on-screen texts",
                 video_id, stats["shots"], stats["dialogue_lines"], stats["dialogue_unreliable"],
                 stats["on_screen_texts"])

    progress(NAME, 1.0, "done")


def load_timeline(video_id: str) -> dict[str, Any]:
    return storage.read_json(storage.video_paths(video_id).timeline)


# =============================================================== 1. shots
def merge_shots(shots: list[dict[str, Any]], duration_ms: int) -> list[dict[str, Any]]:
    """Join shots that a chunk edge cut in two, and remove overlaps so shots tile the movie."""
    out: list[dict[str, Any]] = []
    for s in sorted(shots, key=lambda s: s["start_ms"]):
        s = {**s, "keyframes": list(s.get("keyframes", []))}
        if out:
            prev = out[-1]
            # A shot clipped at the end of one chunk + a shot clipped at the start of the next
            # = the same real shot seen from both sides of the boundary.
            if prev.get("clipped_end") and s.get("clipped_start") and s["start_ms"] <= prev["end_ms"]:
                prev["end_ms"] = max(prev["end_ms"], s["end_ms"])
                prev["clipped_end"] = s.get("clipped_end", False)
                seen = {k["ts_ms"] for k in prev["keyframes"]}
                prev["keyframes"] += [k for k in s["keyframes"] if k["ts_ms"] not in seen]
                continue
            if s["start_ms"] < prev["end_ms"]:          # overlap from padded ranges: trim
                s["start_ms"] = prev["end_ms"]
                if s["end_ms"] <= s["start_ms"]:
                    continue
        out.append(s)

    result = []
    for i, s in enumerate(out, 1):
        result.append({
            "shot": i,
            "start_ms": s["start_ms"],
            "end_ms": min(s["end_ms"], duration_ms) if duration_ms else s["end_ms"],
            "keyframes": sorted(k["ts_ms"] for k in s["keyframes"]),
        })
    return result


# =============================================================== 2. dialogue
def build_dialogue(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for seg in segments:
        words = seg.get("words") or []
        word_prob = round(sum(w["prob"] for w in words) / len(words), 3) if words else None
        reason = unreliable_reason(seg.get("lang_prob"), seg.get("avg_logprob"), word_prob)
        out.append({
            "start_ms": seg["start_ms"],
            "end_ms": seg["end_ms"],
            "text": seg["text"],
            "source": "asr",
            "reliable": reason is None,
            "reason": reason,                 # why it's unreliable: other_language / low_confidence
            "lang": seg.get("lang"),
            "lang_prob": seg.get("lang_prob"),
            "avg_logprob": seg.get("avg_logprob"),
            "word_prob": word_prob,
        })
    return out


def unreliable_reason(lang_prob: Optional[float], avg_logprob: Optional[float],
                      word_prob: Optional[float]) -> Optional[str]:
    """None if the line looks like real dialogue, else why not.

    Main signal: the per-line language check from stage 3. Speech in another (or a
    made-up) language gets forced into English-looking words by Whisper, often WITH
    decent confidence, so confidence alone can't catch it (tested on the Dune clip).
    Backup: very low Whisper confidence (noise, music, mumbling).
    """
    if lang_prob is not None and lang_prob < settings.dialogue_min_lang_prob:
        return "other_language"
    if avg_logprob is not None and avg_logprob < settings.dialogue_min_logprob:
        return "low_confidence"
    if word_prob is not None and word_prob < settings.dialogue_min_word_prob:
        return "low_confidence"
    return None


def is_reliable(avg_logprob: Optional[float], word_prob: Optional[float],
                lang_prob: Optional[float] = None) -> bool:
    return unreliable_reason(lang_prob, avg_logprob, word_prob) is None


_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
_TAGS = re.compile(r"<[^>]+>|\{[^}]+\}")


def parse_srt(text: str) -> list[dict[str, Any]]:
    """Embedded subtitles (from stage 1) = exact dialogue, so always reliable."""
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        for i, line in enumerate(lines):
            m = _SRT_TIME.search(line)
            if not m:
                continue
            h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
            body = _TAGS.sub("", " ".join(lines[i + 1:])).strip()
            if body:
                out.append({
                    "start_ms": ((h1 * 60 + m1) * 60 + s1) * 1000 + ms1,
                    "end_ms": ((h2 * 60 + m2) * 60 + s2) * 1000 + ms2,
                    "text": body, "source": "srt", "reliable": True, "reason": None,
                    "lang": None, "lang_prob": None, "avg_logprob": None, "word_prob": None,
                })
            break
    return out


# =============================================================== 3. on-screen text
MAX_GAP_MS = 6000        # same subtitle seen again within 6 s (a frame in between may have missed it)
SUBTITLE_TAIL_MS = 4000  # assume a subtitle stays up to 4 s after the last frame that showed it


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def _same_text(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= settings.ocr_same_text_ratio


def build_on_screen_text(frames: list[dict[str, Any]], shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the same text read on consecutive frames into one entry with a time span.

    Same text = similar enough (OCR varies slightly between frames) or one text contained
    in the other (OCR sometimes reads only part of a line), seen within MAX_GAP_MS or in the same shot.
    The entry ends where the shot of its last frame ends (subtitle duration is unknown).
    """
    entries: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    for f in sorted(frames, key=lambda f: f["ts_ms"]):
        text = (f.get("on_screen_text") or "").strip()
        if not text or "error" in f:
            continue
        close = current and (f["ts_ms"] - current["last_ts"] <= MAX_GAP_MS
                             or _shot_end(shots, f["ts_ms"]) == _shot_end(shots, current["last_ts"]))  # same shot
        if close and _same_text(current["text"], text):
            current["frames"].append(f["ts_ms"])
            current["variants"].append(text)
            current["last_ts"] = f["ts_ms"]
            if len(_norm(text)) > len(_norm(current["text"])):
                current["text"] = text            # compare future frames against the fullest reading
            continue
        current = {"text": text, "frames": [f["ts_ms"]], "variants": [text], "first_ts": f["ts_ms"], "last_ts": f["ts_ms"]}
        entries.append(current)

    out = []
    for i, e in enumerate(entries):
        best = max(e["variants"], key=lambda v: (len(_norm(v)), e["variants"].count(v)))   # fullest reading
        # End: the shot of the last frame ends, but never more than SUBTITLE_TAIL_MS after the
        # last sighting and never past the next on-screen text.
        end = min(_shot_end(shots, e["last_ts"]), e["last_ts"] + SUBTITLE_TAIL_MS)
        if i + 1 < len(entries):
            end = min(end, entries[i + 1]["first_ts"])
        out.append({
            "start_ms": e["first_ts"],
            "end_ms": max(end, e["last_ts"] + 1),
            "text": best,
            "frames": e["frames"],
        })
    return out


def _shot_end(shots: list[dict[str, Any]], ts: int) -> int:
    for s in shots:
        if s["start_ms"] <= ts < s["end_ms"]:
            return s["end_ms"]
    return ts + 2000


# =============================================================== 4. visuals
_BOILERPLATE = [
    re.compile(r"^the (image|picture|photo|frame) (is|shows|appears to be) (a |an )?(still|scene|screenshot|frame|close-up shot)? ?(from|of) (a |an )?(movie|film|tv show|television show|music video|video)( or (a )?(tv|television) show)?[.,]?$", re.I),
    re.compile(r"^(the )?overall,? (the )?(mood|atmosphere|tone|effect)\b.*", re.I),
    re.compile(r"^the (text|words|caption|title) (on|in) the (image|picture|frame) (reads?|says)\b.*", re.I),
    re.compile(r"^.*\b(is|are) written in white text\b.*", re.I),
]
_GRAPHIC = re.compile(r"\b(abstract design|digital art|digital illustration|minimalist aesthetic|geometric pattern|logo)\b", re.I)


def clean_description(text: str) -> str:
    """Remove sentences that carry no information about THIS frame."""
    text = re.sub(r"\s+", " ", text or "").strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if s and not any(p.match(s.strip()) for p in _BOILERPLATE)]
    cleaned = " ".join(kept).strip()
    # "It shows a man ..." / "The image shows a man ..." -> "A man ..."
    cleaned = re.sub(r"^(It|The (image|picture|photo|frame)) (shows|depicts|is) ", "", cleaned, flags=re.I)
    return (cleaned[:1].upper() + cleaned[1:]) if cleaned else ""


def build_visual(frame: dict[str, Any]) -> dict[str, Any]:
    raw = frame.get("caption", "")
    return {
        "ts_ms": frame["ts_ms"],
        "file": frame["file"],
        "description": clean_description(raw),
        "objects": frame.get("objects", []),
        # Title cards / credits / logos: not story content
        "graphic": bool(_GRAPHIC.search(raw)),
        # ollama engine extras, when present
        **{k: frame[k] for k in ("people", "setting", "mood", "shot_type") if frame.get(k)},
    }


# =============================================================== 5. assign
def assign_to_shots(shots, dialogue, on_screen, visuals) -> list[dict[str, Any]]:
    def overlaps(a_start, a_end, b_start, b_end) -> bool:
        return a_start < b_end and b_start < a_end

    out = []
    for s in shots:
        s_start, s_end = s["start_ms"], s["end_ms"]
        out.append({
            "shot": s["shot"],
            "start_ms": s_start,
            "end_ms": s_end,
            "dialogue": [d for d in dialogue if overlaps(d["start_ms"], d["end_ms"], s_start, s_end)],
            "on_screen_text": [t for t in on_screen if overlaps(t["start_ms"], t["end_ms"], s_start, s_end)],
            "visual": [v for v in visuals if s_start <= v["ts_ms"] < s_end],
        })

    # Safety net: an item that falls in a gap between shots goes to the nearest shot
    if out:
        placed_d = {d["id"] for s in out for d in s["dialogue"]}
        for d in dialogue:
            if d["id"] not in placed_d:
                _nearest(out, (d["start_ms"] + d["end_ms"]) // 2)["dialogue"].append(d)
        placed_v = {v["ts_ms"] for s in out for v in s["visual"]}
        for v in visuals:
            if v["ts_ms"] not in placed_v:
                _nearest(out, v["ts_ms"])["visual"].append(v)
    return out


def _nearest(shots: list[dict[str, Any]], ts: int) -> dict[str, Any]:
    return min(shots, key=lambda s: 0 if s["start_ms"] <= ts < s["end_ms"] else min(abs(ts - s["start_ms"]), abs(ts - s["end_ms"])))


# =============================================================== helpers
def _number(items: list[dict[str, Any]], prefix: str) -> None:
    for i, item in enumerate(items, 1):
        item["id"] = f"{prefix}{i:04d}"


def _upstream_versions(meta: dict[str, Any]) -> dict[str, Optional[str]]:
    stages = meta.get("stages", {})
    return {s: stages.get(s, {}).get("updated_at") for s in UPSTREAM}
