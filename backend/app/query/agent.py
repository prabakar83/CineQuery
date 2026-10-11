"""Question-answering agent: answer ANY question about an indexed video.

How it works (all local, qwen2.5:7b via Ollama):
    1. router.route() pre-runs tools for obvious questions (time ranges, genre, characters).
    2. Loop (max settings.ask_max_steps): the LLM sees the question + every tool result so far
       and replies with JSON: either {"action": <tool>, ...args} or {"action": "answer", "answer": ...}.
    3. The answer must cite timestamps from the tool results. Code checks every cited time
       against the evidence and reports times it could not verify.

Tools (app/query/tools/):
    search_moments(query)        meaning search over scenes, dialogue, on-screen text (stage 8)
    find_scenes_by_tag(tag)      "all romance scenes" (stage 7 tags + stage 8 scores)
    get_range(start, end)        everything in a time window
    search_dialogue(query)       exact / fuzzy quotes
    get_movie_summary()          genre, plot, characters, ending (stage 9)
    get_chapter_summaries()      chapter-by-chapter story (stage 9)
    get_characters()             who appears, roles, where
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Optional

from app.adapters.ollama_llm import OllamaLLM
from app.adapters.ollama_vision import OllamaError
from app.config import settings
from app.domain.timecode import TIME_RE, fmt, parse_time
from app.pipeline.stages import s07_scenes
from app.query import router
from app.query.tools import characters, range as range_tool, search, summary, tags, transcript

log = logging.getLogger(__name__)

TOOLS = ["search_moments", "find_scenes_by_tag", "get_range", "search_dialogue",
         "get_movie_summary", "get_chapter_summaries", "get_characters"]

STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": TOOLS + ["answer"]},
        "query": {"type": "string"},
        "tag": {"type": "string", "enum": s07_scenes.TAGS + [""]},
        "start": {"type": "string"},
        "end": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["thought", "action", "query", "tag", "start", "end", "answer"],
}
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"thought": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["thought", "answer"],
}

TAG_LINES = "\n".join(f"    {t}: {d}" for t, d in s07_scenes.TAG_DEFINITIONS.items())

SYSTEM = f"""You answer questions about ONE video using tools that search its index.
The index was built automatically from speech recognition, subtitles read from the picture and an image
model, so it can contain errors. Speech in a foreign/fictional language is shown as "[speech in another language]".

Tools (reply with JSON; fill only the fields the tool needs, leave the others ""):
- search_moments: query = what to look for, in your own words ("someone warns the crowd"). Finds scenes, lines, on-screen text by meaning.
- find_scenes_by_tag: tag = one of the tags below. Lists scenes with that tag, best first.
{TAG_LINES}
- get_range: start, end = times like "1:42:50" or "0:45". Everything in that time window.
- search_dialogue: query = words that were said or shown ("Lisan al-Gaib"). Finds exact quotes and when.
- get_movie_summary: genre, plot, main characters, themes, ending.
- get_chapter_summaries: the story chapter by chapter.
- get_characters: who appears, their roles, in which scenes.
- answer: answer = your final answer.

Rules:
- Use only facts from the TOOL RESULTS. Never invent names, events or times.
- Every moment you mention MUST carry its time in brackets, copied from the tool results, e.g.
  "[1:05] A man speaks to the robed crowd." or "[0:41-0:58] the warning scene". No time = do not mention it.
- If the results do not contain the answer, say so plainly and say what you did find.
- Do not repeat a tool call you already made. Answer as soon as you have enough.
- Answer in the same language as the question. Be concise: a few sentences or a short list.
- People: give names first; group unnamed people briefly ("several robed men"); say that names come from
  speech recognition and may be misspelled.
- "thought": at most one short sentence. When the answer needs a tool you have not called, call it; when the
  results already answer the question, answer immediately."""


# =============================================================== public
def ask(video_id: str, question: str, *, llm: Optional[OllamaLLM] = None,
        on_step: Optional[Callable[[dict[str, Any]], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_reset: Optional[Callable[[], None]] = None) -> dict[str, Any]:
    """{answer, sources[], steps[], unverified_times[], model, seconds}.

    on_step(step)  : called after every tool call (for progress display)
    on_token(text) : called with each new piece of the answer while the model writes it
    on_reset()     : streamed text so far must be discarded (that LLM call failed; a new answer follows)
    """
    streamed = {"any": False}

    def call_llm(prompt: str, schema: dict[str, Any], need_action: bool) -> dict[str, Any]:
        if not on_token:
            return llm.generate_json(SYSTEM, prompt, schema, temperature=0.1, max_tokens=settings.ask_answer_tokens)
        def emit(text: str) -> None:
            streamed["any"] = True
            on_token(text)
        try:
            return llm.generate_json_stream(SYSTEM, prompt, schema, answer_streamer(emit, need_action),
                                            temperature=0.1, max_tokens=settings.ask_answer_tokens)
        except OllamaError:
            if streamed["any"] and on_reset:
                on_reset()
            streamed["any"] = False
            raise

    started = time.monotonic()
    llm = llm or OllamaLLM(settings.ask_llm_model, settings.ollama_url, settings.ask_timeout_s,
                           0, settings.ask_llm_ctx)          # no retries: a timeout would double the wait
    try:
        llm.check_ready()
    except OllamaError as exc:
        raise RuntimeError(str(exc)) from exc

    steps: list[dict[str, Any]] = []
    done_calls: set[str] = set()

    def run(call: dict[str, Any], source: str) -> None:
        key = json.dumps(call, sort_keys=True)
        if key in done_calls:
            return
        done_calls.add(key)
        t0 = time.monotonic()
        result, text, times = run_tool(video_id, call)
        step = {"source": source, "call": call, "observation": text, "evidence_times": times,
                "items": tool_items(call["action"], result),
                "seconds": round(time.monotonic() - t0, 1)}
        steps.append(step)
        if on_step:
            on_step(step)

    for call in router.route(question):                     # 1. obvious tools first
        run(call, "router")

    answer = ""
    for _ in range(settings.ask_max_steps):                  # 2. agent loop
        try:
            reply = call_llm(build_prompt(question, steps), STEP_SCHEMA, need_action=True)
        except OllamaError as exc:                           # bad/slow step: go straight to the final answer
            log.warning("agent step failed: %s", exc)
            break
        call = to_call(reply)
        if call is None:                                     # action == answer
            answer = (reply.get("answer") or "").strip()
            if answer:
                break
            continue
        key = json.dumps(call, sort_keys=True)
        if key in done_calls:                                # repeating itself -> make it answer
            break
        run(call, "agent")

    if not answer:                                           # 3. forced final answer
        try:
            if streamed["any"] and on_reset:                 # a half-streamed step answer was dropped
                on_reset()
            reply = call_llm(build_prompt(question, steps, must_answer=True), ANSWER_SCHEMA, need_action=False)
        except OllamaError as exc:
            raise RuntimeError(f"The model could not produce an answer ({exc}). Try again, or use a faster model: "
                               f"$env:MOVIEQA_ASK_LLM = \"qwen2.5:3b\"") from exc
        answer = (reply.get("answer") or "").strip() or "I could not find an answer in this video's index."

    return {"answer": answer, "steps": steps, "unverified_times": unverified_times(answer, steps),
            "sources": pick_sources(answer, steps),
            "model": llm.model, "seconds": round(time.monotonic() - started, 1)}


# =============================================================== streaming
_ANSWER_KEY = re.compile(r'"answer"\s*:\s*"')
_ACTION = re.compile(r'"action"\s*:\s*"(\w+)"')
_ESCAPES = {"n": "\n", "t": "\t", "r": "", '"': '"', "\\": "\\", "/": "/", "b": "", "f": ""}


def answer_streamer(emit: Callable[[str], None], need_action: bool) -> Callable[[str], None]:
    """Return feed(piece): receives raw JSON text as the model writes it and calls emit(text) with
    the decoded characters of the "answer" field only (when action == "answer", if need_action)."""
    # show = None while we don't know yet whether action == "answer" (text is held back meanwhile)
    state = {"buf": "", "pos": None, "done": False, "show": None if need_action else True, "held": []}

    def decide() -> None:
        if state["show"] is None:
            m = _ACTION.search(state["buf"])
            if m:
                state["show"] = m.group(1) == "answer"
                if state["show"] and state["held"]:
                    emit("".join(state["held"]))
                state["held"] = []

    def feed(piece: str) -> None:
        state["buf"] += piece
        decide()
        if state["show"] is False:
            return                                      # a tool call, not an answer: nothing to show
        if state["done"]:
            return
        buf = state["buf"]
        if state["pos"] is None:
            m = _ANSWER_KEY.search(buf)
            if not m:
                return
            state["pos"] = m.end()
        i, out = state["pos"], []
        while i < len(buf):
            ch = buf[i]
            if ch == "\\":
                if i + 1 >= len(buf):
                    break                               # wait for the rest of the escape
                nxt = buf[i + 1]
                if nxt == "u":
                    if i + 6 > len(buf):
                        break
                    try:
                        out.append(chr(int(buf[i + 2:i + 6], 16)))
                    except ValueError:
                        pass
                    i += 6
                    continue
                out.append(_ESCAPES.get(nxt, nxt))
                i += 2
                continue
            if ch == '"':
                state["done"] = True
                break
            out.append(ch)
            i += 1
        state["pos"] = i
        if out:
            if state["show"]:
                emit("".join(out))
            else:
                state["held"].append("".join(out))

    return feed


# =============================================================== prompt
def build_prompt(question: str, steps: list[dict[str, Any]], must_answer: bool = False) -> str:
    lines = [f"QUESTION: {question}", ""]
    if steps:
        lines.append("TOOL RESULTS SO FAR:")
        for i, st in enumerate(steps, 1):
            args = {k: v for k, v in st["call"].items() if k != "action"}
            lines.append(f"[{i}] {st['call']['action']}({_args_text(args)}):")
            lines.append(st["observation"])
            lines.append("")
    else:
        lines.append("TOOL RESULTS SO FAR: (none yet)")
    if must_answer:
        lines.append("You must answer now from the results above (say so if they are not enough). "
                     "Put the [time] from the results in front of every moment you mention.")
    else:
        lines.append(f"Choose the next tool, or answer. You may call {settings.ask_max_steps} tools at most.")
    return "\n".join(lines)


def _args_text(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={fmt(v) if k.endswith('_ms') else v!r}" for k, v in args.items())


def to_call(reply: dict[str, Any]) -> Optional[dict[str, Any]]:
    """LLM JSON -> tool call dict, or None for 'answer'. Invalid tool args fall back sensibly."""
    action = reply.get("action")
    q = (reply.get("query") or "").strip()
    if action == "search_moments":
        return {"action": action, "query": q} if q else None
    if action == "search_dialogue":
        return {"action": action, "query": q} if q else None
    if action == "find_scenes_by_tag":
        tag = reply.get("tag") or ""
        return {"action": action, "tag": tag} if tag in s07_scenes.TAGS else ({"action": "search_moments", "query": q} if q else None)
    if action == "get_range":
        start, end = parse_time(reply.get("start")), parse_time(reply.get("end"))
        if start is None:
            return None
        return {"action": action, "start_ms": start, "end_ms": end if end and end > start else start + 30_000}
    if action in ("get_movie_summary", "get_chapter_summaries", "get_characters"):
        return {"action": action}
    return None


# =============================================================== tools
def run_tool(video_id: str, call: dict[str, Any]) -> tuple[Any, str, list[int]]:
    """Run one tool -> (raw result, text for the LLM, evidence times in ms)."""
    a = call["action"]
    try:
        if a == "search_moments":
            res = search.search_moments(video_id, call["query"], top_k=8)
            rows = [f"- {r['kind']} [{_span(r)}] (score {r['score']:.2f}) {r['text'][:300]}" for r in res]
            times = [t for r in res for t in (r["start_ms"], r["end_ms"])]
        elif a == "find_scenes_by_tag":
            res = tags.find_scenes_by_tag(video_id, call["tag"])
            rows = [f"- scene {r['scene']} [{_span(r)}] {r['title']} (score {r['score']:.2f}) evidence: {r['evidence']}"
                    for r in res] or [f"(no scenes tagged {call['tag']})"]
            times = [t for r in res for t in (r["start_ms"], r["end_ms"])]
        elif a == "get_range":
            res = range_tool.get_range(video_id, call["start_ms"], call["end_ms"])
            rows = [f"Window {fmt(res['start_ms'])}-{fmt(res['end_ms'])}"]
            rows += [f"- scene {s['scene']} [{_span(s)}] {s['title']}: {s['description']}" for s in res["scenes"]]
            rows += [f"- says [{fmt(d['start_ms'])}] {d['text']}" for d in res["dialogue"]]
            rows += [f"- on screen [{fmt(t['start_ms'])}] \"{t['text']}\"" for t in res["on_screen_text"]]
            if len(rows) == 1:
                rows.append("(nothing in this window — it may be outside the video)")
            times = [t for x in res["scenes"] + res["dialogue"] + res["on_screen_text"] for t in (x["start_ms"], x["end_ms"])]
            times += [res["start_ms"], res["end_ms"]]
        elif a == "search_dialogue":
            res = transcript.search_dialogue(video_id, call["query"])
            rows = [f"- {r['source']} [{fmt(r['start_ms'])}] \"{r['text']}\" (match {r['score']:.2f})" for r in res] \
                or ["(no matching line)"]
            times = [t for r in res for t in (r["start_ms"], r["end_ms"])]
        elif a == "get_movie_summary":
            res = summary.get_movie_summary(video_id)
            rows = [f"type: {res.get('video_type')}",
                    "genres: " + "; ".join(f"{g['genre']} {g['confidence']} ({g['evidence']})" for g in res.get("genres", [])),
                    f"logline: {res.get('logline')}", f"plot: {res.get('plot')}", f"ending: {res.get('ending')}",
                    f"setting: {res.get('setting')}; tone: {res.get('tone')}; themes: {', '.join(res.get('themes', []))}",
                    "characters: " + "; ".join(f"{c['name']} ({c['role']})" for c in res.get("characters", [])),
                    "tag share of scenes: " + ", ".join(f"{t} {int(v * 100)}%" for t, v in res.get("tag_stats", {}).items())]
            times = []
        elif a == "get_chapter_summaries":
            res = summary.get_chapter_summaries(video_id)
            rows = [f"- chapter {c['chapter']} [{_span(c)}] {c.get('title', '')}: {c.get('summary', '')}" for c in res]
            times = [t for c in res for t in (c["start_ms"], c["end_ms"])]
        elif a == "get_characters":
            res = characters.get_characters(video_id)
            rows = ["main characters (from the summary):"]
            rows += [f"- {c['name']} ({c['role']}): {c['description']}" for c in res["main"]] or ["- (none)"]
            rows.append("people listed per scene:")
            rows += [f"- {e['name']}: scenes {', '.join(map(str, e['scenes']))} [{fmt(e['first_ms'])}-{fmt(e['last_ms'])}]"
                     f"{', named in ' + str(e['mentions']) + ' lines' if e['mentions'] else ''}" for e in res["appearances"]]
            times = [t for e in res["appearances"] for t in (e["first_ms"], e["last_ms"])]
        else:
            return None, f"(unknown tool {a})", []
    except Exception as exc:  # noqa: BLE001 — a failing tool becomes an observation, not a crash
        log.warning("tool %s failed: %s", a, exc)
        return None, f"(tool error: {exc})", []
    text = "\n".join(rows)
    if len(text) > settings.ask_observation_chars:
        text = text[:settings.ask_observation_chars] + "\n(… cut)"
    return res, text, times


def _span(x: dict[str, Any]) -> str:
    return f"{fmt(x['start_ms'])}-{fmt(x['end_ms'])}"


# =============================================================== sources (built by code, not the LLM)
def tool_items(action: str, res: Any) -> list[dict[str, Any]]:
    """Timed items a tool returned: [{label, start_ms, end_ms}] (empty for summary-type tools)."""
    if not res:
        return []
    if action == "search_moments":
        return [{"label": f"{r['kind']}: {r['text'][:90]}", "start_ms": r["start_ms"], "end_ms": r["end_ms"]} for r in res]
    if action == "find_scenes_by_tag":
        return [{"label": f"scene {r['scene']}: {r['title']}", "start_ms": r["start_ms"], "end_ms": r["end_ms"]} for r in res]
    if action == "search_dialogue":
        return [{"label": f"{r['source']}: \"{r['text'][:90]}\"", "start_ms": r["start_ms"], "end_ms": r["end_ms"]} for r in res]
    if action == "get_range":
        return [{"label": f"scene {x['scene']}: {x['title']}", "start_ms": x["start_ms"], "end_ms": x["end_ms"]}
                for x in res["scenes"]]
    if action == "get_chapter_summaries":
        return [{"label": f"chapter {c['chapter']}: {c.get('title', '')}", "start_ms": c["start_ms"], "end_ms": c["end_ms"]}
                for c in res]
    return []


def pick_sources(answer: str, steps: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    """Moments that back the answer, with exact times (the UI turns these into player links).

    Items whose time the answer cites come first; if the answer cites none, the top items of the
    tools that returned timed results are used instead.
    """
    items = [it for st in steps for it in st.get("items", [])]
    seen, uniq = set(), []
    for it in items:
        key = (it["start_ms"], it["end_ms"], it["label"])
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    cited = [t for t in (parse_time(r) for r in TIME_RE.findall(answer)) if t is not None]
    hit = [it for it in uniq if any(it["start_ms"] - 2000 <= t <= it["end_ms"] + 2000 for t in cited)]
    chosen = hit or [it for st in steps for it in st.get("items", [])[:3]]
    out, keys = [], set()
    for it in chosen:
        key = (it["start_ms"], it["end_ms"], it["label"])
        if key not in keys:
            keys.add(key)
            out.append(it)
    return sorted(out[:limit], key=lambda it: it["start_ms"])


# =============================================================== grounding check
def unverified_times(answer: str, steps: list[dict[str, Any]], tolerance_ms: int = 5000) -> list[str]:
    """Times cited in the answer that are not within 5 s of any time in the tool results."""
    evidence = [t for st in steps for t in st["evidence_times"]]
    spans = [(st["call"]["start_ms"], st["call"]["end_ms"]) for st in steps if st["call"]["action"] == "get_range"]
    bad = []
    for raw in TIME_RE.findall(answer):
        t = parse_time(raw)
        if t is None:
            continue
        near = any(abs(t - e) <= tolerance_ms for e in evidence) or any(s <= t <= e for s, e in spans)
        if not near:
            bad.append(raw)
    return bad
