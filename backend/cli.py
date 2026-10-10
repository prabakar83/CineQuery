"""Command-line entry point (used before the API exists, and on Kaggle).

Run from the backend/ folder:

    python3 cli.py ingest "/path/to/movie.mp4"      # import + run all stages
    python3 cli.py normalize vid_8f3a2c [--force]   # re-run one stage
    python3 cli.py chunk vid_8f3a2c [--force]
    python3 cli.py list                             # show ingested videos
    python3 cli.py info vid_8f3a2c                  # print metadata.json
    python3 cli.py chunks vid_8f3a2c                # print the chunk table
    python3 cli.py transcribe vid_8f3a2c [--force]  # needs the .venv with faster-whisper
    python3 cli.py transcript vid_8f3a2c            # print the dialogue with timestamps
    python3 cli.py shots vid_8f3a2c [--force]       # detect cuts + save keyframes (needs .venv)
    python3 cli.py shot-list vid_8f3a2c             # print the shot table
    python3 cli.py caption vid_8f3a2c [--force]     # describe keyframes (Florence-2 on the GPU)
    python3 cli.py captions vid_8f3a2c [--full]     # print the frame descriptions
    python3 cli.py merge vid_8f3a2c [--force]       # build timeline.json (no AI, ~1 s)
    python3 cli.py timeline vid_8f3a2c [--from 1:40 --to 2:00]   # print the merged timeline
    python3 cli.py scenes vid_8f3a2c [--force]      # group shots into scenes + LLM descriptions (needs Ollama + qwen2.5:3b)
    python3 cli.py scene-list vid_8f3a2c            # one line per scene with top tags
    python3 cli.py scene vid_8f3a2c 3               # everything about scene 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.adapters import local_storage as storage
from app.adapters.ffmpeg import FFmpegError
from app.pipeline.base import StageError
from app.pipeline.stages import s01_normalize, s02_chunk, s03_transcribe, s04_shots, s05_caption, s06_merge, s07_scenes

# Stages run in this order. New stages get appended here.
PIPELINE = [s01_normalize, s02_chunk, s03_transcribe, s04_shots, s05_caption, s06_merge, s07_scenes]
STAGES_BY_NAME = {stage.NAME: stage for stage in PIPELINE}


# ---------------------------------------------------------------- progress bar
class ConsoleProgress:
    """Draws a one-line progress bar. (A class only because it remembers the last line drawn.)"""

    def __init__(self) -> None:
        self._last = ""

    def __call__(self, stage: str, fraction: float, message: str) -> None:
        pct = int(fraction * 100)
        bar = "#" * (pct // 4) + "-" * (25 - pct // 4)
        line = f"\r  [{stage:<10}] [{bar}] {pct:3d}%  {message:<30}"
        if line != self._last:
            sys.stdout.write(line)
            sys.stdout.flush()
            self._last = line
        if fraction >= 1.0:
            sys.stdout.write("\n")
            self._last = ""


# ---------------------------------------------------------------- helpers
def _fmt(ms: int) -> str:
    s = ms // 1000
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _require_video(video_id: str) -> None:
    if not storage.video_exists(video_id):
        raise StageError(f"Unknown video id {video_id!r}. Try `python3 cli.py list`.")


def run_pipeline(video_id: str, force: bool = False) -> None:
    progress = ConsoleProgress()
    for stage in PIPELINE:
        stage.run(video_id, force=force, progress=progress)


# ---------------------------------------------------------------- commands
def cmd_ingest(args: argparse.Namespace) -> None:
    video_id = args.id or storage.new_video_id()
    original = storage.import_original(Path(args.file), video_id)
    print(f"Video id: {video_id}")
    print(f"Stored  : {original}")
    run_pipeline(video_id, force=args.force)
    print(f"Ready   : {storage.video_paths(video_id).root}")


def cmd_stage(args: argparse.Namespace) -> None:
    """Run a single stage by name: `cli.py normalize <id>`, `cli.py chunk <id>`."""
    _require_video(args.video_id)
    STAGES_BY_NAME[args.command].run(args.video_id, force=args.force, progress=ConsoleProgress())


def cmd_list(args: argparse.Namespace) -> None:
    ids = storage.list_video_ids()
    if not ids:
        print("No videos yet. Run: python3 cli.py ingest <file>")
    for vid in ids:
        meta = storage.read_metadata(vid)
        stages = ", ".join(f"{k}={v.get('status')}" for k, v in meta.get("stages", {}).items()) or "-"
        print(f"{vid:<14} {meta.get('original_file', '?'):<16} {stages}")


def cmd_info(args: argparse.Namespace) -> None:
    _require_video(args.video_id)
    print(json.dumps(storage.read_metadata(args.video_id), indent=2))


def cmd_chunks(args: argparse.Namespace) -> None:
    _require_video(args.video_id)
    data = storage.read_chunks(args.video_id)
    if not data:
        print("No chunks yet. Run: python3 cli.py chunk", args.video_id)
        return
    print(f"{data['count']} chunk(s), {data['chunk_ms'] // 1000}s each, {data['overlap_ms'] // 1000}s overlap\n")
    print(f"{'id':<6} {'core range':<21} {'processed range':<21} stages")
    for c in data["chunks"]:
        core = f"{_fmt(c['core_start_ms'])}-{_fmt(c['core_end_ms'])}"
        padded = f"{_fmt(c['start_ms'])}-{_fmt(c['end_ms'])}"
        stages = ", ".join(f"{k}={v}" for k, v in c.get("stages", {}).items()) or "-"
        print(f"{c['id']:<6} {core:<21} {padded:<21} {stages}")


def cmd_transcript(args: argparse.Namespace) -> None:
    _require_video(args.video_id)
    segments = s03_transcribe.load_transcript(args.video_id)
    if not segments:
        print("No transcript yet. Run: python cli.py transcribe", args.video_id)
        return
    for seg in segments:
        print(f"[{_fmt(seg['start_ms'])} - {_fmt(seg['end_ms'])}] {seg['text']}")
    print(f"\n{len(segments)} segment(s)")


def cmd_shot_list(args: argparse.Namespace) -> None:
    _require_video(args.video_id)
    shots = s04_shots.load_shots(args.video_id)
    if not shots:
        print("No shots yet. Run: python cli.py shots", args.video_id)
        return
    print(f"{'#':>4}  {'start':<8} {'end':<8} {'length':>7}  keyframes")
    for i, s in enumerate(shots, 1):
        length = (s["end_ms"] - s["start_ms"]) / 1000
        frames = ", ".join(kf["file"].split("/")[-1] for kf in s["keyframes"])
        print(f"{i:>4}  {_fmt(s['start_ms']):<8} {_fmt(s['end_ms']):<8} {length:>6.1f}s  {frames}")
    total_kf = sum(len(s["keyframes"]) for s in shots)
    print(f"\n{len(shots)} shot(s), {total_kf} keyframe(s)")


def cmd_captions(args: argparse.Namespace) -> None:
    _require_video(args.video_id)
    frames = s05_caption.load_captions(args.video_id)
    if not frames:
        print("No captions yet. Run: python cli.py caption", args.video_id)
        return
    for f in frames:
        print(f"[{_fmt(f['ts_ms'])}] {f['file'].split('/')[-1]}")
        if "error" in f:
            print(f"    ERROR: {f['error']}")
            continue
        print(f"    {f['caption']}")
        if f.get("on_screen_text"):
            print(f"    TEXT: \"{f['on_screen_text']}\"")
        if args.full:
            for p in f.get("people", []):            # ollama engine only
                print(f"    PERSON: {p['description']} | {p['action']} | {p['expression']}")
            if f.get("setting"):                     # ollama engine only
                print(f"    SETTING: {f['setting']} ({f.get('time_of_day', '')}) | SHOT: {f.get('shot_type', '')} "
                      f"| MOOD: {f.get('mood', '')}")
            if f.get("objects"):
                print(f"    OBJECTS: {', '.join(f['objects'])}")
    with_text = sum(1 for f in frames if f.get("on_screen_text"))
    errors = sum(1 for f in frames if "error" in f)
    print(f"\n{len(frames)} frame(s), {with_text} with on-screen text, {errors} error(s)")


def _parse_time(value: str | None) -> int | None:
    """'1:42:50' / '1:42' / '95' -> milliseconds."""
    if not value:
        return None
    parts = [float(p) for p in value.split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return int(seconds * 1000)


def cmd_timeline(args: argparse.Namespace) -> None:
    _require_video(args.video_id)
    data = s06_merge.load_timeline(args.video_id)
    if not data:
        print("No timeline yet. Run: python cli.py merge", args.video_id)
        return
    start, end = _parse_time(args.start), _parse_time(args.end)
    printed: set[str] = set()   # a line spanning several shots is printed once (in its first shot)
    for s in data["shots"]:
        if (start is not None and s["end_ms"] <= start) or (end is not None and s["start_ms"] >= end):
            continue
        print(f"\n=== shot {s['shot']}  {_fmt(s['start_ms'])} - {_fmt(s['end_ms'])}")
        for v in s["visual"]:
            tag = " [graphic]" if v.get("graphic") else ""
            desc = v["description"] if args.full else v["description"].split(". ")[0]
            print(f"  VISUAL{tag}: {desc}")
            if args.full and v.get("objects"):
                print(f"  OBJECTS: {', '.join(v['objects'])}")
        for t in s["on_screen_text"]:
            print(f"  TEXT   [{_fmt(t['start_ms'])}]: \"{t['text']}\"")
        for d in s["dialogue"]:
            if d["id"] in printed:
                continue
            printed.add(d["id"])
            notes = []
            if not d["reliable"]:
                notes.append(f"unreliable: {d.get('reason') or 'low_confidence'}")
            if d.get("lang") and d.get("lang_prob") is not None and args.full:
                notes.append(f"heard as {d['lang']}, p(main)={d['lang_prob']}")
            if d.get("subtitled"):
                notes.append("subtitled")
            mark = f"  ({'; '.join(notes)})" if notes else ""
            print(f"  SAYS   [{_fmt(d['start_ms'])}]: {d['text']}{mark}")
    st = data["stats"]
    print(f"\n{st['shots']} shots, {st['dialogue_lines']} dialogue lines ({st['dialogue_unreliable']} unreliable, "
          f"{st.get('dialogue_other_language', 0)} other-language), "
          f"{st['on_screen_texts']} on-screen texts, {st['graphic_frames']} graphic frames")


def cmd_scene_list(args: argparse.Namespace) -> None:
    _require_video(args.video_id)
    scenes = s07_scenes.load_scenes(args.video_id)
    if not scenes:
        print("No scenes yet. Run: python cli.py scenes", args.video_id)
        return
    for sc in scenes:
        if "error" in sc:
            print(f"{sc['scene']:>3}  {_fmt(sc['start_ms'])}-{_fmt(sc['end_ms'])}  ERROR: {sc['error']}")
            continue
        top = sorted(((v, k) for k, v in sc["tags"].items() if v >= 0.5), reverse=True)[:4]
        tags = ", ".join(f"{k} {v:.1f}" for v, k in top) or "-"
        cut = f"{sc['boundary_score']:.2f}" if sc.get("boundary_score") is not None else "  - "
        print(f"{sc['scene']:>3}  {_fmt(sc['start_ms'])}-{_fmt(sc['end_ms'])}  cut={cut}  {sc['title']:<32.32}  "
              f"[{sc['mood']}]  {tags}")
    print(f"\n{len(scenes)} scene(s)")


def cmd_scene(args: argparse.Namespace) -> None:
    _require_video(args.video_id)
    scenes = {sc["scene"]: sc for sc in s07_scenes.load_scenes(args.video_id)}
    sc = scenes.get(args.number)
    if not sc:
        print(f"No scene {args.number}. Scenes: 1-{len(scenes)}")
        return
    print(f"Scene {sc['scene']}: {sc.get('title', '')}   {_fmt(sc['start_ms'])} - {_fmt(sc['end_ms'])}  "
          f"(shots {sc['shots'][0]}-{sc['shots'][1]})")
    if "error" in sc:
        print("ERROR:", sc["error"])
        return
    print(f"\n{sc['description']}\n")
    print("Characters :", "; ".join(sc["characters"]) or "-")
    print("Location   :", sc["location"] or "-", f"({sc['time_of_day']})")
    print("Mood       :", sc["mood"] or "-")
    for e in sc["events"]:
        print("  -", e)
    print("Tags       :", ", ".join(f"{k} {v:.1f}" for k, v in sorted(sc["tags"].items(), key=lambda kv: -kv[1]) if v > 0) or "-")
    for tag, why in sc.get("tag_evidence", {}).items():
        print(f"    {tag}: {why}")
    tl = s06_merge.load_timeline(args.video_id)
    by_id = {d["id"]: d for d in tl.get("dialogue", [])} | {t["id"]: t for t in tl.get("on_screen_text", [])}
    ev = sc["evidence"]
    if ev["dialogue_ids"] or ev["text_ids"]:
        print("\nEvidence:")
        for i in ev["dialogue_ids"]:
            print(f"  SAYS [{_fmt(by_id[i]['start_ms'])}] {by_id[i]['text']}")
        for i in ev["text_ids"]:
            print(f"  TEXT [{_fmt(by_id[i]['start_ms'])}] {by_id[i]['text']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cli.py", description="Movie QA backend CLI")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="import a video file and run all stages")
    p.add_argument("file")
    p.add_argument("--id", help="use this video id instead of a random one")
    p.add_argument("--force", action="store_true", help="re-run stages even if already done")
    p.set_defaults(func=cmd_ingest)

    for stage in PIPELINE:  # one sub-command per stage: normalize, chunk, ...
        p = sub.add_parser(stage.NAME, help=f"run only the {stage.NAME} stage")
        p.add_argument("video_id")
        p.add_argument("--force", action="store_true")
        p.set_defaults(func=cmd_stage)

    p = sub.add_parser("list", help="list ingested videos")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("info", help="print a video's metadata.json")
    p.add_argument("video_id")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("chunks", help="print a video's chunk table")
    p.add_argument("video_id")
    p.set_defaults(func=cmd_chunks)

    p = sub.add_parser("transcript", help="print the dialogue with timestamps")
    p.add_argument("video_id")
    p.set_defaults(func=cmd_transcript)

    p = sub.add_parser("shot-list", help="print the shot table")
    p.add_argument("video_id")
    p.set_defaults(func=cmd_shot_list)

    p = sub.add_parser("scene-list", help="list scenes with titles and top tags")
    p.add_argument("video_id")
    p.set_defaults(func=cmd_scene_list)

    p = sub.add_parser("scene", help="show one scene in full")
    p.add_argument("video_id")
    p.add_argument("number", type=int)
    p.set_defaults(func=cmd_scene)

    p = sub.add_parser("timeline", help="print the merged timeline")
    p.add_argument("video_id")
    p.add_argument("--from", dest="start", help="start time, e.g. 1:40")
    p.add_argument("--to", dest="end", help="end time, e.g. 2:00")
    p.add_argument("--full", action="store_true", help="full descriptions and objects")
    p.set_defaults(func=cmd_timeline)

    p = sub.add_parser("captions", help="print the keyframe descriptions")
    p.add_argument("video_id")
    p.add_argument("--full", action="store_true", help="also show people, setting, objects, mood")
    p.set_defaults(func=cmd_captions)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence chatty libraries (model downloads log every HTTP request).
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "faster_whisper", "pyscenedetect", "urllib3", "transformers"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    try:
        args.func(args)
    except (StageError, FFmpegError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
