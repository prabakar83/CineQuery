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
from app.pipeline.stages import s01_normalize, s02_chunk, s03_transcribe

# Stages run in this order. New stages get appended here.
PIPELINE = [s01_normalize, s02_chunk, s03_transcribe]
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

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
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
