"""Command-line entry point (used before the API exists, and on Kaggle).

Run from the backend/ folder:

    python cli.py ingest "/path/to/movie.mkv"      # copy into data/ and run the pipeline
    python cli.py normalize vid_8f3a2c [--force]   # re-run one stage on an existing video
    python cli.py list                             # show ingested videos
    python cli.py info vid_8f3a2c                  # print metadata.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.adapters.ffmpeg import FFmpegError
from app.adapters.local_storage import VideoStore
from app.pipeline.base import StageError
from app.pipeline.stages.s01_normalize import NormalizeStage

# Stages run in this order. New stages get appended here.
PIPELINE = [NormalizeStage]


# ---------------------------------------------------------------- progress bar
class ConsoleProgress:
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


# ---------------------------------------------------------------- commands
def run_pipeline(video_id: str, store: VideoStore, force: bool = False) -> None:
    progress = ConsoleProgress()
    for stage_cls in PIPELINE:
        stage = stage_cls(store=store, progress=progress)
        stage.run(video_id, force=force)


def cmd_ingest(args: argparse.Namespace, store: VideoStore) -> None:
    video_id = args.id or store.new_video_id()
    original = store.import_original(Path(args.file), video_id)
    print(f"Video id: {video_id}")
    print(f"Stored  : {original}")
    run_pipeline(video_id, store, force=args.force)
    print(f"Ready   : {store.paths(video_id).root}")


def cmd_normalize(args: argparse.Namespace, store: VideoStore) -> None:
    if not store.exists(args.video_id):
        raise StageError(f"Unknown video id {args.video_id!r}. Try `python cli.py list`.")
    NormalizeStage(store=store, progress=ConsoleProgress()).run(args.video_id, force=args.force)


def cmd_list(args: argparse.Namespace, store: VideoStore) -> None:
    ids = store.list_ids()
    if not ids:
        print("No videos yet. Run: python cli.py ingest <file>")
    for vid in ids:
        meta = store.read_metadata(vid)
        stages = ", ".join(f"{k}={v.get('status')}" for k, v in meta.get("stages", {}).items()) or "-"
        print(f"{vid:<14} {meta.get('original_file', '?'):<20} {stages}")


def cmd_info(args: argparse.Namespace, store: VideoStore) -> None:
    print(json.dumps(store.read_metadata(args.video_id), indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cli.py", description="Movie QA backend CLI")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="import a video file and process it")
    p.add_argument("file")
    p.add_argument("--id", help="use this video id instead of a random one")
    p.add_argument("--force", action="store_true", help="re-run stages even if already done")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("normalize", help="run only the normalize stage")
    p.add_argument("video_id")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("list", help="list ingested videos")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("info", help="print a video's metadata.json")
    p.add_argument("video_id")
    p.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        args.func(args, VideoStore())
    except (StageError, FFmpegError, FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
