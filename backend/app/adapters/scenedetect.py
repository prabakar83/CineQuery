"""Shot (camera-cut) detection with PySceneDetect (plain functions).

A shot = continuous footage between two cuts. ContentDetector compares each
frame with the previous one (hue, saturation, brightness); a large jump = a cut.
"""
from __future__ import annotations

from pathlib import Path


def detect_shots(video: Path, start_ms: int, end_ms: int, *, threshold: float = 27.0,
                 min_shot_ms: int = 500) -> list[tuple[int, int]]:
    """Return [(shot_start_ms, shot_end_ms), ...] covering [start_ms, end_ms) of the video.

    The first and last shots are clipped to the requested range, so they may
    really begin earlier / end later than reported.
    """
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError as exc:
        raise RuntimeError(
            "PySceneDetect is not installed in this Python. Activate the project venv "
            "(source .venv/bin/activate) and run: pip install -r requirements.txt"
        ) from exc

    stream = open_video(str(video))
    fps = float(stream.frame_rate)
    min_frames = max(1, round(min_shot_ms * fps / 1000))       # detector wants frames, not ms

    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_frames))
    if start_ms > 0:
        stream.seek(start_ms / 1000)                            # seconds
    manager.detect_scenes(video=stream, end_time=end_ms / 1000)

    shots = [(_ms(a), _ms(b)) for a, b in manager.get_scene_list(start_in_scene=True)]
    if not shots:                                               # no frames decoded at all
        return []
    # Clip to the requested window (decoding can land a frame or two outside it).
    shots = [(max(a, start_ms), min(b, end_ms)) for a, b in shots]
    return [(a, b) for a, b in shots if b > a]


def _ms(timecode) -> int:
    """FrameTimecode -> ms. Works with PySceneDetect 0.6 (get_seconds) and 0.7+ (.seconds)."""
    seconds = getattr(timecode, "seconds", None)
    if seconds is None or callable(seconds):
        seconds = timecode.get_seconds()
    return int(round(float(seconds) * 1000))
