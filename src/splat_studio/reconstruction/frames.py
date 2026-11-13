"""Frame extraction from video with ffmpeg."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

TARGET_FRAMES = 180
MAX_LONG_SIDE = 1600


def sample_fps(duration_s: float, source_fps: float, target_frames: int = TARGET_FRAMES) -> float:
    """Frames per second to extract so a video yields about `target_frames` frames."""
    if duration_s <= 0 or source_fps <= 0:
        raise ValueError("duration and fps must be positive")
    return min(source_fps, target_frames / duration_s)


def scale_filter(max_long_side: int = MAX_LONG_SIDE) -> str:
    """ffmpeg scale filter that shrinks the long side to `max_long_side`, never upscales."""
    m = max_long_side
    return f"scale='if(gt(iw,ih),min({m},iw),-2)':'if(gt(iw,ih),-2,min({m},ih))'"


def ffmpeg_command(video: Path, out_dir: Path, fps: float) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vf", f"fps={fps:.4f},{scale_filter()}",
        "-q:v", "2", str(out_dir / "%05d.jpg"),
    ]  # fmt: skip


def probe(video: Path) -> tuple[float, float]:
    """Return (duration_s, fps) of a video via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate:format=duration", "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    info = json.loads(out)
    num, den = info["streams"][0]["avg_frame_rate"].split("/")
    return float(info["format"]["duration"]), float(num) / float(den or 1)


def extract_frames(video: Path, out_dir: Path, target_frames: int = TARGET_FRAMES) -> int:
    """Extract frames into `out_dir`; returns the number of frames written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration, fps = probe(video)
    subprocess.run(
        ffmpeg_command(video, out_dir, sample_fps(duration, fps, target_frames)), check=True
    )
    return len(list(out_dir.glob("*.jpg")))
