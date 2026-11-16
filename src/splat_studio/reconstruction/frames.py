"""Training images: frames from a video (ffmpeg) or a photo set (Pillow)."""

from __future__ import annotations

import json
import shutil
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


def ffmpeg_command(
    video: Path, out_dir: Path, fps: float, max_long_side: int = MAX_LONG_SIDE
) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vf", f"fps={fps:.4f},{scale_filter(max_long_side)}",
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


def extract_frames(
    video: Path,
    out_dir: Path,
    target_frames: int = TARGET_FRAMES,
    max_long_side: int = MAX_LONG_SIDE,
) -> int:
    """Extract frames into `out_dir`; returns the number of frames written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration, fps = probe(video)
    fps_out = sample_fps(duration, fps, target_frames)
    subprocess.run(ffmpeg_command(video, out_dir, fps_out, max_long_side), check=True)
    return len(list(out_dir.glob("*.jpg")))


def copy_photos(photos: list[Path], out_dir: Path, max_long_side: int = MAX_LONG_SIDE) -> int:
    """Copy photos into `out_dir` as 00000.jpg, 00001.png, ...; returns the number copied.

    Photos whose long side is above `max_long_side` are scaled down; the others are copied
    as they are. Pixels keep their stored orientation (EXIF rotation is not applied, as
    COLMAP and gsplat read them that way) and JPEG EXIF is kept, as COLMAP reads the focal
    length from it.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(photos):
        ext = src.suffix.lower()
        dst = out_dir / f"{i:05d}{ext}"
        with Image.open(src) as im:
            long_side = max(im.size)
            if long_side <= max_long_side:
                shutil.copy(src, dst)
                continue
            scale = max_long_side / long_side
            size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
            small = im.resize(size, Image.Resampling.LANCZOS)
            if ext == ".png":
                small.save(dst)
            else:
                small.save(dst, quality=95, exif=im.info.get("exif", b""))
    return len(photos)
