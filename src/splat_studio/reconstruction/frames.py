"""Training images: the sharpest frames from a video (ffmpeg) or a photo set (Pillow)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

TARGET_FRAMES = 180
MAX_LONG_SIDE = 1600
# Frames are taken at this many times the needed rate, and the sharpest of each group is
# kept. Scoring runs on small grayscale copies; the aspect ratio is not kept, as scores
# are only compared within one video.
CANDIDATES_PER_FRAME = 3
SCORE_SIZE = 480


def sample_fps(duration_s: float, source_fps: float, target_frames: int = TARGET_FRAMES) -> float:
    """Frames per second to extract so a video yields about `target_frames` frames."""
    if duration_s <= 0 or source_fps <= 0:
        raise ValueError("duration and fps must be positive")
    return min(source_fps, target_frames / duration_s)


def scale_filter(max_long_side: int = MAX_LONG_SIDE) -> str:
    """ffmpeg scale filter that shrinks the long side to `max_long_side`, never upscales."""
    m = max_long_side
    return f"scale='if(gt(iw,ih),min({m},iw),-2)':'if(gt(iw,ih),-2,min({m},ih))'"


def candidates_command(
    video: Path, out_dir: Path, fps: float, max_long_side: int = MAX_LONG_SIDE
) -> list[str]:
    """ffmpeg command that writes frames at `fps` as JPGs into `out_dir` and, for scoring,
    the same frames as SCORE_SIZE x SCORE_SIZE grayscale bytes on stdout."""
    s = SCORE_SIZE
    graph = (
        f"[0:v]fps={fps:.4f},split=2[a][b];[a]{scale_filter(max_long_side)}[full];"
        f"[b]scale={s}:{s}:flags=area,format=gray[small]"
    )
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-filter_complex", graph,
        "-map", "[full]", "-q:v", "2", str(out_dir / "%05d.jpg"),
        "-map", "[small]", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]  # fmt: skip


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: high for a sharp image, low for a blurred one."""
    g = gray.astype(np.float32)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def pick_sharpest(scores: list[float], count: int) -> list[int]:
    """Indices of the sharpest frame in each of `count` equal windows, in time order.

    Taking one frame per window keeps the frames spread over the whole video, which the
    sequential matcher needs, while skipping frames blurred by fast motion.
    """
    n = len(scores)
    if count >= n:
        return list(range(n))
    windows = np.array_split(np.arange(n), max(1, count))
    return [int(w[np.argmax([scores[i] for i in w])]) for w in windows]


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
    """Extract about `target_frames` sharp frames into `out_dir`; returns the number written.

    Takes CANDIDATES_PER_FRAME times as many frames (fewer if the video's own frame rate is
    lower), then keeps the sharpest of each group, numbered 00000.jpg, 00001.jpg, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    duration, fps = probe(video)
    fps_out = sample_fps(duration, fps, target_frames)
    fps_cand = min(fps, fps_out * CANDIDATES_PER_FRAME)
    frame_bytes = SCORE_SIZE * SCORE_SIZE
    with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
        cand_dir = Path(tmp)
        cmd = candidates_command(video, cand_dir, fps_cand, max_long_side)
        scores: list[float] = []
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
            assert proc.stdout is not None
            while len(buf := proc.stdout.read(frame_bytes)) == frame_bytes:
                small = np.frombuffer(buf, np.uint8).reshape(SCORE_SIZE, SCORE_SIZE)
                scores.append(sharpness(small))
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        candidates = sorted(cand_dir.glob("*.jpg"))[: len(scores)]
        count = round(len(candidates) * fps_out / fps_cand)
        for i, k in enumerate(pick_sharpest(scores[: len(candidates)], count)):
            shutil.move(candidates[k], out_dir / f"{i:05d}.jpg")
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
