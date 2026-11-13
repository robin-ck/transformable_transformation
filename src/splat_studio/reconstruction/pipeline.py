"""End-to-end reconstruction on one machine (runs inside the Modal GPU container)."""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import colmap, errors, export, frames

VIDEO_EXT = {".mp4", ".mov"}
PHOTO_EXT = {".jpg", ".jpeg", ".png"}
MIN_FRAMES = 20
DEFAULT_STEPS = 7000  # gsplat default is 30k; 7k keeps cost low for v1


@dataclass
class Result:
    ply: Path
    splat: Path
    num_gaussians: int
    num_frames: int


def classify_inputs(input_dir: Path) -> tuple[str, list[Path]]:
    """Return ("video", [file]) or ("photos", files); raises on mixed or empty input."""
    files = sorted(p for p in input_dir.rglob("*") if p.is_file())
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXT]
    photos = [p for p in files if p.suffix.lower() in PHOTO_EXT]
    if len(videos) == 1 and not photos:
        return "video", videos
    if photos and not videos:
        return "photos", photos
    raise errors.ReconstructionError(
        "bad_input", "Upload either one video or a set of photos, not both or neither."
    )


def train_command(gsplat_dir: Path, data_dir: Path, result_dir: Path, steps: int) -> list[str]:
    return [
        "python", str(gsplat_dir / "examples" / "simple_trainer.py"), "default",
        "--data_dir", str(data_dir), "--result_dir", str(result_dir),
        "--data_factor", "1", "--max_steps", str(steps),
        "--save_steps", str(steps), "--ply_steps", str(steps), "--save_ply",
        "--disable_viewer",
    ]  # fmt: skip


def find_ply(result_dir: Path) -> Path:
    plys = sorted((result_dir / "ply").glob("*.ply"), key=lambda p: p.stat().st_mtime)
    if not plys:
        raise errors.training_failed("trainer finished without writing a .ply file")
    return plys[-1]


def run(
    input_dir: Path,
    work_dir: Path,
    gsplat_dir: Path,
    steps: int = DEFAULT_STEPS,
    on_stage: Callable[[str, float], None] | None = None,
) -> Result:
    """Frames -> poses -> training -> export. Raises ReconstructionError on failure."""

    def timed(name: str, fn: Callable):
        t = time.time()
        out = fn()
        if on_stage:
            on_stage(name, time.time() - t)
        return out

    kind, files = classify_inputs(input_dir)
    data_dir = work_dir / "data"
    images = data_dir / "images"
    images.mkdir(parents=True, exist_ok=True)

    if kind == "video":
        n = timed("frames", lambda: frames.extract_frames(files[0], images))
    else:
        for i, p in enumerate(files):
            shutil.copy(p, images / f"{i:05d}{p.suffix.lower()}")
        n = len(files)
    if n < MIN_FRAMES:
        raise errors.too_few_frames(n, MIN_FRAMES)

    timed("poses", lambda: colmap.run(data_dir, sequential=kind == "video"))

    result_dir = work_dir / "result"
    res = timed(
        "train",
        lambda: subprocess.run(
            train_command(gsplat_dir, data_dir, result_dir, steps), capture_output=True, text=True
        ),
    )
    if res.returncode != 0:
        raise errors.training_failed(res.stderr)
    ply = find_ply(result_dir)

    splat = work_dir / "scene.splat"
    count = timed("export", lambda: export.ply_to_splat(ply, splat))
    return Result(ply=ply, splat=splat, num_gaussians=count, num_frames=n)
