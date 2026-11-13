"""Camera pose estimation with COLMAP (CPU build) and checks on the result."""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

from . import errors

MIN_REGISTERED_RATIO = 0.5
# Mapper log lines that mean "the footage did not reconstruct", as opposed to a crash.
POSE_FAILURE_MARKERS = (
    "No good initial image pair found",
    "failed to create sparse model",
    "Failed to create sparse model",
)


def commands(data_dir: Path, sequential: bool) -> list[list[str]]:
    """COLMAP commands for a gsplat-style data dir (`images/` in, `sparse/<n>/` out)."""
    db, images, sparse = data_dir / "database.db", data_dir / "images", data_dir / "sparse"
    matcher = "sequential_matcher" if sequential else "exhaustive_matcher"
    return [
        ["colmap", "feature_extractor", "--database_path", str(db), "--image_path", str(images),
         "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV",
         "--SiftExtraction.use_gpu", "0"],
        ["colmap", matcher, "--database_path", str(db), "--SiftMatching.use_gpu", "0"],
        ["colmap", "mapper", "--database_path", str(db), "--image_path", str(images),
         "--output_path", str(sparse)],
    ]  # fmt: skip


def registered_images(model_dir: Path) -> int:
    """Number of registered images in a binary COLMAP model (first uint64 of images.bin)."""
    path = model_dir / "images.bin"
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def best_model(sparse_dir: Path) -> tuple[Path | None, int]:
    """Pick the sub-model with most registered images."""
    best, best_n = None, 0
    if sparse_dir.exists():
        for sub in sorted(p for p in sparse_dir.iterdir() if p.is_dir()):
            n = registered_images(sub)
            if n > best_n:
                best, best_n = sub, n
    return best, best_n


def check_model(sparse_dir: Path, total_images: int, log: str = "") -> Path:
    """Return the best model dir, or raise a readable error."""
    model, n = best_model(sparse_dir)
    if model is None:
        raise errors.no_poses(log)
    if n < MIN_REGISTERED_RATIO * total_images:
        raise errors.low_registration(n, total_images, MIN_REGISTERED_RATIO)
    return model


def normalise_layout(sparse_dir: Path, model: Path) -> None:
    """gsplat reads `sparse/0`; make sure the best model sits there."""
    target = sparse_dir / "0"
    if model == target:
        return
    if target.exists():
        target.rename(sparse_dir / "_0_smaller")
    model.rename(target)


def is_pose_failure(log: str) -> bool:
    """True if the mapper log says the footage could not be reconstructed (not a crash)."""
    return any(m in log for m in POSE_FAILURE_MARKERS)


def run(data_dir: Path, sequential: bool) -> Path:
    """Run COLMAP and return the model dir used for training."""
    sparse = data_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)  # mapper exits with an error if it is missing
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}  # no display in the container
    mapper_log = ""
    for cmd in commands(data_dir, sequential):
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        log = f"$ {' '.join(cmd[:2])}\n{res.stdout[-1000:]}\n{res.stderr[-3000:]}"
        if cmd[1] == "mapper":
            mapper_log = log
        if res.returncode != 0:
            if cmd[1] == "mapper" and is_pose_failure(log):
                raise errors.no_poses(log)
            raise errors.pose_tool_failed(cmd[1], log)
    total = len(list((data_dir / "images").iterdir()))
    model = check_model(sparse, total, mapper_log)
    normalise_layout(sparse, model)
    return sparse / "0"
