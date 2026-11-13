"""Preview (task 4.3): render the viewer's current view, then restyle it from a prompt.

The browser viewer (Three.js) reports its camera; the Modal job renders the same
view of the scene's `.splat` with gsplat (colour and depth), and SDXL img2img with
a depth ControlNet restyles the colour render while the depth keeps the layout.
The scene itself is never changed.

This module is plain numpy so the Space and the tests can use it without a GPU.

Camera conventions:
  - Three.js: camera looks down its -z axis, +y is up on screen (OpenGL).
    `matrixWorld.elements` is the camera-to-world matrix in column-major order.
  - gsplat: `viewmats` are world-to-camera, camera looks down +z, +y is down on
    screen (OpenCV). So flip the camera's y and z axes and invert.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np

# Three.js camera axes -> OpenCV camera axes.
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
# SDXL works best near one megapixel; sides must be multiples of 64 for the UNet.
TARGET_PIXELS = 1024 * 1024
SIDE_MULTIPLE = 64
MAX_ASPECT = 2.5


class CameraError(ValueError):
    """The viewer's camera could not be read."""


@dataclass(frozen=True)
class ViewCamera:
    """The viewer camera: camera-to-world (row-major, OpenGL axes), vertical fov, aspect."""

    c2w: tuple[tuple[float, ...], ...]
    fov_deg: float
    aspect: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ViewCamera:
        return cls(tuple(tuple(map(float, r)) for r in d["c2w"]), float(d["fov_deg"]),
                   float(d["aspect"]))  # fmt: skip


def camera_from_viewer(message: str | dict | None) -> ViewCamera:
    """Parse the viewer's `{"type": "camera", "matrix": [16], "fov": deg, "aspect": w/h}`."""
    if not message:
        raise CameraError("The viewer did not report its camera. Open a scene first.")
    try:
        d = json.loads(message) if isinstance(message, str) else dict(message)
        m = np.asarray(d["matrix"], dtype=np.float64)
        fov, aspect = float(d["fov"]), float(d["aspect"])
    except (ValueError, KeyError, TypeError) as e:
        raise CameraError(f"The viewer's camera could not be read ({e}).") from e
    if m.shape != (16,) or not np.isfinite(m).all():
        raise CameraError("The viewer's camera matrix is not a 4x4 matrix.")
    if not (1.0 <= fov <= 170.0) or not (0.1 <= aspect <= 10.0):
        raise CameraError(f"Unusual camera (fov {fov:.1f}, aspect {aspect:.2f}).")
    c2w = m.reshape(4, 4).T  # Three.js stores column-major
    if abs(np.linalg.det(c2w[:3, :3])) < 1e-6:
        raise CameraError("The viewer's camera matrix is degenerate.")
    return ViewCamera(tuple(tuple(round(float(x), 6) for x in r) for r in c2w), fov,
                      round(aspect, 6))  # fmt: skip


def viewmat(camera: ViewCamera) -> np.ndarray:
    """World-to-camera matrix in OpenCV axes, as gsplat's `viewmats` expects."""
    c2w = np.asarray(camera.c2w, dtype=np.float64)
    # Drop any scale in the rotation part (Three.js cameras normally have none).
    rot = c2w[:3, :3] / np.linalg.norm(c2w[:3, :3], axis=0, keepdims=True)
    c2w = np.vstack([np.hstack([rot, c2w[:3, 3:]]), [0, 0, 0, 1]])
    return np.linalg.inv(c2w @ _GL_TO_CV)


def output_size(aspect: float, pixels: int = TARGET_PIXELS) -> tuple[int, int]:
    """(width, height) with about `pixels` pixels, sides multiples of 64.

    Very wide or tall views are clamped to MAX_ASPECT; the render then shows a
    centre crop of the view rather than a sliver.
    """
    a = min(max(aspect, 1 / MAX_ASPECT), MAX_ASPECT)
    h = math.sqrt(pixels / a)

    def snap(x: float) -> int:
        return max(SIDE_MULTIPLE, int(round(x / SIDE_MULTIPLE)) * SIDE_MULTIPLE)

    return snap(h * a), snap(h)


def intrinsics(camera: ViewCamera, width: int, height: int) -> np.ndarray:
    """3x3 pinhole matrix that keeps the viewer's vertical field of view.

    If the output aspect differs from the viewer's (clamped), the horizontal view is
    cropped or widened around the centre; the focal length follows the height.
    """
    fy = (height / 2) / math.tan(math.radians(camera.fov_deg) / 2)
    return np.array([[fy, 0, width / 2], [0, fy, height / 2], [0, 0, 1]], dtype=np.float64)


# --- .splat decoding (layout in reconstruction/export.py) ---------------------------

SPLAT_DTYPE = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4),
                        ("rot", "u1", 4)])  # fmt: skip


@dataclass
class Gaussians:
    means: np.ndarray  # (N, 3) float32
    scales: np.ndarray  # (N, 3) float32, linear
    quats: np.ndarray  # (N, 4) float32, (w, x, y, z), unit length
    colors: np.ndarray  # (N, 3) float32 in [0, 1]
    opacities: np.ndarray  # (N,) float32 in [0, 1]

    def __len__(self) -> int:
        return len(self.means)


def read_splat(data: bytes) -> Gaussians:
    """Decode a `.splat` file (32 bytes per Gaussian)."""
    if len(data) % SPLAT_DTYPE.itemsize:
        raise ValueError("not a .splat file: size is not a multiple of 32 bytes")
    v = np.frombuffer(data, dtype=SPLAT_DTYPE)
    quats = (v["rot"].astype(np.float32) - 128.0) / 128.0
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = np.where(norm > 0, quats / np.maximum(norm, 1e-12), [1.0, 0.0, 0.0, 0.0])
    rgba = v["rgba"].astype(np.float32) / 255.0
    return Gaussians(
        means=np.ascontiguousarray(v["pos"], dtype=np.float32),
        scales=np.ascontiguousarray(v["scale"], dtype=np.float32),
        quats=quats.astype(np.float32),
        colors=np.ascontiguousarray(rgba[:, :3]),
        opacities=np.ascontiguousarray(rgba[:, 3]),
    )


# --- depth control image ------------------------------------------------------------

MIN_ALPHA = 0.5  # pixels less covered than this count as empty (background)


def depth_to_control(depth: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Depth map as the ControlNet expects it: uint8, near bright, far and empty dark.

    The SDXL depth ControlNet was trained on monocular depth estimates (MiDaS/DPT),
    which are inverse depth. Uses inverse depth, stretched between the 2nd and 98th
    percentile of covered pixels so a few very near or far Gaussians do not flatten
    the rest.
    """
    depth = np.asarray(depth, dtype=np.float64)
    covered = (np.asarray(alpha) >= MIN_ALPHA) & np.isfinite(depth) & (depth > 1e-6)
    out = np.zeros(depth.shape, dtype=np.uint8)
    if covered.sum() < 16:
        return out
    inv = np.zeros_like(depth)
    inv[covered] = 1.0 / depth[covered]
    lo, hi = np.percentile(inv[covered], [2, 98])
    if hi - lo < 1e-9:
        out[covered] = 255
        return out
    scaled = np.clip((inv - lo) / (hi - lo), 0, 1)
    # Covered pixels start at 10 % grey so the farthest surface still differs from empty.
    out[covered] = np.round(25 + 230 * scaled[covered]).astype(np.uint8)
    return out
