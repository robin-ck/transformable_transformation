"""Add object (task 4.2): normalise a generated object, place it, merge it into a scene.

Coordinates follow the scene: z points up (see space/viewer.py). A generated object
is stored normalised: its base centre sits at the origin and its largest side is 1,
so a placement reads as "put the base here, this big, turned this far".

The same placement is applied in two places that must agree:
  - the browser viewer, which moves the object live (position, quaternion, scale);
  - `apply_placement`, which bakes it into the Gaussians for the new version.
Both use p' = size * Rz(turn) * Ry(tilt_y) * Rx(tilt_x) * p + (x, y, z): the object is
tilted about its own horizontal axes first, then turned about the up axis.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from splat_studio.reconstruction.export import read_ply, to_splat_bytes, write_ply

# Gaussians this transparent are ignored when measuring an object's size, so faint
# haze around it does not shrink the visible object.
MIN_OPACITY_FOR_BOUNDS = 0.1


def _xyz(v: np.ndarray) -> np.ndarray:
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


def _set_xyz(v: np.ndarray, p: np.ndarray) -> None:
    v["x"], v["y"], v["z"] = p[:, 0], p[:, 1], p[:, 2]


def _opacity(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))


def bounds(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust bounding box: 1st-99th percentile of the visible Gaussians."""
    p = _xyz(v)
    visible = _opacity(v) >= MIN_OPACITY_FOR_BOUNDS
    if visible.sum() >= 10:
        p = p[visible]
    return np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)


def normalise_object(v: np.ndarray) -> np.ndarray:
    """Copy of `v` with its base centre at the origin and its largest side 1."""
    lo, hi = bounds(v)
    extent = float(np.max(hi - lo))
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("object has no extent")
    pivot = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    out = v.copy()
    _set_xyz(out, (_xyz(v) - pivot) / extent)
    for i in range(3):
        out[f"scale_{i}"] = v[f"scale_{i}"] - math.log(extent)  # scales are stored as log
    return out


@dataclass(frozen=True)
class Placement:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    turn_deg: float = 0.0  # about the up (z) axis
    size: float = 1.0  # largest side of the object, in scene units
    tilt_x_deg: float = 0.0  # about the object's x axis (applied before the turn)
    tilt_y_deg: float = 0.0  # about the object's y axis (applied before the turn)

    def quaternion_wxyz(self) -> tuple[float, float, float, float]:
        """Rz(turn) * Ry(tilt_y) * Rx(tilt_x) as a (w, x, y, z) quaternion."""

        def about(axis: int, deg: float) -> np.ndarray:
            half = math.radians(deg) / 2
            q = np.zeros(4)
            q[0], q[axis + 1] = math.cos(half), math.sin(half)
            return q

        q = _quat_mul(
            about(2, self.turn_deg),
            _quat_mul(about(1, self.tilt_y_deg), about(0, self.tilt_x_deg)),
        )
        return tuple(float(c) for c in q)  # type: ignore[return-value]

    def is_rotated(self) -> bool:
        return any(d % 360 for d in (self.turn_deg, self.tilt_x_deg, self.tilt_y_deg))

    def to_viewer(self) -> dict:
        """Placement as the viewer's Three.js object expects it (quaternion is x, y, z, w)."""
        w, qx, qy, qz = self.quaternion_wxyz()
        return {
            "position": [self.x, self.y, self.z],
            "quaternion": [qx, qy, qz, w],
            "scale": [self.size] * 3,
            "turn": self.turn_deg,  # so the viewer's drag-to-turn/tilt starts from here
            "tiltX": self.tilt_x_deg,
            "tiltY": self.tilt_y_deg,
        }

    def to_dict(self) -> dict:
        return asdict(self)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (w, x, y, z) quaternions; broadcasts over rows."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def apply_placement(v: np.ndarray, placement: Placement) -> np.ndarray:
    """Copy of a normalised object `v` moved, turned and scaled into the scene."""
    if placement.size <= 0:
        raise ValueError("size must be positive")
    q = np.array(placement.quaternion_wxyz())
    w, qx, qy, qz = q
    rot = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - w * qz), 2 * (qx * qz + w * qy)],
            [2 * (qx * qy + w * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - w * qx)],
            [2 * (qx * qz - w * qy), 2 * (qy * qz + w * qx), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    out = v.copy()
    p = placement.size * (_xyz(v) @ rot.T) + np.array([placement.x, placement.y, placement.z])
    _set_xyz(out, p)
    for i in range(3):
        out[f"scale_{i}"] = v[f"scale_{i}"] + math.log(placement.size)
    r = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    r = _quat_mul(np.broadcast_to(q, r.shape), r)
    for i in range(4):
        out[f"rot_{i}"] = r[:, i]
    # Higher-order colour terms (view-dependent shine) would need rotating too; drop them
    # when the object is turned or tilted. Generated objects normally have none.
    rest = [n for n in v.dtype.names if n.startswith("f_rest_")]
    if rest and placement.is_rotated():
        for n in rest:
            out[n] = 0
    return out


def merge(scene: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """Scene Gaussians followed by the object's, in the scene's field layout.

    Fields the object lacks (e.g. higher-order colour terms) are filled with 0;
    fields only the object has are dropped.
    """
    extra = np.zeros(len(obj), dtype=scene.dtype)
    for name in scene.dtype.names:
        if name in obj.dtype.names:
            extra[name] = obj[name]
    return np.concatenate([scene, extra])


def merge_files(
    scene_ply: Path,
    object_ply: Path,
    placement: Placement,
    out_ply: Path,
    out_splat: Path,
    object_counts: list[int] | None = None,
) -> int:
    """Write the merged scene as `.ply` and `.splat`; returns the number of Gaussians.

    `object_counts` are the sizes of the objects already at the end of `scene_ply`
    (oldest first, from the parent version's `objects` list). The `.splat` is written in
    segments: base scene, each earlier object, then the new one (see `to_splat_bytes`),
    so the new object takes the last `len(object)` Gaussians of both files.
    """
    scene = read_ply(scene_ply)
    obj = apply_placement(read_ply(object_ply), placement)
    earlier = list(object_counts or [])
    base = len(scene) - sum(earlier)
    if base < 0:
        raise ValueError("object counts exceed the scene's Gaussians")
    merged = merge(scene, obj)
    write_ply(out_ply, merged)
    out_splat.write_bytes(to_splat_bytes(merged, [base, *earlier, len(obj)]))
    return len(merged)


def default_placement(scene_positions: np.ndarray) -> tuple[Placement, float]:
    """Start placement (scene centre, a quarter of the scene radius tall) and the radius.

    The radius sets the slider ranges in the Space.
    """
    finite = scene_positions[np.isfinite(scene_positions).all(axis=1)]
    if len(finite) == 0:
        return Placement(), 1.0
    centre = np.median(finite, axis=0)
    radius = float(np.percentile(np.linalg.norm(finite - centre, axis=1), 80))
    radius = max(radius, 1e-3)
    x, y, z = (round(float(c), 3) for c in centre)
    return Placement(x, y, z, 0.0, round(radius / 4, 3)), radius
