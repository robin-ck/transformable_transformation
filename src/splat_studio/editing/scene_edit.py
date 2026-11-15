"""Edit scene (task 4.4): the numpy side of the text-guided scene edit.

The edit follows Instruct-NeRF2NeRF's iterative dataset update, run on Modal
(`editing/edit_worker.py`): render a set of views, edit them with InstructPix2Pix,
fine-tune the Gaussians on the edited views, and keep re-editing renders of the
current scene so the edits agree across views.

The original training cameras are not stored, and later versions (e.g. with an
added object) have no photos at all, so the views are rendered from the scene
itself: an orbit around what the user is looking at in the viewer. The edit is
therefore strongest around that view.

Optional region: the user drags a box in the viewer. Only Gaussians that sit on
the visible surface inside that box may change (`select_region`).

Plain numpy, so the Space and the tests can use it without a GPU.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from .preview import ViewCamera, intrinsics, viewmat

DEFAULT_UP = (0.0, 0.0, 1.0)  # scenes from our pipeline are z-up (see space/viewer.py)


class RegionError(ValueError):
    """The marked region could not be used."""


# --- cameras ------------------------------------------------------------------------


def look_at_c2w(eye, target, up=DEFAULT_UP) -> np.ndarray:
    """Camera-to-world in Three.js/OpenGL axes (camera looks down -z), like Object3D.lookAt."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:  # looking straight along up: pick any side
        x = np.cross([1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def pivot_point(camera: ViewCamera, distance: float) -> np.ndarray:
    """The point `distance` in front of the camera (what the user is looking at)."""
    c2w = np.asarray(camera.c2w, dtype=np.float64)
    forward = -c2w[:3, 2] / np.linalg.norm(c2w[:3, 2])
    return c2w[:3, 3] + distance * forward


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of `v` about the unit `axis`."""
    return (
        v * math.cos(angle)
        + np.cross(axis, v) * math.sin(angle)
        + axis * (axis @ v) * (1 - math.cos(angle))
    )


def orbit_cameras(
    anchor: ViewCamera,
    pivot,
    n: int,
    yaw_deg: float = 30.0,
    pitch_deg: float = 10.0,
    zoom: float = 0.15,
    seed: int = 0,
    up=DEFAULT_UP,
) -> list[ViewCamera]:
    """`n` views around `pivot`: the anchor first, then random turns of the anchor view.

    Each view turns the anchor's position about the pivot by up to `yaw_deg` around
    the up axis and `pitch_deg` up or down, moves up to `zoom` closer or further, and
    looks at the pivot. Field of view and aspect are the anchor's.
    """
    if n < 1:
        raise ValueError("need at least one view")
    rng = np.random.default_rng(seed)
    up_v = np.asarray(up, dtype=np.float64)
    up_v /= np.linalg.norm(up_v)
    pivot = np.asarray(pivot, dtype=np.float64)
    eye0 = np.asarray(anchor.c2w, dtype=np.float64)[:3, 3]
    offset0 = eye0 - pivot
    cams = [anchor]
    for _ in range(n - 1):
        yaw = math.radians(rng.uniform(-yaw_deg, yaw_deg))
        pitch = math.radians(rng.uniform(-pitch_deg, pitch_deg))
        offset = _rotate(offset0, up_v, yaw)
        side = np.cross(up_v, offset)
        if np.linalg.norm(side) > 1e-8:
            offset = _rotate(offset, side / np.linalg.norm(side), pitch)
        offset *= 1 + rng.uniform(-zoom, zoom)
        c2w = look_at_c2w(pivot + offset, pivot, up_v)
        cams.append(ViewCamera(_rows(c2w), anchor.fov_deg, anchor.aspect))
    return cams


def _rows(m: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(x) for x in r) for r in m)


# --- region -------------------------------------------------------------------------

MIN_REGION_SIDE = 0.02  # of the view; smaller boxes are treated as accidental clicks


@dataclass(frozen=True)
class Region:
    """A box in the viewer, as fractions of its width and height (0,0 is top left)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


def region_from_viewer(message: str | dict | None) -> Region | None:
    """The `region` field of the viewer's camera message; None if no box was drawn."""
    if not message:
        return None
    d = json.loads(message) if isinstance(message, str) else dict(message)
    r = d.get("region")
    if r is None:
        return None
    try:
        x0, y0, x1, y1 = (float(c) for c in r)
    except (TypeError, ValueError) as e:
        raise RegionError("The marked region could not be read.") from e
    x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
    y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
    if x1 - x0 < MIN_REGION_SIDE or y1 - y0 < MIN_REGION_SIDE:
        raise RegionError("The marked region is too small; drag a larger box.")
    return Region(round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))


def region_pixels(region: Region, camera: ViewCamera, width: int, height: int):
    """The box in pixels of a `width` x `height` render of `camera`.

    The render keeps the viewer's vertical field of view; if its aspect was clamped
    (see `preview.output_size`), the box is mapped through the viewer's own width.
    """
    view_w = height * camera.aspect  # the viewer's width at the render's height
    u0 = width / 2 + (region.x0 - 0.5) * view_w
    u1 = width / 2 + (region.x1 - 0.5) * view_w
    return u0, region.y0 * height, u1, region.y1 * height


# The visible surface is estimated from the Gaussian centres, not from a depth render:
# rendered depth is an opacity-weighted mean, and a thin layer of haze in front of a
# surface pulls it far towards the camera (seen on the test scene: 0.7 instead of 1.5).
# Per screen cell, the 15th percentile of centre depths ignores sparse haze but finds
# the first dense surface; Gaussians up to 20 % behind it still count.
SURFACE_CELL_PX = 16
SURFACE_QUANTILE = 0.15
SURFACE_TOLERANCE = 0.2


def _project(means: np.ndarray, camera: ViewCamera, width: int, height: int):
    """Pixel position, depth and an in-view flag for each Gaussian centre."""
    p = np.c_[np.asarray(means, dtype=np.float64), np.ones(len(means))]
    cam = p @ viewmat(camera).T
    z = cam[:, 2]
    uvw = cam[:, :3] @ intrinsics(camera, width, height).T
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = uvw[:, 0] / uvw[:, 2], uvw[:, 1] / uvw[:, 2]
    visible = (z > 1e-4) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, z, visible


def view_depth(means: np.ndarray, camera: ViewCamera, width: int, height: int) -> float | None:
    """Median depth of the Gaussian centres in view: where the user is looking."""
    _, _, z, visible = _project(means, camera, width, height)
    return float(np.median(z[visible])) if visible.any() else None


def surface_depth(u: np.ndarray, v: np.ndarray, z: np.ndarray, cell: int = SURFACE_CELL_PX):
    """For each projected centre, the first dense surface depth in its screen cell."""
    cols = int(u.max() // cell) + 2 if len(u) else 1
    key = (v // cell).astype(np.int64) * cols + (u // cell).astype(np.int64)
    order = np.lexsort((z, key))
    ks, zs = key[order], z[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1] if len(ks) else np.zeros(0, int)
    ends = np.r_[starts[1:], len(ks)]
    picks = zs[starts + ((ends - starts) * SURFACE_QUANTILE).astype(np.int64)]
    per_cell = np.empty(len(ks))
    per_cell[order] = np.repeat(picks, ends - starts)
    return per_cell


def select_region(
    means: np.ndarray, camera: ViewCamera, width: int, height: int, region: Region
) -> np.ndarray:
    """Boolean mask of the Gaussians on the visible surface inside the box.

    A centre counts if it projects into the box and lies no more than SURFACE_TOLERANCE
    behind the first dense surface of its screen cell, so the wall behind a marked
    table is not edited along with it.
    """
    u, v, z, visible = _project(means, camera, width, height)
    out = np.zeros(len(means), dtype=bool)
    if not visible.any():
        return out
    idx = np.flatnonzero(visible)
    surface = surface_depth(u[idx], v[idx], z[idx])
    u0, v0, u1, v1 = region_pixels(region, camera, width, height)
    ui, vi = u[idx], v[idx]
    inside = (ui >= u0) & (ui < u1) & (vi >= v0) & (vi < v1)
    out[idx[inside & (z[idx] <= surface * (1 + SURFACE_TOLERANCE))]] = True
    return out


# --- PLY <-> training arrays ----------------------------------------------------------


def split_ply(v: np.ndarray) -> dict[str, np.ndarray]:
    """3DGS PLY fields as gsplat training arrays.

    means (N,3), scales (N,3) log, quats (N,4) wxyz raw, opacities (N,) logit,
    sh0 (N,1,3), shN (N,K-1,3). The PLY stores the higher-order terms channel by
    channel (all red terms, then green, then blue).
    """
    n = len(v)
    rest = sorted((f for f in v.dtype.names if f.startswith("f_rest_")), key=lambda f: int(f[7:]))
    if len(rest) % 3:
        raise ValueError(f"odd number of f_rest fields: {len(rest)}")
    k = len(rest) // 3
    shn = (
        np.stack([v[f] for f in rest], axis=1).reshape(n, 3, k).transpose(0, 2, 1)
        if k
        else np.zeros((n, 0, 3), dtype=np.float32)
    )

    def cols(*names: str) -> np.ndarray:
        return np.stack([v[f] for f in names], axis=1).astype(np.float32)

    return {
        "means": cols("x", "y", "z"),
        "scales": cols("scale_0", "scale_1", "scale_2"),
        "quats": cols("rot_0", "rot_1", "rot_2", "rot_3"),
        "opacities": np.asarray(v["opacity"], dtype=np.float32),
        "sh0": cols("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :],
        "shN": np.ascontiguousarray(shn, dtype=np.float32),
    }


def join_ply(v: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Copy of `v` with the fields replaced from `split_ply`-style arrays."""
    out = v.copy()
    for i, f in enumerate("xyz"):
        out[f] = arrays["means"][:, i]
    for i in range(3):
        out[f"scale_{i}"] = arrays["scales"][:, i]
        out[f"f_dc_{i}"] = arrays["sh0"][:, 0, i]
    for i in range(4):
        out[f"rot_{i}"] = arrays["quats"][:, i]
    out["opacity"] = arrays["opacities"]
    shn = arrays["shN"]
    k = shn.shape[1]
    flat = shn.transpose(0, 2, 1).reshape(len(v), 3 * k)
    for i in range(3 * k):
        out[f"f_rest_{i}"] = flat[:, i]
    return out


def sh_degree(k_total: int) -> int:
    """SH degree for `k_total` coefficients per channel (1, 4, 9, 16 -> 0..3)."""
    d = int(round(math.sqrt(k_total))) - 1
    if (d + 1) ** 2 != k_total:
        raise ValueError(f"{k_total} is not a square number of SH coefficients")
    return d
