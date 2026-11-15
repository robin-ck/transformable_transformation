"""Edit scene (task 4.4): change a whole scene, or one region, from a text instruction.

Approach: iterative dataset update in the style of Instruct-NeRF2NeRF.
  1. Render a ring of views of the scene (or of the marked region) with gsplat.
  2. Edit each view with InstructPix2Pix, conditioned on the original render and
     started from the current render, so later rounds refine rather than restart.
  3. Fine-tune the Gaussians on the edited views; repeat 2-3 for a few rounds, so the
     per-view edits, which disagree a little, settle into one consistent scene.

The original training cameras are not stored with a scene, so the views are virtual:
an orbit around the scene centre (median of the Gaussians) at the distance the browser
viewer starts from, or around the region. Parts of the scene that no orbit view sees
(e.g. the inside of a street seen only from within) change little.

With a region (a sphere in scene coordinates, marked in the viewer), only Gaussians
whose centre lies inside it are optimised; everything else stays as it was.

This module is plain numpy, so the Space and tests use it without a GPU. The GPU part
(rendering, InstructPix2Pix, fine-tuning) is `scene_edit_gpu.py`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .preview import ViewCamera

# Defaults for one edit, sized for an A10G within EDIT_TIMEOUT_S (see modal_app/app.py):
# 36 views x 3 rounds = 108 InstructPix2Pix calls, 3 x 600 fine-tuning steps.
NUM_VIEWS = 36
ROUNDS = 3
STEPS_PER_ROUND = 600
VIEW_SIZE = 512  # square renders; InstructPix2Pix (SD 1.5) works at 512 px
VIEW_FOV_DEG = 60.0
EDIT_STEPS = 20  # diffusion steps per view edit
TEXT_GUIDANCE = 7.5
IMAGE_GUIDANCE = 1.5
# How much of the current render is noised before editing (Instruct-NeRF2NeRF samples
# 0.02-0.98). The first round edits strongly; later rounds only refine.
FIRST_ROUND_STRENGTH = (0.85, 0.98)
LATER_ROUND_STRENGTH = (0.5, 0.8)
ELEVATIONS_DEG = (15.0, 30.0, 45.0)
SCENE_DISTANCE = 1.8  # x scene radius, as the viewer's start camera
REGION_DISTANCE = 2.5  # x region radius
MIN_REGION_GAUSSIANS = 50


class EditError(ValueError):
    """A readable reason why an edit cannot start."""


@dataclass(frozen=True)
class Region:
    """A sphere in scene coordinates; only Gaussians inside it are edited."""

    center: tuple[float, float, float]
    radius: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> Region | None:
        """Region from `{"center": [x, y, z], "radius": r}`; None if `d` is empty."""
        if not d:
            return None
        try:
            center = tuple(float(c) for c in d["center"])
            radius = float(d["radius"])
        except (KeyError, TypeError, ValueError) as e:
            raise EditError(f"The marked region could not be read ({e}).") from e
        if len(center) != 3 or not all(math.isfinite(c) for c in center):
            raise EditError("The marked region has no valid centre.")
        if not (math.isfinite(radius) and radius > 0):
            raise EditError("The marked region needs a positive size.")
        return cls((center[0], center[1], center[2]), radius)


def scene_centre_radius(positions: np.ndarray) -> tuple[np.ndarray, float]:
    """Median centre and 80th-percentile radius, ignoring far floaters (as the viewer does)."""
    finite = positions[np.isfinite(positions).all(axis=1)] if len(positions) else positions
    if len(finite) == 0:
        return np.zeros(3), 1.0
    centre = np.median(finite, axis=0)
    radius = float(np.percentile(np.linalg.norm(finite - centre, axis=1), 80))
    return centre, max(radius, 1e-3)


def region_mask(means: np.ndarray, region: Region | None) -> np.ndarray:
    """True for Gaussians that may change: all without a region, else those inside it."""
    if region is None:
        return np.ones(len(means), dtype=bool)
    d = np.linalg.norm(np.asarray(means, dtype=np.float64) - np.array(region.center), axis=1)
    return d <= region.radius


def check_region(means: np.ndarray, region: Region | None) -> np.ndarray:
    """`region_mask`, but refuses a region with too few Gaussians to edit."""
    mask = region_mask(means, region)
    if region is not None and mask.sum() < MIN_REGION_GAUSSIANS:
        raise EditError(
            "The marked region holds almost nothing of the scene. Click on a surface "
            "to place it, or make it larger."
        )
    return mask


def look_at(position, target, up=(0.0, 0.0, 1.0), fov_deg: float = VIEW_FOV_DEG) -> ViewCamera:
    """Camera at `position` looking at `target`, in the viewer's convention.

    Camera-to-world with OpenGL axes (looks down -z, +y up on screen), like the
    Three.js camera, so `preview.viewmat` turns it into gsplat's world-to-camera.
    """
    pos = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - pos
    forward /= np.linalg.norm(forward)
    up_v = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, up_v)
    if np.linalg.norm(right) < 1e-9:  # looking straight up or down: any side will do
        right = np.cross(forward, np.eye(3)[np.argmin(np.abs(forward))])
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, cam_up, -forward, pos
    return ViewCamera(tuple(tuple(float(x) for x in r) for r in c2w), fov_deg, 1.0)


def orbit_cameras(
    positions: np.ndarray, region: Region | None = None, n: int = NUM_VIEWS
) -> list[ViewCamera]:
    """`n` views around the scene (or region), at a few heights, all looking at its centre.

    Scenes are z-up (see space/viewer.py). Azimuths are spread evenly; each height
    ring is turned a little so views do not line up.
    """
    if region is None:
        centre, radius = scene_centre_radius(positions)
        distance = SCENE_DISTANCE * radius
    else:
        centre, distance = np.array(region.center), REGION_DISTANCE * region.radius
    cams = []
    for i in range(n):
        ring = i % len(ELEVATIONS_DEG)
        elev = math.radians(ELEVATIONS_DEG[ring])
        azim = 2 * math.pi * i / n + ring * 0.37
        offset = np.array(
            [math.cos(azim) * math.cos(elev), math.sin(azim) * math.cos(elev), math.sin(elev)]
        )
        cams.append(look_at(centre + distance * offset, centre))
    return cams


def strength_for(round_index: int, rng: np.random.Generator) -> float:
    """Noise strength for one view edit in round `round_index` (0-based)."""
    lo, hi = FIRST_ROUND_STRENGTH if round_index == 0 else LATER_ROUND_STRENGTH
    return float(rng.uniform(lo, hi))


def combine_guidance(text, image, uncond, text_scale=TEXT_GUIDANCE, image_scale=IMAGE_GUIDANCE):
    """InstructPix2Pix's two-way classifier-free guidance (works on numpy or torch).

    `text`: noise predicted with instruction and image; `image`: with image only;
    `uncond`: with neither.
    """
    return uncond + text_scale * (text - image) + image_scale * (image - uncond)


def write_params(
    v: np.ndarray,
    means: np.ndarray,
    log_scales: np.ndarray,
    quats_wxyz: np.ndarray,
    opacity_logits: np.ndarray,
    sh0: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Copy of the scene `v` (PLY fields) with the fine-tuned values of the masked rows.

    The fine-tuning only learns the view-independent colour (`f_dc_*`), so the
    higher-order colour terms (`f_rest_*`) of edited Gaussians are cleared; they would
    otherwise keep a trace of the old look from some angles. Unmasked rows are kept
    exactly as they were.
    """
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(v):
        raise ValueError("mask and scene differ in length")
    out = v.copy()
    fields = {
        ("x", "y", "z"): means,
        ("scale_0", "scale_1", "scale_2"): log_scales,
        ("rot_0", "rot_1", "rot_2", "rot_3"): quats_wxyz,
        ("f_dc_0", "f_dc_1", "f_dc_2"): sh0,
    }
    for names, values in fields.items():
        values = np.asarray(values)
        for i, name in enumerate(names):
            out[name][mask] = values[mask, i]
    out["opacity"][mask] = np.asarray(opacity_logits)[mask]
    for name in (n for n in v.dtype.names if n.startswith("f_rest_")):
        out[name][mask] = 0
    return out


def plan_summary(n_views: int = NUM_VIEWS, rounds: int = ROUNDS) -> dict:
    """Settings recorded in the run log and `version.json`."""
    return {
        "views": n_views,
        "rounds": rounds,
        "steps_per_round": STEPS_PER_ROUND,
        "view_size": VIEW_SIZE,
        "edit_steps": EDIT_STEPS,
        "text_guidance": TEXT_GUIDANCE,
        "image_guidance": IMAGE_GUIDANCE,
    }
