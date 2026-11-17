"""Drawn areas for Kontext futures (add-area-layout).

The user paints the area to change on the start image ("a willow here", "turn this
meadow into a wetland"). Unlike "Object to change" this also works for things that are
not in the view yet and for areas without clear edges. The worker:

1. crops the start image around the area, with context on every side (`crop_box`), so
   Kontext sees the area large and in the middle of its input;
2. edits the crop with an instruction that places the change in the middle
   (`area_prompt`);
3. pastes the edited crop back (`paste`) and keeps it only inside a grown, feathered
   copy of the painted area (`masking.soft_mask`, `masking.composite`).

Outside the area the future is the start image, pixel for pixel. Pure numpy/PIL.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from . import masking

MIN_COVER = 0.002  # smallest area, as a fraction of the image (about 40 x 50 px at 1408)
CONTEXT = 0.6  # the crop adds this fraction of the area's size on every side
MIN_CROP = 0.35  # the crop's shorter side is at least this fraction of the image's
ALPHA_ON = 32  # a painted pixel: alpha above this (anti-aliased brush edges are lower)


class AreaError(ValueError):
    """A drawn area that cannot be used; the message says why."""


def from_layers(layers: list[Any], size: tuple[int, int] | None = None) -> np.ndarray:
    """The painted area (bool, height x width) of an image editor's layers.

    `layers` are RGBA images (PIL or arrays); a pixel is painted when any layer's alpha
    is above ALPHA_ON. `size` (width, height) resizes the result, nearest neighbour, to
    the start image's size. An empty list gives an empty mask (of `size`, else 1 x 1).
    """
    from PIL import Image

    hard = None
    for layer in layers or []:
        a = np.asarray(layer)
        if a.ndim != 3 or a.shape[2] < 4:
            continue
        on = a[..., 3] > ALPHA_ON
        hard = on if hard is None or hard.shape != on.shape else hard | on
    if hard is None:
        w, h = size or (1, 1)
        return np.zeros((h, w), dtype=bool)
    if size is not None and (hard.shape[1], hard.shape[0]) != tuple(size):
        img = Image.fromarray(hard.astype(np.uint8) * 255).resize(size, Image.NEAREST)
        hard = np.asarray(img) > 127
    return hard


def check(hard: np.ndarray) -> np.ndarray:
    """`hard` as a bool array; raises AreaError when almost nothing is painted."""
    hard = np.asarray(hard, dtype=bool)
    if hard.size == 0 or hard.mean() < MIN_COVER:
        raise AreaError(
            "The drawn area is empty or too small. Paint over the part of the view that "
            "should change, or clear the drawing to change the whole view."
        )
    return hard


def to_png(hard: np.ndarray) -> bytes:
    """The area as a 1-bit PNG (a few kB; sent to the worker, stored as area.png)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(hard, dtype=bool)).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def from_png(data: bytes, size: tuple[int, int] | None = None) -> np.ndarray:
    """An area PNG as a bool mask, resized (nearest) to `size` (width, height) if given."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("L")
        if size is not None and img.size != tuple(size):
            img = img.resize(size, Image.NEAREST)
        return np.asarray(img) > 127


def bounds(hard: np.ndarray) -> tuple[int, int, int, int]:
    """The painted pixels' box (x0, y0, x1, y1), x1 and y1 exclusive."""
    ys, xs = np.nonzero(hard)
    if len(xs) == 0:
        raise AreaError("The drawn area is empty.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _span(lo: int, hi: int, want: int, limit: int) -> tuple[int, int]:
    """[lo, hi) widened evenly to `want` pixels, shifted to stay inside [0, limit)."""
    want = min(max(want, hi - lo), limit)
    lo -= (want - (hi - lo)) // 2
    lo = min(max(lo, 0), limit - want)
    return lo, lo + want


def crop_box(hard: np.ndarray) -> tuple[int, int, int, int]:
    """The part of the image Kontext edits: the area's box with CONTEXT on every side,
    at least MIN_CROP of the image's shorter side, inside the image."""
    h, w = hard.shape
    x0, y0, x1, y1 = bounds(hard)
    floor = round(min(w, h) * MIN_CROP)
    cw = max(round((x1 - x0) * (1 + 2 * CONTEXT)), floor)
    ch = max(round((y1 - y0) * (1 + 2 * CONTEXT)), floor)
    cx0, cx1 = _span(x0, x1, cw, w)
    cy0, cy1 = _span(y0, y1, ch, h)
    return cx0, cy0, cx1, cy1


def paste(start: Any, edited_crop: Any, box: tuple[int, int, int, int]) -> Any:
    """`start` (PIL) with `edited_crop` resized into `box`; the full-size edited image."""
    from PIL import Image

    x0, y0, x1, y1 = box
    out = start.convert("RGB").copy()
    crop = edited_crop.convert("RGB")
    if crop.size != (x1 - x0, y1 - y0):
        crop = crop.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    out.paste(crop, (x0, y0))
    return out


def area_prompt(prompt: str) -> str:
    """The instruction Kontext gets for the crop: the change sits in the middle of it.

    Kontext cannot see the drawn area, but the crop puts it in the middle; the result is
    kept inside the area only."""
    text = " ".join((prompt or "").split()).rstrip(" .") or "Change this place"
    if masking.is_removal(prompt):
        tail = ("Fill the space in the middle of the image with what is behind it, "
                "matching the surroundings and the lighting. Keep everything else "
                "unchanged.")  # fmt: skip
    else:
        tail = ("Make the change in the middle of the image, matching the perspective, "
                "scale, lighting and style of the scene. Keep everything else "
                "unchanged.")  # fmt: skip
    return f"{text}. {tail}"


def area_info(hard: np.ndarray, box: tuple[int, int, int, int], removal: bool) -> dict:
    """What run.json records about an area future (`area`)."""
    h, w = hard.shape
    grow_px, feather_px = masking.mask_sizes((w, h), removal)
    return {
        "bounds": list(bounds(hard)),
        "crop": list(box),
        "cover": round(float(hard.mean()), 4),
        "grow_px": grow_px,
        "feather_px": feather_px,
        "removal": removal,
    }


def summary(hard: np.ndarray) -> dict:
    """The area as stored in the future's params: its box and how much of the view it
    covers (the mask itself is area.png, written by the worker)."""
    hard = check(hard)
    return {"bounds": list(bounds(hard)), "cover": round(float(hard.mean()), 4)}
