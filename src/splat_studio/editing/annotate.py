"""Labelled detection image for object-masked futures (add-detection-preview).

Draws the start image with every detection's mask (semi-transparent) and box in its own
colour, labelled "<n> <label> <score>" on a filled background, so the user can see which
objects "Object to change" matched. Pure PIL/numpy; no GPU.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

# Distinct, saturated colours (Tableau 10 order); detection i gets COLOURS[i % 10].
COLOURS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (188, 189, 34),
    (23, 190, 207),
    (127, 127, 127),
)
MASK_ALPHA = 0.45  # opacity of a detection's mask over the image


def label_text(n: int, label: str, score: float) -> str:
    """The text on detection number `n` (1-based), e.g. "1 chair 0.62"."""
    return f"{n} {' '.join(label.split())} {score:.2f}"


def detections_caption(labels: list[str]) -> str:
    """ "Detected: 3 × chair, 1 × stool" (most frequent first; ties in first-seen order)."""
    counts = Counter(" ".join(lb.split()) or "object" for lb in labels)
    if not counts:
        return "Detected: nothing"
    return "Detected: " + ", ".join(f"{n} × {lb}" for lb, n in counts.most_common())


def _font(size: int) -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)  # scalable since Pillow 10.1
    except TypeError:
        return ImageFont.load_default()


def draw_detections(image: Any, detections: list, masks: np.ndarray | None = None) -> Any:
    """The labelled image (RGB, `image`'s size).

    `detections` are `masking.Detection`s (box in image pixels, score, label), numbered
    1, 2, ... in the given order; `masks` is a (len(detections), height, width) bool array,
    or None to draw boxes only.
    """
    from PIL import Image, ImageDraw

    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    if masks is not None:
        for i, m in enumerate(np.asarray(masks, dtype=bool)[: len(detections)]):
            colour = np.array(COLOURS[i % len(COLOURS)], dtype=np.float32)
            base[m] = (1 - MASK_ALPHA) * base[m] + MASK_ALPHA * colour
    out = Image.fromarray(np.clip(np.rint(base), 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    w, h = out.size
    line = max(2, round(max(w, h) / 400))
    font = _font(max(12, round(max(w, h) / 45)))  # readable when shown small
    pad = max(2, line)
    for i, d in enumerate(detections):
        colour = COLOURS[i % len(COLOURS)]
        x0, y0, x1, y1 = d.box
        draw.rectangle((x0, y0, x1, y1), outline=colour, width=line)
        text = label_text(i + 1, d.label, d.score)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left + 2 * pad, bottom - top + 2 * pad
        # Above the box if there is room, else inside its top; always inside the image.
        tx = min(max(x0, 0), max(w - tw, 0))
        ty = y0 - th if y0 - th >= 0 else min(y0, max(h - th, 0))
        draw.rectangle((tx, ty, tx + tw, ty + th), fill=colour)
        draw.text((tx + pad - left, ty + pad - top), text, fill=(255, 255, 255), font=font)
    return out
