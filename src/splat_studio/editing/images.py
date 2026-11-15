"""Image editing tab: text-to-image and text + image-to-image with FLUX.1-schnell.

Plain Python so the Space and the tests can use it without a GPU. The Modal worker
(`ImageStudio` in modal_app/app.py) uses the same sizes and step counts.
"""

from __future__ import annotations

import math

# FLUX works best near one megapixel; sides must be multiples of 16 (VAE x patch size).
TARGET_PIXELS = 1024 * 1024
SIDE_MULTIPLE = 16
MAX_SIDE = 2048
MAX_ASPECT = 2.5

MODES = ("Text to image", "Text + image to image")
TEXT_TO_IMAGE, IMAGE_TO_IMAGE = MODES

# Label -> width / height, for text-to-image.
ASPECTS = {
    "Square (1:1)": 1.0,
    "Landscape (3:2)": 3 / 2,
    "Wide (16:9)": 16 / 9,
    "Portrait (2:3)": 2 / 3,
    "Tall (9:16)": 9 / 16,
}
DEFAULT_ASPECT = "Landscape (3:2)"

# schnell is distilled for 4 steps. img2img only runs the last `strength` share of the
# schedule, so more steps are scheduled to keep about 4 real ones.
BASE_STEPS = 4
MAX_STEPS = 12


def output_size(aspect: float, pixels: int = TARGET_PIXELS) -> tuple[int, int]:
    """(width, height) with about `pixels` pixels and sides multiples of 16."""
    a = min(max(float(aspect), 1 / MAX_ASPECT), MAX_ASPECT)
    h = math.sqrt(pixels / a)

    def snap(x: float) -> int:
        return min(MAX_SIDE, max(SIDE_MULTIPLE, int(round(x / SIDE_MULTIPLE)) * SIDE_MULTIPLE))

    return snap(h * a), snap(h)


def aspect_of(label: str | None) -> float:
    return ASPECTS.get(label or "", ASPECTS[DEFAULT_ASPECT])


def img2img_steps(strength: float) -> int:
    """Scheduled steps so that about BASE_STEPS of them actually run."""
    s = min(max(float(strength), 0.05), 1.0)
    return min(MAX_STEPS, max(BASE_STEPS, math.ceil(BASE_STEPS / s)))
