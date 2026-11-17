"""Object masks for Kontext futures (add-object-masking).

The user names the object to change ("chairs"); Grounding DINO finds its boxes in the
start image, SAM turns them into a mask, and Kontext's result is kept only inside a grown
and feathered copy of that mask:  final = m * edited + (1 - m) * start.
Outside the mask the future is the start image, pixel for pixel.

Everything except `ObjectMasker` is pure numpy/PIL. `ObjectMasker` imports torch and
transformers only when it loads or runs, so the Space and the tests can import this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_QUERY = 60  # characters of "Object to change"
MAX_PHRASES = 5  # objects named in one query, separated by commas
BOX_THRESHOLD = 0.3  # Grounding DINO box score (model card default)
TEXT_THRESHOLD = 0.25  # Grounding DINO text score (model card default)
MAX_DETECTIONS = 20
GROW = 0.015  # mask grows by this fraction of the image's longer side (21 px at 1408)
GROW_REMOVAL = 0.04  # removals: room for the painted background and the shadow (56 px)
DIM = 0.35  # brightness of the unchanged part in the "Changed area" image

_REMOVAL = re.compile(r"\b(remove|delete|erase|take away|get rid of|without)\b", re.IGNORECASE)


class MaskError(ValueError):
    """An object query that cannot be used; the message says why."""


def object_phrases(query: str | None) -> list[str]:
    """The objects named in a query ("Chairs, the table" -> ["chairs", "the table"]).

    Empty for no query; raises MaskError for a query that is too long."""
    query = " ".join((query or "").split())
    if len(query) > MAX_QUERY:
        raise MaskError(
            f"'Object to change' is {len(query)} characters long; the limit is {MAX_QUERY}."
        )
    phrases = [p.strip(" .").lower() for p in re.split(r"[,;.]", query)]
    phrases = [p for p in phrases if p]
    if len(phrases) > MAX_PHRASES:
        raise MaskError(f"Name at most {MAX_PHRASES} objects in 'Object to change'.")
    return phrases


def detector_text(query: str) -> str:
    """Grounding DINO's text input: lowercase phrases, each ending with a period."""
    return " ".join(f"{p}." for p in object_phrases(query))


def is_removal(prompt: str) -> bool:
    """True when the prompt removes something; its mask then gets a wider margin.
    Reads only the verb, never which object is meant."""
    return bool(_REMOVAL.search(prompt or ""))


def mask_sizes(size: tuple[int, int], removal: bool) -> tuple[int, int]:
    """(grow, feather) in pixels for an image of `size` (width, height). The feather (a
    Gaussian sigma) is a third of the grow, so the object's own edge stays fully inside."""
    grow_px = max(3, round(max(size) * (GROW_REMOVAL if removal else GROW)))
    return grow_px, max(1, round(grow_px / 3))


def grow(mask: np.ndarray, radius: int) -> np.ndarray:
    """Boolean mask grown by `radius` pixels (square max filter via an integral image, so
    any radius costs the same)."""
    m = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return m.copy()
    k = 2 * radius + 1
    p = np.pad(m.astype(np.int32), radius)
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1), dtype=np.int64)
    ii[1:, 1:] = p.cumsum(0).cumsum(1)
    window = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    return window > 0


def soft_mask(hard: np.ndarray, removal: bool = False) -> np.ndarray:
    """The blend mask (float32 in [0, 1], exactly 0 far from the object) of a boolean
    object mask: grown, then feathered. Quantised to 8 bits, as stored in mask.png."""
    from PIL import Image, ImageFilter

    h, w = hard.shape
    grow_px, feather_px = mask_sizes((w, h), removal)
    grown = Image.fromarray((grow(hard, grow_px) * 255).astype(np.uint8))
    soft = grown.filter(ImageFilter.GaussianBlur(feather_px))
    return np.asarray(soft, dtype=np.float32) / 255


def composite(start: Any, edited: Any, mask: np.ndarray) -> Any:
    """`edited` inside the mask, `start` outside it (PIL images; result is `start`'s size)."""
    from PIL import Image

    if edited.size != start.size:
        edited = edited.resize(start.size, Image.LANCZOS)
    s = np.asarray(start.convert("RGB"), dtype=np.float32)
    e = np.asarray(edited.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)[..., None]
    out = m * e + (1 - m) * s
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def mask_image(mask: np.ndarray) -> Any:
    """The blend mask as an 8-bit greyscale PIL image (mask.png)."""
    from PIL import Image

    return Image.fromarray(np.clip(np.rint(mask * 255), 0, 255).astype(np.uint8))


def changed_area(image: Any, mask_img: Any) -> Any:
    """The future with everything outside the mask dimmed (the "Changed area" image)."""
    from PIL import Image

    image = image.convert("RGB")
    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size, Image.BILINEAR)
    m = np.asarray(mask_img.convert("L"), dtype=np.float32)[..., None] / 255
    out = np.asarray(image, dtype=np.float32) * (DIM + (1 - DIM) * m)
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in start-image pixels
    score: float
    label: str


def keep_detections(
    boxes: list, scores: list, labels: list, size: tuple[int, int]
) -> list[Detection]:
    """Detections clipped to the image, empty boxes dropped, best first, at most
    MAX_DETECTIONS."""
    w, h = size
    out = []
    for box, score, label in zip(boxes, scores, labels, strict=True):
        x0, y0, x1, y1 = (float(v) for v in box)
        x0, x1 = sorted((min(max(x0, 0), w), min(max(x1, 0), w)))
        y0, y1 = sorted((min(max(y0, 0), h), min(max(y1, 0), h)))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        out.append(Detection((round(x0), round(y0), round(x1), round(y1)), float(score),
                             str(label)))  # fmt: skip
    out.sort(key=lambda d: d.score, reverse=True)
    return out[:MAX_DETECTIONS]


def mask_info(
    query: str,
    detections: list[Detection],
    hard: np.ndarray,
    removal: bool,
    models: list[str] | tuple[str, ...] = (),
) -> dict:
    """What run.json records about a masked future (`mask`)."""
    h, w = hard.shape
    grow_px, feather_px = mask_sizes((w, h), removal)
    return {
        "query": query,
        "detections": len(detections),
        "boxes": [list(d.box) for d in detections],
        "scores": [round(d.score, 3) for d in detections],
        "labels": [d.label for d in detections],
        "area": round(float(np.asarray(hard, dtype=bool).mean()), 4),
        "grow_px": grow_px,
        "feather_px": feather_px,
        "removal": removal,
        "models": list(models),
    }


class ObjectMasker:
    """Grounding DINO (boxes for a text query) and SAM (a mask from the boxes)."""

    def __init__(self, detector: Any, segmenter: Any, device: str = "cuda") -> None:
        self.detector_spec, self.segmenter_spec = detector, segmenter  # models.ModelSpec
        self.device = device
        self.detector = self.det_processor = self.sam = self.sam_processor = None

    @property
    def labels(self) -> list[str]:
        return [self.detector_spec.label, self.segmenter_spec.label]

    def load(self) -> None:
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )

        d, s = self.detector_spec, self.segmenter_spec
        self.det_processor = AutoProcessor.from_pretrained(d.repo_id, revision=d.revision)
        self.detector = (
            AutoModelForZeroShotObjectDetection.from_pretrained(d.repo_id, revision=d.revision)
            .to(self.device)
            .eval()
        )
        self.sam_processor = SamProcessor.from_pretrained(s.repo_id, revision=s.revision)
        self.sam = SamModel.from_pretrained(s.repo_id, revision=s.revision).to(self.device).eval()

    def detect(self, image: Any, query: str) -> list[Detection]:
        """Boxes of the objects named in `query` in `image` (PIL), best first."""
        import torch

        if self.detector is None:
            self.load()
        inputs = self.det_processor(images=image, text=detector_text(query), return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.detector(**inputs)
        (r,) = self.det_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )
        labels = r.get("text_labels", r.get("labels"))
        return keep_detections(r["boxes"].tolist(), r["scores"].tolist(), list(labels),
                               image.size)  # fmt: skip

    def segment(self, image: Any, detections: list[Detection]) -> np.ndarray:
        """One boolean mask (height, width) covering every detection: per box, the best
        of SAM's three masks (highest predicted IoU)."""
        import torch

        if self.sam is None:
            self.load()
        boxes = [[list(map(float, d.box)) for d in detections]]
        inputs = self.sam_processor(image, input_boxes=boxes, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam(**inputs)
        (masks,) = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )  # (boxes, 3, height, width) bool
        best = outputs.iou_scores[0].argmax(-1).cpu()  # (boxes,)
        chosen = masks[torch.arange(len(detections)), best]
        return chosen.any(0).numpy()
