"""Labelled detection image of add-detection-preview (numpy and PIL only)."""

import numpy as np
from PIL import Image

from splat_studio.editing import annotate
from splat_studio.editing.masking import Detection


def test_label_and_caption_text():
    assert annotate.label_text(1, " chair ", 0.6234) == "1 chair 0.62"
    assert annotate.detections_caption(["chair", "stool", "chair", "chair"]) == (
        "Detected: 3 × chair, 1 × stool"
    )
    assert annotate.detections_caption([]) == "Detected: nothing"


def test_draws_masks_boxes_and_labels_in_distinct_colours():
    img = Image.new("RGB", (400, 200), (128, 128, 128))
    dets = [Detection((40, 60, 120, 160), 0.8, "chair"), Detection((250, 0, 390, 90), 0.5, "stool")]
    masks = np.zeros((2, 200, 400), bool)
    masks[0, 100:150, 60:100] = True
    masks[1, 40:80, 300:350] = True
    out = annotate.draw_detections(img, dets, masks)
    a = np.asarray(out).astype(int)
    assert out.size == img.size and out.mode == "RGB"
    c0, c1 = annotate.COLOURS[0], annotate.COLOURS[1]
    assert c0 != c1
    # Mask interiors are tinted towards their own colour, the rest is untouched.
    tint0 = np.rint(0.55 * 128 + 0.45 * np.array(c0)).astype(int)
    assert (a[125, 80] == tint0).all()
    assert (a[190, 200] == 128).all()
    # Box outlines in the detection's colour.
    assert tuple(a[110, 40]) == c0 and tuple(a[50, 250]) == c1
    # Label of the first box sits above it on its colour; the second (at the top edge)
    # is drawn inside the image instead.
    assert (np.abs(a[45:59, 42:60] - np.array(c0)).sum(-1) == 0).any()
    assert (np.abs(a[0:12, 252:270] - np.array(c1)).sum(-1) == 0).any()


def test_boxes_only_and_label_kept_inside_the_image():
    img = Image.new("RGB", (100, 50), "black")
    out = annotate.draw_detections(img, [Detection((90, 40, 100, 50), 0.9, "a long label")])
    assert out.size == (100, 50)
    assert np.asarray(out).any()  # something was drawn
