"""Pure mask logic of add-object-masking (numpy and PIL only)."""

import numpy as np
import pytest
from PIL import Image

from splat_studio.editing import masking


def test_query_becomes_detector_phrases():
    assert masking.object_phrases(None) == []
    assert masking.object_phrases("  ") == []
    assert masking.object_phrases("Chairs,  the Table.") == ["chairs", "the table"]
    assert masking.detector_text("chairs, table") == "chairs. table."
    with pytest.raises(masking.MaskError, match="limit"):
        masking.object_phrases("x" * (masking.MAX_QUERY + 1))
    with pytest.raises(masking.MaskError, match="at most"):
        masking.object_phrases("a, b, c, d, e, f")


def test_removal_reads_only_the_verb():
    assert masking.is_removal("Remove the chairs")
    assert masking.is_removal("the square without parked cars")
    assert masking.is_removal("take away the fence")
    assert not masking.is_removal("make the chairs red")
    assert not masking.is_removal("removed")  # whole words only


def test_grow_matches_a_square_max_filter():
    m = np.zeros((9, 11), bool)
    m[4, 5] = True
    g = masking.grow(m, 2)
    assert g.shape == m.shape and g.sum() == 25 and g[2:7, 3:8].all()
    edge = np.zeros((5, 5), bool)
    edge[0, 0] = True
    assert masking.grow(edge, 1).sum() == 4  # clipped at the border
    assert (masking.grow(m, 0) == m).all()


def test_soft_mask_covers_the_object_and_is_zero_far_away():
    hard = np.zeros((768, 1408), bool)
    hard[300:400, 600:700] = True
    soft = masking.soft_mask(hard)
    grow_px, feather_px = masking.mask_sizes((1408, 768), removal=False)
    assert (grow_px, feather_px) == (21, 7)
    assert soft.dtype == np.float32 and soft.shape == hard.shape
    assert (soft[hard] == 1).all()  # the object itself is fully replaced
    assert soft[350, 700 + grow_px // 2] > 0.9  # margin around it too
    assert soft[0, 0] == 0 and soft[350, 700 + 2 * grow_px] == 0
    wide = masking.soft_mask(hard, removal=True)
    assert masking.mask_sizes((1408, 768), removal=True) == (56, 19)
    assert (wide > 0).sum() > (soft > 0).sum()


def test_composite_keeps_the_start_image_outside_the_mask():
    start = Image.fromarray(np.random.default_rng(0).integers(0, 256, (60, 80, 3), np.uint8))
    edited = Image.new("RGB", (40, 30), (255, 0, 0))  # Kontext size differs: resized
    hard = np.zeros((60, 80), bool)
    hard[20:40, 30:50] = True
    m = masking.soft_mask(hard)
    out = masking.composite(start, edited, m)
    a, s = np.asarray(out), np.asarray(start)
    assert out.size == start.size
    assert (a[m == 0] == s[m == 0]).all()
    assert (a[m == 1] == [255, 0, 0]).all()
    # mask.png round trip gives the same blend
    stored = np.asarray(masking.mask_image(m), dtype=np.float32) / 255
    assert np.array_equal(stored, m)


def test_changed_area_dims_the_rest():
    img = Image.new("RGB", (4, 2), (200, 200, 200))
    mask = Image.fromarray(np.array([[255, 255, 0, 0]] * 2, np.uint8))
    out = np.asarray(masking.changed_area(img, mask))
    assert (out[:, :2] == 200).all() and (out[:, 2:] == 70).all()
    big = masking.changed_area(Image.new("RGB", (8, 4), "white"), mask)  # mask resized
    assert big.size == (8, 4)


def test_detections_are_clipped_sorted_and_recorded():
    dets = masking.keep_detections(
        [[-5, 10, 30, 40], [50, 50, 50.4, 60], [10, 10, 20, 20]],
        [0.4, 0.9, 0.8],
        ["chairs", "chairs", "table"],
        (64, 48),
    )
    assert [d.label for d in dets] == ["table", "chairs"]  # empty box dropped
    assert dets[1].box == (0, 10, 30, 40)
    hard = np.zeros((48, 64), bool)
    hard[:12] = True
    info = masking.mask_info("chairs, table", dets, hard, False, ["a@1", "b@2"])
    assert info["detections"] == 2 and info["boxes"][0] == [10, 10, 20, 20]
    assert info["scores"] == [0.8, 0.4] and info["area"] == 0.25
    assert info["models"] == ["a@1", "b@2"] and info["removal"] is False


def test_masker_builds_without_loading_models():
    from splat_studio import models

    m = masking.ObjectMasker(models.MASK_DETECTOR, models.MASK_SEGMENTER)
    assert m.detector is None and m.sam is None
    assert m.labels == [models.MASK_DETECTOR.label, models.MASK_SEGMENTER.label]
