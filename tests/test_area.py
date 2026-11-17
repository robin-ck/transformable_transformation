"""Pure logic of add-area-layout ("Area to change"; numpy and PIL only)."""

import numpy as np
import pytest
from PIL import Image

from splat_studio.editing import area


def _layer(w, h, box):
    a = np.zeros((h, w, 4), np.uint8)
    x0, y0, x1, y1 = box
    a[y0:y1, x0:x1] = (255, 59, 48, 255)
    return a


def test_painted_layers_become_one_mask_at_the_start_image_size():
    one = _layer(100, 50, (10, 10, 20, 20))
    two = _layer(100, 50, (60, 30, 70, 40))
    faint = np.zeros((50, 100, 4), np.uint8)
    faint[..., 3] = area.ALPHA_ON  # anti-aliased edge: not painted
    hard = area.from_layers([one, two, faint])
    assert hard.shape == (50, 100) and hard.sum() == 200
    big = area.from_layers([Image.fromarray(one)], size=(200, 100))
    assert big.shape == (100, 200) and big[30, 30] and not big[0, 0]
    assert not area.from_layers([], size=(8, 4)).any()
    assert area.from_layers([np.zeros((4, 4, 3), np.uint8)]).shape == (1, 1)  # no alpha


def test_empty_or_tiny_areas_are_refused():
    with pytest.raises(area.AreaError, match="empty or too small"):
        area.check(np.zeros((100, 100), bool))
    tiny = np.zeros((100, 100), bool)
    tiny[0, 0] = True
    with pytest.raises(area.AreaError):
        area.check(tiny)
    ok = np.zeros((100, 100), bool)
    ok[:10, :10] = True
    assert area.check(ok).sum() == 100
    assert area.summary(ok) == {"bounds": [0, 0, 10, 10], "cover": 0.01}


def test_png_round_trip_and_resize():
    hard = np.zeros((40, 60), bool)
    hard[5:15, 20:30] = True
    data = area.to_png(hard)
    assert data[:4] == b"\x89PNG" and len(data) < 1000
    assert (area.from_png(data) == hard).all()
    assert area.from_png(data, size=(120, 80))[20, 50]


def test_crop_has_context_on_every_side_and_stays_in_the_image():
    hard = np.zeros((400, 800), bool)
    hard[180:220, 380:420] = True  # 40 x 40 in the middle
    x0, y0, x1, y1 = area.crop_box(hard)
    floor = round(400 * area.MIN_CROP)  # small area: the crop's floor applies
    assert (x1 - x0, y1 - y0) == (floor, floor)
    assert x0 < 380 and x1 > 420 and y0 < 180 and y1 > 220
    corner = np.zeros((400, 800), bool)
    corner[:100, 700:] = True  # at the top right
    x0, y0, x1, y1 = area.crop_box(corner)
    assert x1 == 800 and y0 == 0 and x1 - x0 == 220 and y1 - y0 == 220
    whole = np.ones((400, 800), bool)
    assert area.crop_box(whole) == (0, 0, 800, 400)


def test_paste_puts_the_edited_crop_back_at_its_place():
    start = Image.new("RGB", (100, 50), "green")
    out = area.paste(start, Image.new("RGB", (512, 512), "red"), (10, 5, 40, 35))
    a = np.asarray(out)
    assert out.size == (100, 50)
    assert (a[20, 20] == [255, 0, 0]).all() and (a[0, 0] == [0, 128, 0]).all()
    assert (a[20, 45] == [0, 128, 0]).all()


def test_prompt_places_the_change_in_the_middle():
    p = area.area_prompt("  a weeping willow. ")
    assert p.startswith("a weeping willow. Make the change in the middle of the image")
    assert "Fill the space" in area.area_prompt("remove the fence")
    assert area.area_prompt("").startswith("Change this place.")


def test_area_info_for_run_json():
    hard = np.zeros((100, 200), bool)
    hard[10:30, 20:60] = True
    info = area.area_info(hard, (0, 0, 100, 60), removal=False)
    assert info["bounds"] == [20, 10, 60, 30] and info["crop"] == [0, 0, 100, 60]
    assert info["cover"] == 0.04 and info["removal"] is False and info["grow_px"] >= 3
