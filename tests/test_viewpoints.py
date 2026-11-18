import json

import pytest

from splat_studio import models
from splat_studio.space import viewpoints as vps
from splat_studio.space.jobs import LocalStore


def cam(x: float = 0.0) -> dict:
    return {"c2w": [[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]],
            "fov_deg": 50.0, "aspect": 1.5}  # fmt: skip


def test_new_scene_has_no_viewpoints_and_default_starters(tmp_path) -> None:
    setup = vps.load_setup(LocalStore(tmp_path), "s1")
    assert setup.viewpoints == [] and setup.editor == models.DEFAULT_EDITOR
    assert [s.label for s in setup.starters] == [
        "nature first", "flood safe", "recreation", "housing",
    ]  # fmt: skip
    assert setup.starter("flood-safe").text.startswith("a river landscape")


def test_viewpoints_add_rename_reorder_remove_round_trip(tmp_path) -> None:
    store = LocalStore(tmp_path)
    setup = vps.load_setup(store, "s1")
    bridge = vps.add_viewpoint(setup, "  bridge ", 1, cam(0), target=[0, 0, 0], now=1.0)
    east = vps.add_viewpoint(setup, "east bank", 1, cam(1), now=2.0)
    auto = vps.add_viewpoint(setup, "", 2, cam(2), now=3.0)
    assert (bridge.name, auto.name, auto.version) == ("bridge", "view 3", 2)
    assert len({bridge.id, east.id, auto.id}) == 3
    vps.rename_viewpoint(setup, east.id, "east  bank, low")
    vps.move_viewpoint(setup, auto.id, -1)
    vps.move_viewpoint(setup, bridge.id, -1)  # already first: stays
    vps.save_setup(store, "s1", setup)

    raw = json.loads((tmp_path / "viewpoints/s1/setup.json").read_text())
    assert set(raw) == {"viewpoints", "starters", "editor", "instructions"}
    back = vps.load_setup(store, "s1")
    assert [v.name for v in back.ordered()] == ["bridge", "view 3", "east bank, low"]
    assert back.get(bridge.id).camera == cam(0) and back.get(bridge.id).target == [0, 0, 0]

    vps.remove_viewpoint(back, east.id)
    assert [v.name for v in back.ordered()] == ["bridge", "view 3"]
    with pytest.raises(vps.ViewpointError):
        vps.remove_viewpoint(back, east.id)
    with pytest.raises(vps.ViewpointError):
        vps.rename_viewpoint(back, bridge.id, " ")
    with pytest.raises(vps.ViewpointError):
        vps.add_viewpoint(back, "x" * 41, 1, cam())
    assert vps.view_prefix("s1", bridge.id) == f"viewpoints/s1/{bridge.id}"


def test_same_view_ignores_fov_and_tiny_differences() -> None:
    a, b = cam(0), {**cam(0), "fov_deg": 60.0, "aspect": 1.0}
    b["c2w"][0][3] = 1e-5
    assert vps.same_view(a, b)
    assert not vps.same_view(a, cam(0.5))
    assert not vps.same_view(a, None)


def test_viewer_target_is_ahead_of_camera_without_stored_target() -> None:
    vp = vps.Viewpoint("v", "x", 1, cam(2), 0, 0.0)
    assert vps.viewer_target(vp) == [2.0, 0.0, 4.0]  # one unit along -z
    vp.target = [1, 2, 3]
    assert vps.viewer_target(vp) == [1, 2, 3]


def test_starters_parse_keep_ids_and_reject_bad_lines() -> None:
    old = vps.default_setup().starters
    text = vps.starters_text(old).replace("new low-rise housing", "terraced houses")
    new = vps.parse_starters(text + "\n\n", old)
    assert [s.id for s in new] == [s.id for s in old]
    assert new[3].text.startswith("terraced houses")
    renamed = vps.parse_starters("Water play: a beach for children", old)
    assert renamed[0].id == "water-play"
    for bad in ["no colon here", "label:", "a: x\nA: y", ":text", f"a: {'x' * 201}"]:
        with pytest.raises(vps.ViewpointError):
            vps.parse_starters(bad, old)
    assert vps.parse_starters("", old) == []


def test_editor_choice_is_checked_against_registry() -> None:
    setup = vps.default_setup()
    assert vps.set_editor(setup, "sdxl-depth").editor == "sdxl-depth"
    with pytest.raises(vps.ViewpointError):
        vps.set_editor(setup, "unknown")
