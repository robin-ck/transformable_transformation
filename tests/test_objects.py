import math
from pathlib import Path

import numpy as np
import pytest

from splat_studio.editing import objects
from splat_studio.editing.objects import Placement
from splat_studio.reconstruction import export

BASE = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
BASE += ["scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def gaussians(xyz, extra: list[str] = ()) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    v = np.zeros(len(xyz), dtype=[(n, "<f4") for n in BASE + list(extra)])
    v["x"], v["y"], v["z"] = xyz.T
    v["opacity"] = 5.0  # sigmoid ~ 0.99
    v["rot_0"] = 1.0
    return v


def xyz(v: np.ndarray) -> np.ndarray:
    return np.stack([v["x"], v["y"], v["z"]], axis=1)


def box(lo, hi, n=11) -> np.ndarray:
    """Points on a grid filling the box, so percentile bounds equal the box."""
    axes = [np.linspace(a, b, n) for a, b in zip(lo, hi, strict=True)]
    return np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, 3)


def test_normalise_puts_base_centre_at_origin_and_size_one() -> None:
    v = gaussians(box([2, 4, 1], [4, 5, 9]))  # 2 x 1 x 8, base at z=1
    n = objects.normalise_object(v)
    lo, hi = objects.bounds(n)
    assert np.allclose(lo, [-0.125, -0.0625, 0], atol=0.01)
    assert np.allclose(hi, [0.125, 0.0625, 1], atol=0.01)
    assert n["scale_0"][0] == pytest.approx(-math.log(8), abs=1e-5)


def test_normalise_ignores_faint_haze() -> None:
    v = gaussians(np.vstack([box([0, 0, 0], [1, 1, 1]), [[0, 0, 50]] * 30]))
    v["opacity"][-30:] = -10  # nearly invisible
    n = objects.normalise_object(v)
    assert objects.bounds(n)[1][2] == pytest.approx(1, abs=0.01)


def test_normalise_rejects_flat_object() -> None:
    with pytest.raises(ValueError):
        objects.normalise_object(gaussians([[1, 1, 1]] * 20))


def test_placement_moves_turns_and_scales() -> None:
    v = gaussians([[1, 0, 0], [0, 0, 1]])
    out = objects.apply_placement(v, Placement(x=10, y=20, z=30, turn_deg=90, size=2))
    assert np.allclose(xyz(out), [[10, 22, 30], [10, 20, 32]], atol=1e-5)
    assert out["scale_0"][0] == pytest.approx(math.log(2), abs=1e-5)
    q = [out[f"rot_{i}"][0] for i in range(4)]
    assert np.allclose(q, [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)], atol=1e-6)


def test_placement_rotation_composes_with_existing_rotation() -> None:
    v = gaussians([[0, 0, 0]])
    v["rot_0"], v["rot_3"] = math.cos(math.pi / 4), math.sin(math.pi / 4)  # 90 deg about z
    out = objects.apply_placement(v, Placement(turn_deg=90))
    q = [out[f"rot_{i}"][0] for i in range(4)]
    assert np.allclose(q, [0, 0, 0, 1], atol=1e-6)  # 180 deg about z


def test_placement_viewer_and_merge_agree() -> None:
    p = Placement(x=1, y=2, z=3, turn_deg=30, size=0.5)
    cfg = p.to_viewer()
    qx, qy, qz, w = cfg["quaternion"]
    assert (w, qx, qy, qz) == p.quaternion_wxyz()
    assert cfg["position"] == [1, 2, 3] and cfg["scale"] == [0.5] * 3


def test_tilt_about_x_and_y() -> None:
    v = gaussians([[0, 0, 1]])
    assert np.allclose(xyz(objects.apply_placement(v, Placement(tilt_x_deg=90))), [[0, -1, 0]])
    assert np.allclose(xyz(objects.apply_placement(v, Placement(tilt_y_deg=90))), [[1, 0, 0]])


def test_tilt_is_applied_before_turn() -> None:
    # Tilt top towards +x, then turn 90 deg: the top ends up towards +y.
    v = gaussians([[0, 0, 1]])
    out = objects.apply_placement(v, Placement(turn_deg=90, tilt_y_deg=90))
    assert np.allclose(xyz(out), [[0, 1, 0]], atol=1e-6)


def test_tilted_placement_rotates_gaussians_like_points() -> None:
    p = Placement(turn_deg=30, tilt_x_deg=20, tilt_y_deg=-45)
    v = gaussians([[0.3, -0.2, 0.7]])
    out = objects.apply_placement(v, p)
    q = np.array([out[f"rot_{i}"][0] for i in range(4)])  # existing rotation was identity
    assert np.allclose(q, p.quaternion_wxyz())
    assert np.linalg.norm(xyz(out)[0]) == pytest.approx(np.linalg.norm([0.3, -0.2, 0.7]))
    cfg = p.to_viewer()
    assert (cfg["tiltX"], cfg["tiltY"]) == (20, -45)


def test_turn_drops_higher_order_colour_only_when_turned() -> None:
    v = gaussians([[0, 0, 0]], extra=["f_rest_0"])
    v["f_rest_0"] = 0.7
    assert objects.apply_placement(v, Placement(turn_deg=0))["f_rest_0"][0] == pytest.approx(0.7)
    assert objects.apply_placement(v, Placement(turn_deg=360))["f_rest_0"][0] == pytest.approx(0.7)
    assert objects.apply_placement(v, Placement(turn_deg=45))["f_rest_0"][0] == 0
    assert objects.apply_placement(v, Placement(tilt_x_deg=10))["f_rest_0"][0] == 0


def test_placement_rejects_zero_size() -> None:
    with pytest.raises(ValueError):
        objects.apply_placement(gaussians([[0, 0, 0]]), Placement(size=0))


def test_merge_keeps_scene_layout_and_fills_missing_fields() -> None:
    scene = gaussians([[0, 0, 0], [1, 1, 1]], extra=["nx", "f_rest_0"])
    scene["f_rest_0"] = 0.3
    obj = gaussians([[5, 5, 5]], extra=["only_in_object"])
    merged = objects.merge(scene, obj)
    assert merged.dtype == scene.dtype and len(merged) == 3
    assert list(merged["x"]) == [0, 1, 5]
    assert merged["f_rest_0"][2] == 0 and merged["f_rest_0"][0] == pytest.approx(0.3)


def test_merge_files_roundtrip(tmp_path: Path) -> None:
    scene = gaussians(box([0, 0, 0], [4, 4, 1], n=5), extra=["f_rest_0"])
    obj = objects.normalise_object(gaussians(box([0, 0, 0], [1, 1, 2], n=5)))
    export.write_ply(tmp_path / "scene.ply", scene)
    export.write_ply(tmp_path / "obj.ply", obj)
    n = objects.merge_files(
        tmp_path / "scene.ply",
        tmp_path / "obj.ply",
        Placement(x=2, y=2, z=1, size=0.5),
        tmp_path / "out.ply",
        tmp_path / "out.splat",
    )
    out = export.read_ply(tmp_path / "out.ply")
    assert n == len(out) == len(scene) + len(obj)
    assert out.dtype.names == scene.dtype.names
    placed = out[len(scene) :]
    assert placed["z"].min() == pytest.approx(1, abs=1e-5)  # base on the floor
    assert placed["z"].max() == pytest.approx(1.5, abs=1e-5)  # 0.5 tall
    assert (tmp_path / "out.splat").stat().st_size == 32 * n


def test_write_ply_roundtrip(tmp_path: Path) -> None:
    v = gaussians([[1, 2, 3], [4, 5, 6]], extra=["nx"])
    export.write_ply(tmp_path / "a.ply", v)
    back = export.read_ply(tmp_path / "a.ply")
    assert back.dtype.names == v.dtype.names
    assert np.array_equal(xyz(back), xyz(v))


def test_default_placement_from_scene() -> None:
    pts = np.vstack([box([-1, -1, 0], [1, 1, 2]), [[1e6, 0, 0]]])
    p, radius = objects.default_placement(pts)
    assert (p.x, p.y, p.z) == pytest.approx((0, 0, 1), abs=0.01)
    assert p.size == pytest.approx(radius / 4, abs=0.001) and radius < 3


# --- placing in the Space: new version only on confirm --------------------------


def _store_with_scene(tmp_path: Path):
    from splat_studio.space import jobs, versions

    store = jobs.LocalStore(tmp_path / "store")
    scene = gaussians(box([0, 0, 0], [4, 4, 1], n=5))
    v1 = tmp_path / "store" / "scenes" / "s1" / "v1"
    v1.mkdir(parents=True)
    export.write_ply(v1 / "scene.ply", scene)
    (v1 / "scene.splat").write_bytes(export.to_splat_bytes(scene))
    obj_dir = tmp_path / "store" / "objects" / "o1"
    obj_dir.mkdir(parents=True)
    export.write_ply(
        obj_dir / "object.ply", objects.normalise_object(gaussians(box([0] * 3, [1] * 3)))
    )
    job = jobs.Job(
        "o1", "object", "oak tree", 0.0, status="done", scene_path="objects/o1",
        run={"model": "flux + trellis"}, params={"prompt": "a mature oak tree", "seed": 3},
    )  # fmt: skip
    return store, versions.list_versions(store, "s1")[0], job, len(scene)


def test_start_placement_fits_scene(tmp_path: Path) -> None:
    from splat_studio.space.add_object import start_placement

    store, v1, _job, _n = _store_with_scene(tmp_path)
    placement, ranges = start_placement(store, v1)
    assert ranges.x[0] < placement.x < ranges.x[1]
    assert ranges.size[0] < placement.size < ranges.size[1]


def test_place_object_adds_version_with_log(tmp_path: Path) -> None:
    from splat_studio.space import versions
    from splat_studio.space.add_object import place_object

    store, v1, job, n_scene = _store_with_scene(tmp_path)
    v2 = place_object(store, v1, job, Placement(x=2, y=2, z=1, size=0.5), now=100.0)
    assert v2.number == 2 and v2.mode == "add_object" and v2.parent == 1
    listed = versions.list_versions(store, "s1")
    assert [v.number for v in listed] == [2, 1]
    assert listed[0].prompt == "a mature oak tree" and listed[0].model == "flux + trellis"
    meta = store.read_json("scenes/s1/v2/version.json")
    assert meta["object_job"] == "o1" and meta["placement"]["size"] == 0.5
    merged = export.read_ply(store.download("scenes/s1/v2/scene.ply"))
    assert len(merged) == n_scene + 11**3
    # v1 is untouched
    assert len(export.read_ply(store.download("scenes/s1/v1/scene.ply"))) == n_scene


def test_place_object_refuses_unfinished_job(tmp_path: Path) -> None:
    from dataclasses import replace

    from splat_studio.space.add_object import PlaceError, place_object

    store, v1, job, _n = _store_with_scene(tmp_path)
    with pytest.raises(PlaceError):
        place_object(store, v1, replace(job, status="running"), Placement())
    assert store.list_dirs("scenes/s1") == ["v1"]


# --- hideable objects (redesign-space-gui 4.1, 4.2) -----------------------------


def test_splat_segments_sort_within_each_segment() -> None:
    v = gaussians([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    v["scale_0"] = [0.0, 1.0, 0.0, 2.0]  # larger last within each pair
    flat = np.frombuffer(export.to_splat_bytes(v), dtype="<f4").reshape(-1, 8)
    seg = np.frombuffer(export.to_splat_bytes(v, [2, 2]), dtype="<f4").reshape(-1, 8)
    assert list(flat[:, 0]) == [3, 1, 0, 2]  # one sort over all
    assert list(seg[:, 0]) == [1, 0, 3, 2]  # each pair sorted on its own, kept in place
    with pytest.raises(ValueError):
        export.to_splat_bytes(v, [3, 2])


def test_ply_vertex_count_reads_header(tmp_path: Path) -> None:
    export.write_ply(tmp_path / "a.ply", gaussians(box([0] * 3, [1] * 3, n=3)))
    assert export.ply_vertex_count(tmp_path / "a.ply") == 27


def test_two_added_objects_keep_their_ranges(tmp_path: Path) -> None:
    from splat_studio.space import versions
    from splat_studio.space.add_object import place_object

    store, v1, job, n_scene = _store_with_scene(tmp_path)
    n_obj = 11**3
    v2 = place_object(store, v1, job, Placement(x=1, y=1, z=1, size=0.5), now=100.0)
    v3 = place_object(store, v2, job, Placement(x=3, y=3, z=1, size=0.5), now=200.0)
    listed = {v.number: v for v in versions.list_versions(store, "s1")}
    assert listed[1].objects == ()
    assert [o["id"] for o in listed[2].objects] == ["o1"]
    o1, o2 = listed[3].objects
    assert (o1["id"], o1["start"], o1["count"]) == ("o1", n_scene, n_obj)
    assert (o2["id"], o2["start"], o2["count"]) == ("o2", n_scene + n_obj, n_obj)
    assert o2["prompt"] == "a mature oak tree" and o2["object_job"] == "o1"
    assert v3.objects == listed[3].objects
    # Each object's byte range in the .splat holds exactly that object's Gaussians.
    splat = np.frombuffer(store.download("scenes/s1/v3/scene.splat").read_bytes(), "<f4")
    pos = splat.reshape(-1, 8)[:, :3]
    first = pos[o1["start"] : o1["start"] + o1["count"]]
    second = pos[o2["start"] : o2["start"] + o2["count"]]
    assert np.allclose(first[:, :2].mean(axis=0), [1, 1], atol=0.01)
    assert np.allclose(second[:, :2].mean(axis=0), [3, 3], atol=0.01)
    assert pos[: o1["start"], 2].min() >= 0 and pos[: o1["start"], 2].max() <= 1.0


def test_filtered_splat_drops_hidden_objects_and_is_cached(tmp_path: Path) -> None:
    from splat_studio.space import versions
    from splat_studio.space.add_object import place_object

    store, v1, job, n_scene = _store_with_scene(tmp_path)
    v2 = place_object(store, v1, job, Placement(x=1, y=1, z=1, size=0.5))
    v3 = place_object(store, v2, job, Placement(x=3, y=3, z=1, size=0.5))
    full = store.download(f"{v3.path}/scene.splat")
    assert versions.filtered_splat(store, v3, []) == full
    assert versions.filtered_splat(store, v3, ["o9"]) == full  # unknown id: nothing hidden
    only_o2 = versions.filtered_splat(store, v3, {"o1"})
    data = full.read_bytes()
    o1, o2 = v3.objects
    assert only_o2.read_bytes() == data[: o1["start"] * 32] + data[o2["start"] * 32 :]
    neither = versions.filtered_splat(store, v3, ["o2", "o1"])
    assert neither.read_bytes() == data[: n_scene * 32]
    assert neither.name == "s1-v3-o1-o2.splat" and neither.parent.name == "hidden"
    mtime = neither.stat().st_mtime_ns
    assert versions.filtered_splat(store, v3, ["o1", "o2"]).stat().st_mtime_ns == mtime
    # Downloads stay complete.
    assert versions.download_file(store, v3, "splat").read_bytes() == data


def test_floor_height_ignores_floaters_below_ground() -> None:
    from splat_studio.space.add_object import floor_height

    ground = box([0, 0, 2], [4, 4, 2.0], n=20)
    pts = np.vstack([ground, box([1, 1, 2], [2, 2, 5], n=10), [[0, 0, -50]] * 5])
    assert floor_height(pts) == pytest.approx(2.0, abs=0.01)
    assert floor_height(np.zeros((0, 3))) == 0.0


def test_start_placement_reports_floor(tmp_path: Path) -> None:
    from splat_studio.space.add_object import start_placement

    store, v1, _job, _n = _store_with_scene(tmp_path)
    _placement, ranges = start_placement(store, v1)
    assert ranges.floor == pytest.approx(0.0, abs=0.3)


# --- move a placed object (objects table) -----------------------------------------


def test_move_object_keeps_the_others_in_place(tmp_path: Path) -> None:
    from splat_studio.space import versions
    from splat_studio.space.add_object import move_object, object_placement, place_object

    store, v1, job, n_scene = _store_with_scene(tmp_path)
    n_obj = 11**3
    v2 = place_object(store, v1, job, Placement(x=1, y=1, z=1, size=0.5))
    v3 = place_object(store, v2, job, Placement(x=3, y=3, z=1, size=0.5))
    assert object_placement(v3.objects[0]) == Placement(x=1, y=1, z=1, size=0.5)
    v4 = move_object(store, v3, "o1", Placement(x=2, y=0, z=1, size=0.25), now=300.0)
    assert v4.mode == "move_object" and v4.parent == 3 and v4.prompt == "move a mature oak tree"
    o2, o1 = versions.list_versions(store, "s1")[0].objects  # the moved one goes last
    assert (o2["id"], o2["start"], o2["count"]) == ("o2", n_scene, n_obj)
    assert (o1["id"], o1["start"], o1["count"]) == ("o1", n_scene + n_obj, n_obj)
    assert object_placement(o1) == Placement(x=2, y=0, z=1, size=0.25)
    pos = np.frombuffer(store.download(f"{v4.path}/scene.splat").read_bytes(), "<f4")
    pos = pos.reshape(-1, 8)[:, :3]
    assert len(pos) == n_scene + 2 * n_obj
    assert np.allclose(pos[n_scene : n_scene + n_obj, :2].mean(axis=0), [3, 3], atol=0.01)
    assert np.allclose(pos[n_scene + n_obj :, :2].mean(axis=0), [2, 0], atol=0.01)
    # v3 is untouched
    assert len(export.read_ply(store.download(f"{v3.path}/scene.ply"))) == n_scene + 2 * n_obj


def test_old_object_entries_get_placement_from_their_version(tmp_path: Path) -> None:
    from splat_studio.space import versions
    from splat_studio.space.add_object import object_placement, place_object

    store, v1, job, _n = _store_with_scene(tmp_path)
    v2 = place_object(store, v1, job, Placement(x=1, y=1, z=1, size=0.5))
    v3 = place_object(store, v2, job, Placement(x=3, y=3, z=1, size=0.5))
    for v in (v2, v3):  # as written before entries kept their placement
        meta = store.read_json(f"{v.path}/version.json")
        meta["objects"] = [
            {k: x for k, x in o.items() if k != "placement"} for o in meta["objects"]
        ]
        store.write_json(f"{v.path}/version.json", meta)
    o1, o2 = versions.list_versions(store, "s1")[0].objects
    assert object_placement(o1) == Placement(x=1, y=1, z=1, size=0.5)
    assert object_placement(o2) == Placement(x=3, y=3, z=1, size=0.5)
    assert object_placement({"id": "o9"}) is None


def test_object_and_edit_rows() -> None:
    from splat_studio.space import jobs, versions
    from splat_studio.space.add_object import edit_rows, object_rows

    v = versions.Version("s1", 3, objects=({"id": "o1", "prompt": "oak"}, {"id": "o2"}))
    js = [
        jobs.Job("a", "object", "oak", 1.0, status="done", scene_path="objects/a",
                 params={"scene_id": "s1", "seed": 2}),
        jobs.Job("b", "object", "pond", 2.0, status="running", params={"scene_id": "s1"}),
        jobs.Job("c", "object", "other", 3.0, status="done", params={"scene_id": "s2"}),
    ]  # fmt: skip
    rows, keys = object_rows(v, ["o2"], js, "s1", moving="o1")
    assert rows == [
        ["1. oak", "✋ moving", ""],
        ["2. object", "hidden", "👁 Show"],
        ["pond (seed 0)", "⚙️ generating", ""],
        ["oak (seed 2)", "generated", "➕ Add"],
    ]
    assert keys == [["placed", "o1"], ["placed", "o2"], ["job", "b"], ["library", "a"]]
    vs = [
        versions.Version("s1", 5, mode="edit_scene", prompt="snow", parent=1),
        versions.Version("s1", 4, mode="add_object", prompt="bench", parent=3),
        versions.Version("s1", 3, mode="edit_scene", prompt="greener", parent=2),
        versions.Version("s1", 2, mode="add_object", prompt="oak", parent=1),
        versions.Version("s1", 1),
    ]
    edits = [
        jobs.Job(
            "e", "edit", "rain", 5.0, status="failed", params={"prompt": "rain", "version": 4}
        ),
        jobs.Job("f", "edit", "fog", 6.0, status="running", params={"prompt": "fog", "version": 5}),
    ]
    rows, numbers = edit_rows(vs, edits, shown=4)  # line v4 <- v3 <- v2 <- v1: not v5
    assert rows == [["from v4", "rain", "❌ failed", ""], ["v3", "greener", "", "Show"]]
    assert numbers == [None, 3]
    assert edit_rows(vs, [], shown=3)[0] == [["v3", "greener", "● shown", ""]]
    assert edit_rows(vs, edits, shown=5)[0] == [
        ["from v5", "fog", "⚙️ running", ""],
        ["v5", "snow", "● shown", ""],
    ]
    assert edit_rows(vs, edits, shown=None) == ([], [])
