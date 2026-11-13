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


def test_turn_drops_higher_order_colour_only_when_turned() -> None:
    v = gaussians([[0, 0, 0]], extra=["f_rest_0"])
    v["f_rest_0"] = 0.7
    assert objects.apply_placement(v, Placement(turn_deg=0))["f_rest_0"][0] == pytest.approx(0.7)
    assert objects.apply_placement(v, Placement(turn_deg=360))["f_rest_0"][0] == pytest.approx(0.7)
    assert objects.apply_placement(v, Placement(turn_deg=45))["f_rest_0"][0] == 0


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
