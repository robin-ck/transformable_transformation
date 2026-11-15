"""Edit scene (task 4.4): the GPU-free parts and storing the result as a version."""

from pathlib import Path

import numpy as np
import pytest
from test_objects import _store_with_scene, box, gaussians
from test_space import Clock, FakeClient

from splat_studio.editing import preview, scene_edit
from splat_studio.editing.objects import Placement
from splat_studio.editing.scene_edit import EditError, Region
from splat_studio.reconstruction import export
from splat_studio.space import edit_scene, jobs, versions

# --- region ---------------------------------------------------------------------


def test_region_parsing() -> None:
    r = Region.from_dict({"center": [1, 2, "3"], "radius": 0.5})
    assert r == Region((1.0, 2.0, 3.0), 0.5)
    assert Region.from_dict(r.to_dict()) == r
    assert Region.from_dict(None) is None and Region.from_dict({}) is None
    for bad in ({"center": [1, 2], "radius": 1}, {"center": [0, 0, 0], "radius": 0},
                {"center": [0, 0, float("nan")], "radius": 1}, {"radius": 1}):  # fmt: skip
        with pytest.raises(EditError):
            Region.from_dict(bad)


def test_region_mask_and_empty_region() -> None:
    means = box([0, 0, 0], [4, 4, 0], n=21)  # a 4 x 4 floor, 441 Gaussians
    assert scene_edit.region_mask(means, None).all()
    mask = scene_edit.region_mask(means, Region((2.0, 2.0, 0.0), 1.0))
    assert 0 < mask.sum() < len(means)
    assert np.all(np.linalg.norm(means[mask] - [2, 2, 0], axis=1) <= 1.0)
    with pytest.raises(EditError, match="almost nothing"):
        scene_edit.check_region(means, Region((50.0, 50.0, 0.0), 1.0))


def test_parse_region_from_the_space() -> None:
    region = {"center": [1, 1, 0], "radius": 0.3}
    assert edit_scene.parse_region(False, region) is None
    assert edit_scene.parse_region(True, region) == {"center": (1.0, 1.0, 0.0), "radius": 0.3}
    with pytest.raises(EditError, match="mark the region"):
        edit_scene.parse_region(True, None)


# --- cameras ----------------------------------------------------------------------


def test_look_at_points_the_camera_at_the_target() -> None:
    cam = scene_edit.look_at((5, 1, 2), (1, 1, 0))
    w2c = preview.viewmat(cam)
    target = w2c @ np.array([1, 1, 0, 1.0])
    assert target[2] > 0  # in front (OpenCV: +z forward)
    assert np.allclose(target[:2], 0, atol=1e-9)  # on the optical axis
    above = w2c @ np.array([1, 1, 1, 1.0])
    assert above[1] < 0  # world up is up on screen (OpenCV: -y)
    straight_down = scene_edit.look_at((0, 0, 5), (0, 0, 0))
    assert np.isfinite(np.asarray(straight_down.c2w)).all()


def test_orbit_cameras_ring_the_scene_or_region() -> None:
    pts = box([-1, -1, 0], [1, 1, 1], n=11)
    cams = scene_edit.orbit_cameras(pts, n=12)
    centre, radius = scene_edit.scene_centre_radius(pts)
    positions = np.array([np.asarray(c.c2w)[:3, 3] for c in cams])
    dist = np.linalg.norm(positions - centre, axis=1)
    assert len(cams) == 12 and np.allclose(dist, scene_edit.SCENE_DISTANCE * radius)
    assert (positions[:, 2] > centre[2]).all()  # all from above the centre
    assert len({round(float(np.arctan2(p[1], p[0])), 3) for p in positions - centre}) == 12
    for c in cams:
        ahead = preview.viewmat(c) @ np.append(centre, 1.0)
        assert ahead[2] > 0 and np.allclose(ahead[:2], 0, atol=1e-6)
    region = Region((0.5, 0.5, 0.2), 0.3)
    near = scene_edit.orbit_cameras(pts, region, n=6)
    d = [np.linalg.norm(np.asarray(c.c2w)[:3, 3] - region.center) for c in near]
    assert np.allclose(d, scene_edit.REGION_DISTANCE * 0.3)


def test_strength_and_guidance() -> None:
    rng = np.random.default_rng(0)
    first = [scene_edit.strength_for(0, rng) for _ in range(50)]
    later = [scene_edit.strength_for(2, rng) for _ in range(50)]
    assert min(first) >= 0.85 and max(first) <= 0.98
    assert min(later) >= 0.5 and max(later) <= 0.8
    text, image, uncond = np.array(3.0), np.array(2.0), np.array(1.0)
    # uncond + 7.5 * (text - image) + 1.5 * (image - uncond)
    assert scene_edit.combine_guidance(text, image, uncond) == pytest.approx(1 + 7.5 + 1.5)


# --- writing the fine-tuned values --------------------------------------------------


def test_write_params_changes_only_the_region() -> None:
    v = gaussians([[0, 0, 0], [1, 0, 0], [2, 0, 0]], extra=["f_rest_0"])
    v["f_rest_0"] = 0.4
    n = len(v)
    new = scene_edit.write_params(
        v,
        means=np.full((n, 3), 9.0),
        log_scales=np.full((n, 3), -2.0),
        quats_wxyz=np.tile([0.0, 1.0, 0.0, 0.0], (n, 1)),
        opacity_logits=np.full(n, -1.0),
        sh0=np.full((n, 3), 0.25),
        mask=np.array([False, True, False]),
    )
    assert new.dtype == v.dtype
    assert new[0] == v[0] and new[2] == v[2]  # untouched rows are identical
    assert (new["x"][1], new["scale_0"][1], new["rot_1"][1]) == (9.0, -2.0, 1.0)
    assert (new["opacity"][1], new["f_dc_2"][1], new["f_rest_0"][1]) == (-1.0, 0.25, 0.0)
    with pytest.raises(ValueError):
        scene_edit.write_params(v, *[np.zeros((n, 3))] * 3, np.zeros(n), np.zeros((n, 3)), [True])


# --- job and version ---------------------------------------------------------------


class EditClient(FakeClient):
    def submit_edit(self, job_id, scene_path, prompt, region=None, seed=0) -> str:
        self.submitted.append((job_id, scene_path, prompt, region, seed))
        return f"fc-{job_id}"


def _finished_edit(tmp_path: Path, region=None):
    """Scene s1 with v1 and v2 (an added object), and a finished edit of v2."""
    from splat_studio.space.add_object import place_object

    store, v1, obj_job, _n = _store_with_scene(tmp_path)
    v2 = place_object(store, v1, obj_job, Placement(x=2, y=2, z=1, size=0.5), now=100.0)
    client = EditClient()
    mgr = jobs.JobManager(store, client, Clock(), prefix="edits", timeout_s=2400)
    job = mgr.create_edit("make it autumn", v2.path, "s1", 2, region=region, seed=4)
    assert client.submitted == [(job.job_id, v2.path, "make it autumn", region, 4)]
    assert job.kind == "edit" and job.params["model"].startswith("timbrooks/instruct-pix2pix@")
    out = tmp_path / "store" / "edits" / job.job_id
    edited = export.read_ply(store.download(f"{v2.path}/scene.ply"))
    edited["f_dc_0"] = 0.9
    export.write_ply(out / "scene.ply", edited)
    (out / "scene.splat").write_bytes(export.to_splat_bytes(edited))
    client.results[job.call_id] = {"status": "done", "scene_path": f"edits/{job.job_id}",
                                   "model": "timbrooks/instruct-pix2pix@31519b5"}  # fmt: skip
    return store, mgr, mgr.refresh(job)


def test_finished_edit_becomes_one_new_version(tmp_path: Path) -> None:
    region = {"center": [2.0, 2.0, 0.5], "radius": 1.0}
    store, mgr, job = _finished_edit(tmp_path, region)
    v3 = edit_scene.finish_edit(store, mgr, job, now=300.0)
    assert (v3.number, v3.mode, v3.parent, v3.prompt) == (3, "edit_scene", 2, "make it autumn")
    assert v3.model == "timbrooks/instruct-pix2pix@31519b5"
    assert v3.objects == ()  # objects became part of the scene
    meta = store.read_json("scenes/s1/v3/version.json")
    assert meta["edit_job"] == job.job_id and meta["region"] == region and meta["seed"] == 4
    assert "objects" not in meta
    # Asking again (e.g. the next timer tick, or after a restart) gives the same version.
    again = edit_scene.finish_edit(store, mgr, mgr.get(job.job_id))
    assert again.number == 3
    assert [v.number for v in versions.list_versions(store, "s1")] == [3, 2, 1]
    stored = export.read_ply(store.download("scenes/s1/v3/scene.ply"))
    assert np.allclose(stored["f_dc_0"], 0.9)


def test_finish_edit_refuses_unfinished_or_missing_files(tmp_path: Path) -> None:
    from dataclasses import replace

    store, mgr, job = _finished_edit(tmp_path)
    with pytest.raises(edit_scene.FinishError, match="not finished"):
        edit_scene.finish_edit(store, mgr, replace(job, status="running"))
    (tmp_path / "store" / job.scene_path / "scene.splat").unlink()
    with pytest.raises(edit_scene.FinishError, match="missing"):
        edit_scene.finish_edit(store, mgr, job)
    assert [v.number for v in versions.list_versions(store, "s1")] == [2, 1]


def test_edit_job_errors_use_edit_wording(tmp_path: Path) -> None:
    store, client = jobs.LocalStore(tmp_path), EditClient()
    mgr = jobs.JobManager(store, client, Clock(), prefix="edits", timeout_s=2400)
    job = mgr.create_edit("x", "scenes/s/v1", "s", 1)
    client.results[job.call_id] = RuntimeError("worker lost")
    assert mgr.refresh(job).error["code"] == "edit_failed"
    slow = mgr.create_edit("x", "scenes/s/v1", "s", 1)
    client.results[slow.call_id] = type("FunctionTimeoutError", (Exception,), {})()
    assert "40 minutes" in mgr.refresh(slow).error["message"]


def test_viewer_gets_floor_and_scene_radius(tmp_path: Path) -> None:
    import html
    import json
    import re

    from splat_studio.space import viewer

    store = jobs.LocalStore(tmp_path)
    scene = gaussians(box([0, 0, 2], [4, 4, 3], n=9))
    (tmp_path / "scenes/s/v1").mkdir(parents=True)
    (tmp_path / "scenes/s/v1/scene.splat").write_bytes(export.to_splat_bytes(scene))
    page = html.unescape(viewer.scene_html(store, "scenes/s/v1"))
    cfg = json.loads(re.search(r"const cfg = (.*?);\n", page).group(1))
    assert cfg["floor"] == pytest.approx(2.0, abs=0.01)
    assert cfg["sceneRadius"] == pytest.approx(
        scene_edit.scene_centre_radius(
            viewer.splat_positions(tmp_path / "scenes/s/v1/scene.splat")
        )[1],
        abs=1e-3,
    )
