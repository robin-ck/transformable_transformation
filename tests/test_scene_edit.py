import json
from pathlib import Path

import numpy as np
import pytest

from splat_studio.editing import preview, scene_edit
from splat_studio.editing.scene_edit import Region, RegionError
from splat_studio.reconstruction.export import read_ply, write_ply
from splat_studio.space import jobs, prompt_log, viewer
from splat_studio.space.versions import Version


def cam_at(eye, target=(0, 0, 0), fov=50.0, aspect=1.5) -> preview.ViewCamera:
    c2w = scene_edit.look_at_c2w(eye, target)
    return preview.ViewCamera(tuple(tuple(r) for r in c2w), fov, aspect)


def project(cam, p, w=640, h=448):
    pc = preview.viewmat(cam) @ np.append(np.asarray(p, dtype=np.float64), 1.0)
    uv = preview.intrinsics(cam, w, h) @ pc[:3]
    return uv[0] / uv[2], uv[1] / uv[2], pc[2]


def test_pivot_is_in_front_of_the_camera() -> None:
    cam = cam_at((0, -5, 0))
    np.testing.assert_allclose(scene_edit.pivot_point(cam, 5.0), [0, 0, 0], atol=1e-9)


def test_orbit_cameras_look_at_pivot_within_limits() -> None:
    anchor = cam_at((0, -4, 1))
    cams = scene_edit.orbit_cameras(anchor, (0, 0, 0), 25, yaw_deg=30, pitch_deg=10, seed=3)
    assert len(cams) == 25 and cams[0] == anchor
    r0 = np.linalg.norm([0, -4, 1])
    for c in cams[1:]:
        u, v, z = project(c, (0, 0, 0))
        assert (u, v) == pytest.approx((320, 224), abs=1e-6)  # pivot at the image centre
        eye = np.asarray(c.c2w)[:3, 3]
        assert 0.85 * r0 - 1e-9 <= np.linalg.norm(eye) <= 1.15 * r0 + 1e-9
        yaw = np.degrees(np.arctan2(eye[0], -eye[1]))
        assert abs(yaw) <= 30 + 1e-6
        assert (c.fov_deg, c.aspect) == (anchor.fov_deg, anchor.aspect)
    assert len({np.asarray(c.c2w).tobytes() for c in cams}) == 25  # all different


def test_orbit_cameras_are_repeatable() -> None:
    a = scene_edit.orbit_cameras(cam_at((0, -4, 1)), (0, 0, 0), 5, seed=1)
    b = scene_edit.orbit_cameras(cam_at((0, -4, 1)), (0, 0, 0), 5, seed=1)
    assert a == b


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (None, None),
        (json.dumps({"region": None}), None),
        (json.dumps({"region": [0.6, 0.8, 0.2, 0.1]}), Region(0.2, 0.1, 0.6, 0.8)),
        (json.dumps({"region": [-0.5, 0.0, 2.0, 0.5]}), Region(0.0, 0.0, 1.0, 0.5)),
    ],
)
def test_region_from_viewer(message, expected) -> None:
    assert scene_edit.region_from_viewer(message) == expected


@pytest.mark.parametrize("bad", [[0.1, 0.1, 0.11, 0.5], [0.1, "x", 0.5, 0.5], [0.1, 0.2]])
def test_region_from_viewer_rejects_bad_boxes(bad) -> None:
    with pytest.raises(RegionError):
        scene_edit.region_from_viewer({"region": bad})


def test_select_region_picks_centres_inside_the_box() -> None:
    cam = cam_at((0, -5, 0))
    # A row of points along x at the target depth, plus one behind the camera.
    means = np.array([[x, 0.0, 0.0] for x in np.linspace(-2, 2, 9)] + [[0, -10, 0]])
    sel = scene_edit.select_region(means, cam, 640, 448, Region(0.5, 0.3, 1.0, 0.7))
    # Right half of the view; x=0 lands on the box's left edge, which counts as inside.
    assert sel.tolist() == [False] * 4 + [True] * 5 + [False]


def test_select_region_keeps_the_surface_not_the_wall_behind_or_sparse_haze() -> None:
    cam = cam_at((0, -5, 0))
    rng = np.random.default_rng(0)
    grid = rng.uniform(-1, 1, size=(4000, 2))
    surface = np.c_[grid[:, 0], np.zeros(4000), grid[:, 1]]  # dense, at depth 5
    wall = np.c_[grid[:1000, 0] * 1.6, np.full(1000, 3.0), grid[:1000, 1] * 1.6]  # depth 8
    haze = np.c_[grid[:40, 0] * 0.4, np.full(40, -3.5), grid[:40, 1] * 0.4]  # sparse, depth 1.5
    means = np.vstack([surface, wall, haze])
    sel = scene_edit.select_region(means, cam, 640, 448, Region(0.0, 0.0, 1.0, 1.0))
    assert sel[:4000].mean() > 0.95  # the surface, despite the haze in front of it
    assert not sel[4000:5000].any()  # not the wall behind it
    assert sel[5000:].all()  # haze in the box is in front of the surface: it may change


def test_view_depth_is_the_median_centre_depth() -> None:
    cam = cam_at((0, -5, 0))
    means = np.array([[0.0, 0.0, 0.0], [0.1, 1.0, 0.0], [0.0, -1.0, 0.1], [0, -10, 0]])
    assert scene_edit.view_depth(means, cam, 640, 448) == pytest.approx(5.0)
    assert scene_edit.view_depth(np.array([[0, -10, 0.0]]), cam, 640, 448) is None


def test_region_pixels_follow_the_viewer_width_when_aspect_is_clamped() -> None:
    cam = preview.ViewCamera(cam_at((0, -5, 0)).c2w, 50.0, 4.0)  # wider than MAX_ASPECT
    w, h = preview.output_size(cam.aspect, 512 * 512)
    u0, v0, u1, v1 = scene_edit.region_pixels(Region(0.25, 0.0, 0.75, 1.0), cam, w, h)
    assert (u0 + u1) / 2 == pytest.approx(w / 2)
    assert u1 - u0 == pytest.approx(0.5 * h * 4.0)  # half of the viewer's width
    assert (v0, v1) == (0.0, h)


def _ply(n: int = 20, rest: int = 45) -> np.ndarray:
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(rest)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    v = np.zeros(n, dtype=[(k, "<f4") for k in names])
    rng = np.random.default_rng(0)
    for k in names:
        v[k] = rng.normal(size=n)
    return v


def test_split_and_join_ply_round_trip(tmp_path: Path) -> None:
    v = _ply()
    write_ply(tmp_path / "a.ply", v)
    v = read_ply(tmp_path / "a.ply")
    a = scene_edit.split_ply(v)
    assert a["sh0"].shape == (20, 1, 3) and a["shN"].shape == (20, 15, 3)
    # PLY stores all red terms first: f_rest_0..14 are channel 0, coefficients 1..15.
    assert a["shN"][3, 0, 0] == v["f_rest_0"][3] and a["shN"][3, 0, 1] == v["f_rest_15"][3]
    assert a["shN"][3, 14, 2] == v["f_rest_44"][3]
    back = scene_edit.join_ply(v, a)
    assert back.dtype == v.dtype
    for name in v.dtype.names:
        np.testing.assert_array_equal(back[name], v[name])
    changed = dict(a, sh0=a["sh0"] + 1)
    assert (scene_edit.join_ply(v, changed)["f_dc_1"] == v["f_dc_1"] + 1).all()


def test_split_ply_without_higher_order_terms() -> None:
    a = scene_edit.split_ply(_ply(rest=0))
    assert a["shN"].shape == (20, 0, 3)
    assert scene_edit.sh_degree(1 + a["shN"].shape[1]) == 0
    assert [scene_edit.sh_degree(k) for k in (1, 4, 9, 16)] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        scene_edit.sh_degree(5)


# --- Space side ---------------------------------------------------------------------


class FakeClient:
    def __init__(self) -> None:
        self.submitted: list[tuple] = []
        self.results: dict[str, object] = {}

    def submit_edit(self, job_id, scene_id, number, camera, prompt, region=None, seed=0):
        self.submitted.append((job_id, scene_id, number, prompt, region, seed))
        return f"fc-{job_id}"

    def poll(self, call_id: str):
        r = self.results.get(call_id)
        if isinstance(r, Exception):
            raise r
        return ("finished", r) if r is not None else ("running", None)


def test_edit_job_records_params_and_finishes(tmp_path: Path) -> None:
    store, client = jobs.LocalStore(tmp_path / "s"), FakeClient()
    mgr = jobs.JobManager(store, client, lambda: 5.0, prefix="edits", timeout_s=1800)
    cam = cam_at((0, -5, 1)).to_dict()
    job = mgr.create_edit("make the  table green", "s1", 2, cam, [0.1, 0.1, 0.5, 0.5], seed=4)
    assert client.submitted == [(job.job_id, "s1", 2, "make the  table green",
                                 [0.1, 0.1, 0.5, 0.5], 4)]  # fmt: skip
    assert job.kind == "edit" and job.summary == "make the table green"
    assert job.params["version"] == 2 and job.params["model"].startswith("timbrooks/")
    assert (store.root / "edits" / job.job_id / "job.json").is_file()
    client.results[job.call_id] = {"status": "done", "scene_path": "scenes/s1/v3", "version": 3}
    done = mgr.refresh(job)
    assert done.status == "done" and done.run["version"] == 3
    crashed = mgr.create_edit("x", "s1", 1, cam)
    client.results[crashed.call_id] = RuntimeError("lost")
    assert mgr.refresh(crashed).error["code"] == "edit_failed"
    slow = mgr.create_edit("x", "s1", 1, cam)
    client.results[slow.call_id] = type("FunctionTimeoutError", (Exception,), {})()
    assert "30 minutes" in mgr.refresh(slow).error["message"]


def test_prompt_log_lists_unfinished_edits_once() -> None:
    versions = [Version("s1", 2, 20.0, "edit_scene", "green table", "ip2p", parent=1),
                Version("s1", 1, 10.0)]  # fmt: skip
    done = jobs.Job("e1", "edit", "green table", 15.0, status="done")
    done.params = {"scene_id": "s1", "version": 1, "prompt": "green table"}
    failed = jobs.Job("e2", "edit", "red chair", 30.0, status="failed")
    failed.params = {"scene_id": "s1", "version": 2, "prompt": "red chair", "model": "ip2p"}
    other = jobs.Job("e3", "edit", "x", 40.0, status="running")
    other.params = {"scene_id": "s2", "version": 1}
    rows = [e.row() for e in prompt_log.scene_log("s1", versions, [], [done, failed, other])]
    assert [r[4] for r in rows] == ["edit of v2 (job e2) [failed]", "v2", "v1"]
    assert rows[0][1:4] == ["edit scene", "red chair", "ip2p"]


def test_viewer_has_region_mode() -> None:
    doc = viewer.viewer_document("/f.splat", viewer.Camera((0, -3, 1), (0, 0, 0)))
    assert '"region_mode"' in doc and "region:" in doc
