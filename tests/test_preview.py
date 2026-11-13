import json
from pathlib import Path

import numpy as np
import pytest

from splat_studio.editing import preview
from splat_studio.editing.preview import CameraError, camera_from_viewer
from splat_studio.reconstruction.export import to_splat_bytes
from splat_studio.space import jobs, viewer


def three_look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world as Three.js builds it in Object3D.lookAt for a camera."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def viewer_message(c2w: np.ndarray, fov=50.0, aspect=1.5) -> str:
    # Three.js `matrixWorld.elements` is column-major.
    return json.dumps({"type": "camera", "id": "a", "matrix": c2w.T.reshape(-1).tolist(),
                       "fov": fov, "aspect": aspect})  # fmt: skip


def project(cam: preview.ViewCamera, p, width=768, height=512) -> np.ndarray:
    pc = preview.viewmat(cam) @ np.append(np.asarray(p, dtype=np.float64), 1.0)
    uv = preview.intrinsics(cam, width, height) @ pc[:3]
    return np.array([uv[0] / uv[2], uv[1] / uv[2], pc[2]])


def test_viewer_camera_maps_to_opencv_view() -> None:
    cam = camera_from_viewer(viewer_message(three_look_at((0, -5, 0), (0, 0, 0))))
    u, v, depth = project(cam, (0, 0, 0))
    assert depth == pytest.approx(5)  # in front of the camera, 5 units away
    assert (u, v) == pytest.approx((384, 256))  # the look-at point is the image centre
    right = project(cam, (1, 0, 0))  # +x is to the right when looking along +y, z up
    assert right[0] > 384 and right[1] == pytest.approx(256)
    above = project(cam, (0, 0, 1))  # +z (up) is higher on screen, i.e. smaller v
    assert above[1] < 256 and above[0] == pytest.approx(384)


def test_vertical_fov_is_kept() -> None:
    cam = camera_from_viewer(viewer_message(three_look_at((0, -1, 0), (0, 0, 0)), fov=90))
    # With a 90 degree vertical fov, a point at 45 degrees up lands on the top edge.
    _, v, _ = project(cam, (0, 0, 1), width=640, height=480)
    assert v == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize(
    "message",
    [None, "", "not json", json.dumps({"matrix": [1, 2, 3], "fov": 50, "aspect": 1}),
     json.dumps({"matrix": [0] * 16, "fov": 50, "aspect": 1}),
     json.dumps({"matrix": np.eye(4).reshape(-1).tolist(), "fov": 0, "aspect": 1})],
)  # fmt: skip
def test_bad_camera_messages_are_rejected(message) -> None:
    with pytest.raises(CameraError):
        camera_from_viewer(message)


def test_camera_round_trips_through_dict() -> None:
    cam = camera_from_viewer(viewer_message(three_look_at((1, 2, 3), (0, 0, 0))))
    assert preview.ViewCamera.from_dict(json.loads(json.dumps(cam.to_dict()))) == cam


@pytest.mark.parametrize("aspect", [1.0, 1.5, 16 / 9, 0.6, 10.0])
def test_output_size_near_one_megapixel(aspect: float) -> None:
    w, h = preview.output_size(aspect)
    assert w % 64 == 0 and h % 64 == 0
    assert 0.8e6 < w * h < 1.25e6
    assert w / h == pytest.approx(min(aspect, preview.MAX_ASPECT), rel=0.15)


def _gaussians(n: int = 50) -> np.ndarray:
    rng = np.random.default_rng(0)
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]  # fmt: skip
    v = np.zeros(n, dtype=[(k, "<f4") for k in names])
    for k in names:
        v[k] = rng.normal(size=n)
    return v


def test_read_splat_decodes_what_export_writes() -> None:
    v = _gaussians()
    g = preview.read_splat(to_splat_bytes(v))
    assert len(g) == len(v)
    # .splat is sorted by size x opacity, so compare as sets of rows via the positions.
    order = np.lexsort(g.means.T)
    ref = np.stack([v["x"], v["y"], v["z"]], axis=1)
    ref_order = np.lexsort(ref.T)
    np.testing.assert_allclose(g.means[order], ref[ref_order], rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(g.quats, axis=1), 1, atol=1e-5)
    sig = 1 / (1 + np.exp(-v["opacity"][ref_order]))
    np.testing.assert_allclose(g.opacities[order], sig, atol=1 / 255 + 1e-6)
    assert g.colors.min() >= 0 and g.colors.max() <= 1


def test_read_splat_rejects_wrong_size() -> None:
    with pytest.raises(ValueError):
        preview.read_splat(b"x" * 33)


def test_depth_control_near_bright_far_dark_empty_black() -> None:
    depth = np.tile(np.linspace(1, 10, 40), (40, 1))
    alpha = np.ones_like(depth)
    alpha[:, :5] = 0  # empty strip
    img = preview.depth_to_control(depth, alpha)
    assert img.dtype == np.uint8 and img.shape == depth.shape
    assert (img[:, :5] == 0).all()
    assert img[0, 5] > img[0, 20] > img[0, 39] > 0  # nearer is brighter; far still > empty


def test_depth_control_handles_empty_and_flat() -> None:
    assert not preview.depth_to_control(np.ones((8, 8)), np.zeros((8, 8))).any()
    flat = preview.depth_to_control(np.full((8, 8), 3.0), np.ones((8, 8)))
    assert (flat == 255).all()


# --- Space side ---------------------------------------------------------------------


class FakeClient:
    def __init__(self) -> None:
        self.submitted: list[tuple] = []
        self.results: dict[str, object] = {}

    def submit_preview(self, job_id, scene_path, camera, prompt, seed=0, strength=0.7) -> str:
        self.submitted.append((job_id, scene_path, camera, prompt, seed, strength))
        return f"fc-{job_id}"

    def poll(self, call_id: str):
        r = self.results.get(call_id)
        if isinstance(r, Exception):
            raise r
        return ("finished", r) if r is not None else ("running", None)


def test_preview_job_lives_under_previews_and_never_touches_scenes(tmp_path: Path) -> None:
    store, client = jobs.LocalStore(tmp_path / "s"), FakeClient()
    mgr = jobs.JobManager(store, client, lambda: 1.0, prefix="previews", timeout_s=600)
    cam = camera_from_viewer(viewer_message(three_look_at((0, -5, 1), (0, 0, 0)))).to_dict()
    job = mgr.create_preview("autumn  street", "scenes/s1/v2", cam, seed=3, strength=0.5)
    assert client.submitted == [(job.job_id, "scenes/s1/v2", cam, "autumn  street", 3, 0.5)]
    assert job.kind == "preview" and job.summary == "autumn street"
    assert job.params["scene_path"] == "scenes/s1/v2" and job.params["camera"] == cam
    client.results[job.call_id] = {"status": "done", "scene_path": f"previews/{job.job_id}"}
    assert mgr.refresh(job).status == "done"
    assert not (store.root / "scenes").exists()
    crashed = mgr.create_preview("x", "scenes/s1/v1", cam)
    client.results[crashed.call_id] = RuntimeError("worker lost")
    assert mgr.refresh(crashed).error["code"] == "preview_failed"
    slow = mgr.create_preview("x", "scenes/s1/v1", cam)
    client.results[slow.call_id] = type("FunctionTimeoutError", (Exception,), {})()
    assert "10 minutes" in mgr.refresh(slow).error["message"]


def test_viewer_answers_camera_requests() -> None:
    doc = viewer.viewer_document("/f.splat", viewer.Camera((0, -3, 1), (0, 0, 0)))
    assert '"get_camera"' in doc and "matrixWorld.elements" in doc
