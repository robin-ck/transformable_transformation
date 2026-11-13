import html
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from splat_studio import modal_client
from splat_studio.space import jobs, validation, versions, viewer
from splat_studio.space.validation import UploadError, validate_upload

# --- upload validation (task 3.1) ---------------------------------------------


def _files(tmp_path: Path, n: int, ext: str = ".jpg", size: int = 10) -> list[Path]:
    out = []
    for i in range(n):
        p = tmp_path / f"f{i:03d}{ext}"
        p.write_bytes(b"x" * size)
        out.append(p)
    return out


def _probe(seconds: float):
    return lambda _p: (seconds, 30.0)


def test_valid_video(tmp_path: Path) -> None:
    [video] = _files(tmp_path, 1, ".MP4")
    up = validate_upload([video], probe=_probe(45))
    assert up.kind == "video" and up.files == [video] and "45 s" in up.summary


def test_video_too_long(tmp_path: Path) -> None:
    with pytest.raises(UploadError, match="limit is 120 s"):
        validate_upload(_files(tmp_path, 1, ".mov"), probe=_probe(121))


def test_video_too_big(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validation, "MAX_VIDEO_BYTES", 100)
    with pytest.raises(UploadError, match="limit is"):
        validate_upload(_files(tmp_path, 1, ".mp4", size=101), probe=_probe(10))


def test_unreadable_video(tmp_path: Path) -> None:
    def broken(_p: Path) -> tuple[float, float]:
        raise RuntimeError("ffprobe failed")

    with pytest.raises(UploadError, match="could not be read"):
        validate_upload(_files(tmp_path, 1, ".mp4"), probe=broken)


def test_two_videos_rejected(tmp_path: Path) -> None:
    with pytest.raises(UploadError, match="only one video"):
        validate_upload(_files(tmp_path, 2, ".mp4"), probe=_probe(10))


def test_too_few_photos(tmp_path: Path) -> None:
    with pytest.raises(UploadError, match="At least 20 photos are needed .you uploaded 8"):
        validate_upload(_files(tmp_path, 8))


def test_photo_limits_inclusive(tmp_path: Path) -> None:
    assert validate_upload(_files(tmp_path / "a", 20, ".png")).kind == "photos"
    assert len(validate_upload(_files(tmp_path / "b", 300)).files) == 300
    with pytest.raises(UploadError, match="At most 300"):
        validate_upload(_files(tmp_path / "c", 301))


@pytest.fixture(autouse=True)
def _mkdirs(tmp_path: Path) -> None:
    for d in "abc":
        (tmp_path / d).mkdir()


def test_mixed_and_unsupported_and_empty(tmp_path: Path) -> None:
    with pytest.raises(UploadError, match="not both"):
        validate_upload(_files(tmp_path, 1, ".mp4") + _files(tmp_path / "a", 20), _probe(5))
    with pytest.raises(UploadError, match="Unsupported file type: notes.txt"):
        (tmp_path / "notes.txt").write_text("hi")
        validate_upload([tmp_path / "notes.txt"])
    with pytest.raises(UploadError, match="Nothing uploaded"):
        validate_upload(None)


# --- job manager with mocked Modal (task 3.2) -----------------------------------


class FakeClient:
    def __init__(self) -> None:
        self.submitted: list[tuple[str, int]] = []
        self.results: dict[str, object] = {}  # call_id -> result dict or exception

    def submit_reconstruction(self, job_id: str, steps: int = 7000) -> str:
        self.submitted.append((job_id, steps))
        return f"fc-{job_id}"

    def poll(self, call_id: str):
        r = self.results.get(call_id)
        if isinstance(r, Exception):
            raise r
        return ("finished", r) if r is not None else ("running", None)


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def setup(tmp_path: Path):
    store = jobs.LocalStore(tmp_path / "store")
    client, clock = FakeClient(), Clock()
    upload = validate_upload(_files(tmp_path / "a", 20))
    return store, client, clock, jobs.JobManager(store, client, clock), upload


def test_create_uploads_inputs_and_spawns(setup) -> None:
    store, client, _clock, mgr, upload = setup
    job = mgr.create(upload, steps=500)
    assert client.submitted == [(job.job_id, 500)]
    assert job.status == "queued" and job.call_id == f"fc-{job.job_id}"
    inputs = sorted((store.root / "jobs" / job.job_id / "input").iterdir())
    assert len(inputs) == 20 and inputs[0].name == "000_f000.jpg"
    assert mgr.get(job.job_id) == job


def test_status_goes_queued_running_done(setup) -> None:
    store, client, clock, mgr, upload = setup
    job = mgr.create(upload)
    assert mgr.refresh(job).status == "queued"
    store.write_json(f"jobs/{job.job_id}/run.json", {"status": "running"})
    assert mgr.refresh(job).status == "running"
    clock.t += 300
    client.results[job.call_id] = {"status": "done", "scene_path": f"scenes/{job.job_id}/v1"}
    job = mgr.refresh(job)
    assert job.status == "done" and job.scene_path.endswith("/v1")
    assert job.elapsed(clock.t + 999) == 300  # frozen once finished
    # survives a page reload: a fresh manager reads the same state from the store
    again = jobs.JobManager(store, FakeClient(), clock).list()
    assert [(j.job_id, j.status) for j in again] == [(job.job_id, "done")]


def test_failed_job_keeps_readable_reason(setup) -> None:
    _store, client, _clock, mgr, upload = setup
    job = mgr.create(upload)
    err = {"code": "pose_failed", "message": "Camera positions could not be recovered.", "tip": "x"}
    client.results[job.call_id] = {"status": "failed", "error": err}
    assert mgr.refresh(job).error == err


def test_modal_timeout_maps_to_timeout_error(setup) -> None:
    _store, client, _clock, mgr, upload = setup
    job = mgr.create(upload)
    client.results[job.call_id] = type("FunctionTimeoutError", (Exception,), {})()
    job = mgr.refresh(job)
    assert job.status == "failed" and job.error["code"] == "timeout"


def test_poll_error_prefers_worker_run_log(setup) -> None:
    store, client, _clock, mgr, upload = setup
    job = mgr.create(upload)
    store.write_json(f"jobs/{job.job_id}/run.json", {"status": "done", "scene_path": "s/v1"})
    client.results[job.call_id] = RuntimeError("call expired")
    assert mgr.refresh(job).status == "done"


def test_spawn_failure_is_reported(setup) -> None:
    _store, client, _clock, mgr, upload = setup

    def boom(*a, **k):
        raise ConnectionError("no token")

    client.submit_reconstruction = boom
    job = mgr.create(upload)
    assert job.status == "failed" and job.error["code"] == "start_failed"


def test_list_newest_first(setup) -> None:
    _store, _client, clock, mgr, upload = setup
    first = mgr.create(upload)
    clock.t += 10
    second = mgr.create(upload)
    assert [j.job_id for j in mgr.list()] == [second.job_id, first.job_id]


def test_format_elapsed() -> None:
    assert jobs.format_elapsed(42) == "42 s"
    assert jobs.format_elapsed(65) == "1 min 05 s"
    assert jobs.format_elapsed(3720) == "1 h 02 min"


def test_poll_reraises_function_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class ModalTimeout(Exception):
        pass

    class FunctionTimeout(ModalTimeout):
        pass

    def get(timeout: float):
        raise FunctionTimeout

    fake = types.SimpleNamespace(
        FunctionCall=types.SimpleNamespace(from_id=lambda cid: types.SimpleNamespace(get=get)),
        exception=types.SimpleNamespace(
            TimeoutError=ModalTimeout, FunctionTimeoutError=FunctionTimeout
        ),
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    with pytest.raises(FunctionTimeout):
        modal_client.poll("fc-1")


# --- splat viewer (task 3.3) ----------------------------------------------------


def _write_splat(path: Path, positions: np.ndarray) -> Path:
    rec = np.zeros(len(positions), dtype=[("pos", "<f4", 3), ("rest", "u1", 20)])
    rec["pos"] = positions
    path.write_bytes(rec.tobytes())
    return path


def test_splat_positions_reads_and_subsamples(tmp_path: Path) -> None:
    pts = np.arange(3000, dtype=np.float32).reshape(1000, 3)
    f = _write_splat(tmp_path / "s.splat", pts)
    assert np.array_equal(viewer.splat_positions(f), pts)
    sub = viewer.splat_positions(f, max_points=100)
    assert len(sub) == 100 and np.array_equal(sub[1], pts[10])
    assert viewer.splat_positions(_write_splat(tmp_path / "e.splat", pts[:0])).shape == (0, 3)


def test_camera_looks_at_centre_from_above_and_ignores_floaters() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal([5.0, -2.0, 1.0], 1.0, size=(5000, 3)).astype(np.float32)
    pts[:10] = 1e4  # floaters
    pts[10] = np.nan
    cam = viewer.camera_for(pts)
    assert np.allclose(cam.look_at, [5, -2, 1], atol=0.1)
    offset = np.subtract(cam.position, cam.look_at)
    assert offset[2] > 0  # above, with the default z-up
    assert 1.0 < np.linalg.norm(offset) < 10.0
    assert cam.up == (0.0, 0.0, 1.0)
    assert viewer.camera_for(np.zeros((0, 3))).look_at == (0.0, 0.0, 0.0)


def test_viewer_iframe_escapes_and_carries_config() -> None:
    cam = viewer.Camera((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    doc = viewer.viewer_document('/gradio_api/file=/x/"a</script>.splat', cam)
    assert "</script>.splat" not in doc  # cannot close the script tag early
    assert '"position": [1.0, 2.0, 3.0]' in doc and "Reset view" in doc
    assert viewer.VIEWER_JS in doc and viewer.THREE_JS in doc and "@0." in viewer.VIEWER_JS
    frame = viewer.viewer_iframe("/s.splat", cam)
    assert frame.startswith("<iframe") and frame.count("<") == 2  # page is fully escaped
    assert html.unescape(frame.split('srcdoc="')[1].split('" style')[0]) == viewer.viewer_document(
        "/s.splat", cam
    )


def test_file_url_is_gradio_route(tmp_path: Path) -> None:
    url = viewer.file_url(tmp_path / "scene.splat")
    assert url.startswith("/gradio_api/file=") and url.endswith("/scene.splat") and "\\" not in url


def test_scene_html_from_store(tmp_path: Path) -> None:
    store = jobs.LocalStore(tmp_path)
    assert "Pick a finished scene" in viewer.scene_html(store, None)
    assert "No scene file" in viewer.scene_html(store, "scenes/j1/v1")
    (tmp_path / "scenes/j1/v1").mkdir(parents=True)
    _write_splat(tmp_path / "scenes/j1/v1/scene.splat", np.ones((10, 3), np.float32))
    out = viewer.scene_html(store, "scenes/j1/v1")
    assert out.startswith("<iframe") and "scenes/j1/v1/scene.splat" in out
    assert store.file_dir == tmp_path


def test_scene_html_reports_fetch_error() -> None:
    class Broken:
        def download(self, path: str) -> Path | None:
            raise ConnectionError("offline")

    assert "could not be fetched (ConnectionError)" in viewer.scene_html(Broken(), "scenes/x/v1")


# --- scene versions and download (task 3.4) ---------------------------------------


def _scene_files(tmp_path: Path, tag: str) -> dict[str, Path]:
    out = {}
    for kind in versions.FILES:
        p = tmp_path / f"{tag}.{kind}"
        p.write_bytes(tag.encode() * 4)
        out[kind] = p
    return out


def test_versions_list_newest_first_with_v1_fallback(tmp_path: Path) -> None:
    store = jobs.LocalStore(tmp_path / "store")
    assert versions.list_versions(store, "j1") == []
    store.upload_files({"scenes/j1/v1/scene.splat": _scene_files(tmp_path, "a")["splat"]})
    (tmp_path / "store/scenes/j1/notes").mkdir()  # not a version folder
    v2 = versions.add_version(
        store, "j1", _scene_files(tmp_path, "b"),
        mode="edit_scene", prompt="make the façade green", model="m", parent=1, now=2e9,
    )  # fmt: skip
    v3 = versions.add_version(store, "j1", _scene_files(tmp_path, "c"), mode="add_object", now=3e9)
    assert (v2.number, v3.number) == (2, 3)
    listed = versions.list_versions(store, "j1", v1_created_at=1e9)
    assert [v.number for v in listed] == [3, 2, 1]
    assert listed[1] == v2 and listed[2].created_at == 1e9 and listed[2].mode == "reconstruction"
    assert listed[1].label() == 'v2 · 2033-05-18 03:33 UTC · edit scene: "make the façade green"'
    assert "date unknown" in versions.list_versions(store, "j1")[2].label()


def test_versions_sort_numerically(tmp_path: Path) -> None:
    store = jobs.LocalStore(tmp_path)
    for n in (1, 2, 10):
        (tmp_path / f"scenes/j1/v{n}").mkdir(parents=True)
    assert [v.number for v in versions.list_versions(store, "j1")] == [10, 2, 1]
    assert versions.add_version(store, "j1", {}, mode="edit_scene").number == 11


def test_switching_back_keeps_later_versions(tmp_path: Path) -> None:
    store = jobs.LocalStore(tmp_path / "store")
    for tag in "abcd":
        versions.add_version(store, "j1", _scene_files(tmp_path, tag), mode="edit_scene")
    v2 = next(v for v in versions.list_versions(store, "j1") if v.number == 2)
    assert "scenes/j1/v2/scene.splat" in viewer.scene_html(store, v2.path)
    assert [v.number for v in versions.list_versions(store, "j1")] == [4, 3, 2, 1]
    assert (tmp_path / "store/scenes/j1/v4/scene.ply").read_bytes() == b"dddd"


def test_add_version_rejects_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown file kinds"):
        versions.add_version(jobs.LocalStore(tmp_path), "j1", {"obj": tmp_path}, mode="x")


def test_download_file_named_per_version(tmp_path: Path) -> None:
    store = jobs.LocalStore(tmp_path / "store")
    for tag in "ab":
        versions.add_version(store, "j1", _scene_files(tmp_path, tag), mode="edit_scene")
    [v2, v1] = versions.list_versions(store, "j1")
    p2 = versions.download_file(store, v2, "ply")
    p1 = versions.download_file(store, v1, "ply")
    assert p2.name == "j1-v2.ply" and p2.read_bytes() == b"bbbb"
    assert p1.name == "j1-v1.ply" and p1.read_bytes() == b"aaaa"
    assert versions.download_file(store, v2, "ply") == p2  # reused
    assert p2.parent == tmp_path / "store/downloads"
    missing = versions.Version("j9", 1)
    assert versions.download_file(store, missing, "splat") is None
