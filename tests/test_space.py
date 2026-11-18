import html
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from splat_studio import modal_client
from splat_studio.reconstruction import settings as scene_settings
from splat_studio.space import jobs, validation, versions, viewer
from splat_studio.space.validation import UploadError, validate_upload

# --- upload validation (task 3.1) ---------------------------------------------


def _files(tmp_path: Path, n: int, ext: str = ".jpg", size: int = 10) -> list[Path]:
    out = []
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    with pytest.raises(UploadError, match="limit is 300 s"):
        validate_upload(_files(tmp_path, 1, ".mov"), probe=_probe(301))


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

    def submit_reconstruction(self, job_id: str, settings: dict | None = None) -> str:
        self.submitted.append((job_id, settings))
        return f"fc-{job_id}"

    def submit_object(self, job_id: str, prompt: str, seed: int = 0, **kwargs) -> str:
        self.submitted.append((job_id, prompt, seed, *kwargs.values()))
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
    job = mgr.create(upload, {"steps": 3000})
    used = {**scene_settings.PRESETS["Standard"], "steps": 3000}
    assert client.submitted == [(job.job_id, used)]
    assert job.params == used  # recorded in job.json
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


def test_create_defaults_to_standard(setup) -> None:
    _store, client, _clock, mgr, upload = setup
    job = mgr.create(upload)
    assert client.submitted == [(job.job_id, scene_settings.PRESETS["Standard"])]


@pytest.mark.parametrize(
    "settings, match",
    [({"steps": 500}, "between 1000 and 30000"), ({"steps": 30000}, "limit for one scene")],
)
def test_create_refuses_bad_settings_before_upload(setup, settings, match) -> None:
    store, client, _clock, mgr, _upload = setup
    photos = validate_upload(_files(store.root.parent / "many", 300))
    with pytest.raises(ValueError, match=match):
        mgr.create(photos, settings)
    assert client.submitted == [] and not (store.root / "jobs").exists()


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


def test_camera_from_viewer_reply_keeps_view() -> None:
    reply = {"matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1], "target": [1, 2, 3]}
    cam = viewer.camera_from_json(json.dumps(reply))
    assert cam == viewer.Camera((5.0, 6.0, 7.0), (1.0, 2.0, 3.0), viewer.DEFAULT_UP)
    assert viewer.camera_from_json("") is None
    assert viewer.camera_from_json(json.dumps({**reply, "target": None})) is None
    assert viewer.camera_from_json(json.dumps({**reply, "target": [5, 6, 7]})) is None


def test_scene_html_uses_given_splat_and_camera(tmp_path: Path) -> None:
    store = jobs.LocalStore(tmp_path / "store")
    other = tmp_path / "filtered.splat"
    other.write_bytes(np.zeros((2, 8), dtype="<f4").tobytes())
    cam = viewer.Camera((9.0, 9.0, 9.0), (0.0, 0.0, 0.0))
    out = html.unescape(viewer.scene_html(store, "scenes/j1/v1", splat=other, camera=cam))
    assert "filtered.splat" in out and '"position": [9.0, 9.0, 9.0]' in out


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


def test_history_rows_include_v1_without_version_json(tmp_path: Path) -> None:
    store = jobs.LocalStore(tmp_path / "store")
    store.upload_files({"scenes/j1/v1/scene.splat": _scene_files(tmp_path, "a")["splat"]})
    versions.add_version(
        store, "j1", _scene_files(tmp_path, "b"), mode="add_object",
        prompt="a mature\noak tree", model="flux + trellis", parent=1, now=2e9,
    )  # fmt: skip
    listed = versions.list_versions(store, "j1", v1_created_at=1e9)
    rows = versions.history_rows(listed, v1_input="video · 42 s")
    assert versions.HISTORY_COLUMNS == ["Version", "Date", "Mode", "Prompt", "Model", "From"]
    assert rows == [
        ["v2", "2033-05-18 03:33 UTC", "add object", "a mature oak tree", "flux + trellis", "v1"],
        ["v1", "2001-09-09 01:46 UTC", "reconstruction", "video · 42 s", "", ""],
    ]
    assert versions.history_rows(versions.list_versions(store, "j1"))[1][1:4] == [
        "date unknown", "reconstruction", "",
    ]  # fmt: skip
    assert versions.version_from_row(rows[0]) == 2
    assert versions.version_from_row(["x"]) is None and versions.version_from_row([]) is None


def test_versions_skip_malformed_objects(tmp_path: Path) -> None:
    store = jobs.LocalStore(tmp_path)
    store.write_json(
        "scenes/j1/v2/version.json",
        {"mode": "add_object", "objects": [{"id": "o1", "start": 5, "count": 3}, {"id": "o2"}]},
    )
    [v2] = versions.list_versions(store, "j1")
    assert [o["id"] for o in v2.objects] == ["o1"]


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


# --- object jobs (task 4.2) -------------------------------------------------------


def test_object_job_lives_under_objects_and_finishes(tmp_path: Path) -> None:
    store, client, clock = jobs.LocalStore(tmp_path / "s"), FakeClient(), Clock()
    mgr = jobs.JobManager(store, client, clock, prefix="objects", timeout_s=900)
    prompt = "a  mature\toak tree"
    job = mgr.create_object(prompt, scene_id="scene1", seed=7)
    assert client.submitted == [(job.job_id, prompt, 7)]
    assert job.kind == "object" and job.summary == "a mature oak tree"
    assert job.params == {"prompt": prompt, "seed": 7, "scene_id": "scene1"}
    assert (store.root / "objects" / job.job_id / "job.json").is_file()
    assert not (store.root / "jobs").exists()
    store.write_json(f"objects/{job.job_id}/run.json", {"status": "running"})
    assert mgr.refresh(job).status == "running"
    client.results[job.call_id] = {"status": "done", "scene_path": f"objects/{job.job_id}"}
    assert mgr.refresh(job).scene_path == f"objects/{job.job_id}"
    assert [j.job_id for j in mgr.list()] == [job.job_id]


def test_object_job_from_photo_uploads_reference(tmp_path: Path) -> None:
    store, client, clock = jobs.LocalStore(tmp_path / "s"), FakeClient(), Clock()
    mgr = jobs.JobManager(store, client, clock, prefix="objects", timeout_s=900)
    photo = tmp_path / "My Bench.JPG"
    photo.write_bytes(b"jpeg")
    job = mgr.create_object("", scene_id="scene1", image=photo)
    ref = f"objects/{job.job_id}/input/reference.jpg"
    assert client.submitted == [(job.job_id, "", 0, ref)]
    assert (store.root / ref).read_bytes() == b"jpeg"
    assert job.summary == "object from My Bench.JPG"
    assert job.params["source"] == "image" and job.params["reference_path"] == ref
    named = mgr.create_object("old bench", image=photo)
    assert named.summary == "old bench"


def test_validate_object_image(tmp_path: Path) -> None:
    from splat_studio.space.validation import UploadError, validate_object_image

    ok = tmp_path / "a.webp"
    ok.write_bytes(b"x")
    assert validate_object_image(str(ok)) == ok
    gif = tmp_path / "a.gif"
    gif.write_bytes(b"x")
    for bad in (None, gif):
        with pytest.raises(UploadError):
            validate_object_image(bad)


def test_object_job_errors_use_object_wording(tmp_path: Path) -> None:
    store, client, clock = jobs.LocalStore(tmp_path / "s"), FakeClient(), Clock()
    mgr = jobs.JobManager(store, client, clock, prefix="objects", timeout_s=900)
    crashed = mgr.create_object("lamp")
    client.results[crashed.call_id] = RuntimeError("worker lost")
    assert mgr.refresh(crashed).error["code"] == "object_failed"
    slow = mgr.create_object("lamp")
    client.results[slow.call_id] = type("FunctionTimeoutError", (Exception,), {})()
    assert "15 minutes" in mgr.refresh(slow).error["message"]


def test_viewer_script_is_valid_javascript(tmp_path: Path) -> None:
    """The page script is assembled from Python strings; a stray edit breaks the viewer."""
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    obj = {"url": "o", "position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "scale": [1] * 3}
    for o in (None, obj):
        page = viewer.viewer_document("u", viewer.Camera((0, 0, 1), (0, 0, 0)), o)
        script = re.search(r'<script type="module">(.*?)</script>', page, re.S).group(1)
        (tmp_path / "viewer.mjs").write_text(script)
        res = subprocess.run([node, "--check", str(tmp_path / "viewer.mjs")], capture_output=True)
        assert res.returncode == 0, res.stderr.decode()[-500:]


def test_object_list_and_library_choices() -> None:
    from splat_studio.space.add_object import library_choices, object_choices

    v = versions.Version(
        "s1", 3, objects=({"id": "o1", "prompt": "oak"}, {"id": "o2", "prompt": ""})
    )
    assert object_choices(v) == [("1. oak", "o1"), ("2. object", "o2")]
    assert object_choices(None) == []

    def job(i, scene, status="done", kind="object", t=0.0):
        j = jobs.Job(i, kind, f"thing {i}", t, status=status, scene_path=f"objects/{i}")
        j.params = {"scene_id": scene, "seed": 2}
        return j

    listed = [job("a", "s1", t=1), job("b", "s1", t=2), job("c", "s2"),
              job("d", "s1", status="running"), job("e", "s1", kind="edit")]  # fmt: skip
    assert library_choices(listed, "s1") == [
        ("thing b (seed 2, b)", "b"),
        ("thing a (seed 2, a)", "a"),
    ]
    assert library_choices(listed, None) == []


def test_scene_html_keeps_the_last_view_per_scene(tmp_path: Path) -> None:
    store = jobs.LocalStore(tmp_path)
    (tmp_path / "scenes/j1/v2").mkdir(parents=True)
    np.zeros((4, 8), dtype=np.float32).tofile(tmp_path / "scenes/j1/v2/scene.splat")
    doc = html.unescape(viewer.scene_html(store, "scenes/j1/v2"))
    assert '"viewKey": "j1"' in doc and '"restore": true' in doc  # same key for every version
    cam = viewer.Camera((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    doc = html.unescape(viewer.scene_html(store, "scenes/j1/v2", camera=cam))
    assert '"viewKey": "j1"' in doc and '"restore": false' in doc  # a reload keeps its camera
    assert '"viewKey": null' in viewer.viewer_document("/s.splat", cam)
