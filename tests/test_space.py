import sys
import types
from pathlib import Path

import pytest

from splat_studio import modal_client
from splat_studio.space import jobs, validation
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
