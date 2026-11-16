import json

import pytest

from splat_studio import hub_outputs
from splat_studio.reconstruction.cost import RunLog


class FakeApi:
    """Records commits; each one moves `clock` on by `seconds`."""

    def __init__(self, clock: list[float] | None = None, seconds: float = 0.0) -> None:
        self.commits: list[dict] = []
        self.clock, self.seconds = clock, seconds

    def create_commit(self, **kwargs) -> None:
        if self.clock is None:
            raise RuntimeError("hub down")
        self.clock[0] += self.seconds
        self.commits.append(kwargs)


@pytest.fixture
def api(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(hub_outputs.time, "time", lambda: clock[0])
    return FakeApi(clock, seconds=26.0)


def _content(op) -> bytes:
    data = op.path_or_fileobj
    return data if isinstance(data, bytes) else open(data, "rb").read()


def test_commit_files_makes_one_commit(api, tmp_path):
    png = tmp_path / "a.png"
    png.write_bytes(b"png")
    hub_outputs.commit_files(
        api, "me/data", {"x/a.png": png, "x/b.bin": b"raw", "x/c.json": {"k": 1}}, "msg"
    )
    (commit,) = api.commits
    assert commit["repo_id"] == "me/data" and commit["repo_type"] == "dataset"
    assert commit["commit_message"] == "msg"
    ops = {op.path_in_repo: _content(op) for op in commit["operations"]}
    assert ops["x/a.png"] == b"png" and ops["x/b.bin"] == b"raw"
    assert json.loads(ops["x/c.json"]) == {"k": 1}


def test_finish_with_outputs_commits_files_and_run_log_together(api, tmp_path):
    log = RunLog(job_id="j1", gpu="L4", started_at=990.0)
    log.stage("restyle", 5.0)
    files = {}
    for name in ["render.png", "depth.png", "preview.png"]:
        (tmp_path / name).write_bytes(name.encode())
        files[f"previews/j1/{name}"] = tmp_path / name

    result = hub_outputs.finish_with_outputs(
        api, "me/data", "previews/j1", log, files, {"scene_path": "previews/j1"}, "Preview j1"
    )

    (commit,) = api.commits
    ops = {op.path_in_repo: _content(op) for op in commit["operations"]}
    assert sorted(ops) == sorted([*files, "previews/j1/run.json"])
    run = json.loads(ops["previews/j1/run.json"])
    assert run["status"] == "done" and run["scene_path"] == "previews/j1"
    assert "upload" not in run["stages"] and run["seconds"] == 10.0
    # The returned result (stored by the Space) has the commit time.
    assert result["status"] == "done" and result["scene_path"] == "previews/j1"
    assert result["stages"] == {"restyle": 5.0, "upload": 26.0}
    assert result["seconds"] == 36.0


def test_finish_with_outputs_raises_when_the_commit_fails(tmp_path):
    log = RunLog(job_id="j1", gpu="L4")
    with pytest.raises(RuntimeError):
        hub_outputs.finish_with_outputs(FakeApi(), "me/data", "p", log, {"p/a": b"x"}, {}, "m")
    assert "upload" not in log.stages
