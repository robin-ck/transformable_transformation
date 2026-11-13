"""Job tracking for the Space: upload inputs, start the Modal job, poll, keep state on HF Hub.

State lives in the dataset repo, so a job survives the user closing the page:
  jobs/<job_id>/input/...   uploaded files (read by the Modal job)
  jobs/<job_id>/job.json    Space-side record (this module)
  jobs/<job_id>/run.json    worker-side run log (written by modal_app/app.py)
  scenes/<job_id>/v1/...    finished scene
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

from splat_studio import modal_client
from splat_studio.reconstruction import errors

from .validation import ValidUpload

FINAL = {"done", "failed"}
TRAIN_TIMEOUT_S = 60 * 60  # keep in sync with modal_app/app.py


class Store(Protocol):
    def upload_files(self, files: dict[str, Path], message: str) -> None: ...
    def write_json(self, path: str, data: dict) -> None: ...
    def read_json(self, path: str) -> dict | None: ...
    def list_dirs(self, prefix: str) -> list[str]: ...
    def download(self, path: str) -> Path | None: ...


class LocalStore:
    """Store on the local disk; for tests and UI work without HF Hub."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def upload_files(self, files: dict[str, Path], message: str = "") -> None:
        for dest, src in files.items():
            target = self.root / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(src).read_bytes())

    def write_json(self, path: str, data: dict) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2))

    def read_json(self, path: str) -> dict | None:
        target = self.root / path
        return json.loads(target.read_text()) if target.is_file() else None

    def list_dirs(self, prefix: str) -> list[str]:
        base = self.root / prefix
        return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []

    def download(self, path: str) -> Path | None:
        target = self.root / path
        return target if target.is_file() else None

    @property
    def file_dir(self) -> Path:
        """Folder that holds downloaded files (for Gradio's `allowed_paths`)."""
        return self.root


class HubStore:
    """Store in a private HF Hub dataset repo (needs HF_TOKEN with write access)."""

    def __init__(
        self, repo_id: str, token: str | None = None, cache_dir: Path = Path("scene_cache")
    ) -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.file_dir = Path(cache_dir)

    def upload_files(self, files: dict[str, Path], message: str = "") -> None:
        from huggingface_hub import CommitOperationAdd

        ops = [CommitOperationAdd(path_in_repo=d, path_or_fileobj=str(s)) for d, s in files.items()]
        self.api.create_commit(
            self.repo_id, ops, commit_message=message or "upload", repo_type="dataset"
        )

    def write_json(self, path: str, data: dict) -> None:
        self.api.upload_file(
            path_or_fileobj=json.dumps(data, indent=2).encode(),
            path_in_repo=path,
            repo_id=self.repo_id,
            repo_type="dataset",
        )

    def read_json(self, path: str) -> dict | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            local = hf_hub_download(
                self.repo_id, path, repo_type="dataset", token=self.api.token, force_download=True
            )
        except EntryNotFoundError:
            return None
        return json.loads(Path(local).read_text())

    def list_dirs(self, prefix: str) -> list[str]:
        from huggingface_hub import RepoFolder
        from huggingface_hub.errors import EntryNotFoundError

        try:
            tree = self.api.list_repo_tree(self.repo_id, path_in_repo=prefix, repo_type="dataset")
            return sorted(e.path.rsplit("/", 1)[-1] for e in tree if isinstance(e, RepoFolder))
        except EntryNotFoundError:
            return []

    def download(self, path: str) -> Path | None:
        """Local copy of a repo file. Scene versions never change, so a cached copy is reused."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            local = hf_hub_download(
                self.repo_id, path, repo_type="dataset", token=self.api.token,
                local_dir=self.file_dir,
            )  # fmt: skip
        except EntryNotFoundError:
            return None
        return Path(local)


@dataclass
class Job:
    job_id: str
    kind: str
    summary: str
    created_at: float
    status: str = "queued"  # queued | running | done | failed
    call_id: str | None = None
    finished_at: float | None = None
    scene_path: str | None = None
    error: dict[str, str] | None = None
    run: dict[str, Any] = field(default_factory=dict)

    def elapsed(self, now: float) -> float:
        return (self.finished_at or now) - self.created_at

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def new_job_id(now: float) -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime(now)) + "-" + uuid.uuid4().hex[:6]


def format_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


class JobManager:
    def __init__(
        self,
        store: Store,
        client: Any = modal_client,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store, self.client, self.clock = store, client, clock

    def _save(self, job: Job) -> Job:
        self.store.write_json(f"jobs/{job.job_id}/job.json", asdict(job))
        return job

    def create(self, upload: ValidUpload, steps: int = 7000) -> Job:
        """Upload inputs, record the job, start the Modal reconstruction."""
        now = self.clock()
        job = Job(new_job_id(now), upload.kind, upload.summary, now)
        files = {f"jobs/{job.job_id}/input/{i:03d}_{p.name}": p for i, p in enumerate(upload.files)}
        self.store.upload_files(files, f"Inputs for job {job.job_id}")
        self._save(job)
        try:
            job.call_id = self.client.submit_reconstruction(job.job_id, steps=steps)
        except Exception as e:
            job.status, job.finished_at = "failed", self.clock()
            job.error = errors.ReconstructionError(
                "start_failed",
                "The GPU job could not be started. Please try again later.",
                f"Technical detail: {e!r}"[:300],
            ).to_dict()
        return self._save(job)

    def get(self, job_id: str) -> Job | None:
        data = self.store.read_json(f"jobs/{job_id}/job.json")
        return Job.from_dict(data) if data else None

    def _apply_result(self, job: Job, result: dict) -> None:
        job.status = "done" if result.get("status") == "done" else "failed"
        job.error = result.get("error")
        job.scene_path = result.get("scene_path")
        job.run = result
        job.finished_at = self.clock()

    def refresh(self, job: Job) -> Job:
        """Update a running job from Modal; finished jobs are returned unchanged."""
        if job.status in FINAL:
            return job
        before = asdict(job)
        try:
            state, result = self.client.poll(job.call_id) if job.call_id else ("running", None)
        except Exception as e:
            run = self.store.read_json(f"jobs/{job.job_id}/run.json")
            if run and run.get("status") in FINAL:
                self._apply_result(job, run)
            elif type(e).__name__ == "FunctionTimeoutError":
                self._apply_result(
                    job, {"status": "failed", "error": errors.timeout(TRAIN_TIMEOUT_S).to_dict()}
                )
            else:
                self._apply_result(
                    job, {"status": "failed", "error": errors.training_failed(repr(e)).to_dict()}
                )
        else:
            if state == "finished":
                self._apply_result(job, result or {})
            else:
                run = self.store.read_json(f"jobs/{job.job_id}/run.json")
                job.status = "running" if run else "queued"
                job.run = run or {}
        return self._save(job) if asdict(job) != before else job

    def list(self, refresh: bool = True) -> list[Job]:
        """All jobs, newest first; unfinished jobs are refreshed."""
        jobs = [j for j in (self.get(i) for i in self.store.list_dirs("jobs")) if j]
        if refresh:
            jobs = [self.refresh(j) for j in jobs]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
