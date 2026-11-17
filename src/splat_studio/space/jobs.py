"""Job tracking for the Space: upload inputs, start the Modal job, poll, keep state on HF Hub.

State lives in the dataset repo, so a job survives the user closing the page:
  jobs/<job_id>/input/...   uploaded files (read by the Modal job)
  jobs/<job_id>/job.json    Space-side record (this module)
  jobs/<job_id>/run.json    worker-side run log (written by modal_app/app.py)
  scenes/<job_id>/v1/...    finished scene
Object jobs (task 4.2) use the same layout under `objects/<job_id>/`, with the
generated object in `object.ply`, `object.splat` and `reference.png`; an uploaded
reference photo goes to `objects/<job_id>/input/reference.<ext>`.
Preview jobs (task 4.3) live under `previews/<job_id>/` with `render.png`, `depth.png`
and `preview.png`; they never write to `scenes/`. Futures (add-viewpoint-futures) are
preview jobs with extra `params` (see `futures.py`); their render and depth stay in the
viewpoint's folder, so their own folder holds only `preview.png` and `run.json`.
Edit jobs (task 4.4) live under `edits/<job_id>/` with before/after renders; the worker
itself saves the edited scene as the next version under `scenes/`, and its run log
names it (`version`).
Image jobs (Image editing tab) live under `images/<job_id>/` with `result.png`; an
uploaded input image goes to `images/<job_id>/input/source.<ext>`. A base image saved
without a GPU step is stored as-is as `images/<job_id>/result.<ext>`.
A reconstruction job's `job.json` also holds the scene's `name` and `notes`, which the
user can change; deleting a scene removes its folders (see `scenes.py`).
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

from splat_studio import modal_client, models
from splat_studio.editing.scene_edit import check_budget
from splat_studio.reconstruction import errors
from splat_studio.reconstruction import settings as scene_settings

from .validation import ValidUpload

FINAL = {"done", "failed"}
TRAIN_TIMEOUT_S = 60 * 60  # keep in sync with modal_app/app.py
OBJECT_TIMEOUT_S = 15 * 60  # keep in sync with modal_app/app.py
PREVIEW_TIMEOUT_S = 10 * 60  # keep in sync with modal_app/app.py
EDIT_TIMEOUT_S = 30 * 60  # keep in sync with modal_app/app.py
IMAGE_TIMEOUT_S = 10 * 60  # keep in sync with modal_app/app.py
_FOLDER = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9._-]+$")  # e.g. "scenes/<job_id>"


def check_folders(folders: list[str]) -> list[str]:
    """Folders that `Store.delete_folders` may remove: exactly `<top>/<job id>`.

    Guards against an empty or odd id deleting a whole top-level folder.
    """
    for f in folders:
        name = f.rsplit("/", 1)[-1]
        if not _FOLDER.match(f) or name in {".", ".."}:
            raise ValueError(f"refusing to delete {f!r}: not a <top>/<job id> folder")
    return list(folders)


class Store(Protocol):
    file_dir: Path  # local folder that holds downloaded files

    def upload_files(self, files: dict[str, Path], message: str) -> None: ...
    def write_json(self, path: str, data: dict) -> None: ...
    def read_json(self, path: str) -> dict | None: ...
    def list_dirs(self, prefix: str) -> list[str]: ...
    def download(self, path: str) -> Path | None: ...
    def delete_folders(self, folders: list[str], message: str) -> None: ...


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

    def delete_folders(self, folders: list[str], message: str = "") -> None:
        for folder in check_folders(folders):
            shutil.rmtree(self.root / folder, ignore_errors=True)

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
        from huggingface_hub.errors import EntryNotFoundError
        from huggingface_hub.hf_api import RepoFolder  # top-level export is newer

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

    def delete_folders(self, folders: list[str], message: str = "") -> None:
        """Delete whole folders in one commit, then their local copies.

        The files stay in the dataset's git history (see the Hub's repo settings to
        squash it)."""
        from huggingface_hub import CommitOperationDelete

        folders = check_folders(folders)
        if not folders:
            return
        ops = [CommitOperationDelete(path_in_repo=f"{f}/", is_folder=True) for f in folders]
        self.api.create_commit(
            self.repo_id, ops, commit_message=message or "delete", repo_type="dataset"
        )
        meta = self.file_dir / ".cache" / "huggingface" / "download"
        for f in folders:
            shutil.rmtree(self.file_dir / f, ignore_errors=True)
            shutil.rmtree(meta / f, ignore_errors=True)


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
    params: dict[str, Any] = field(default_factory=dict)  # e.g. prompt, seed, scene
    name: str | None = None  # scenes only: set by the user, shown instead of the job id
    notes: str = ""  # scenes only: free text from the user

    def elapsed(self, now: float) -> float:
        return (self.finished_at or now) - self.created_at

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def image_result_path(job: Job) -> str:
    """Dataset path of an image job's result (generated: result.png; saved: as uploaded)."""
    return f"{job.scene_path}/{job.params.get('result_file', 'result.png')}"


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
        prefix: str = "jobs",
        timeout_s: int = TRAIN_TIMEOUT_S,
    ) -> None:
        self.store, self.client, self.clock = store, client, clock
        self.prefix, self.timeout_s = prefix, timeout_s
        # Images a worker returned with its result (JPEG bytes, by job id); kept out of
        # job.json and shown by the Space before it fetches anything from the Hub.
        self.images: dict[str, bytes] = {}
        # Object masks of masked futures (PNG bytes, by job id; add-object-masking).
        self.masks: dict[str, bytes] = {}

    def _save(self, job: Job) -> Job:
        self.store.write_json(f"{self.prefix}/{job.job_id}/job.json", asdict(job))
        return job

    def _start(self, job: Job, submit: Callable[[], str]) -> Job:
        try:
            job.call_id = submit()
        except Exception as e:
            job.status, job.finished_at = "failed", self.clock()
            job.error = errors.ReconstructionError(
                "start_failed",
                "The GPU job could not be started. Please try again later.",
                f"Technical detail: {e!r}"[:300],
            ).to_dict()
        return self._save(job)

    def create(self, upload: ValidUpload, settings: dict | None = None) -> Job:
        """Upload inputs, record the job, start the Modal reconstruction.

        `settings` (`reconstruction.settings.LIMITS`) are checked first, also against the
        time limit; raises ValueError before anything is uploaded.
        """
        used = scene_settings.check_settings(settings)
        scene_settings.check_time(used, len(upload.files) if upload.kind == "photos" else None)
        now = self.clock()
        job = Job(new_job_id(now), upload.kind, upload.summary, now, params=used)
        files = {f"jobs/{job.job_id}/input/{i:03d}_{p.name}": p for i, p in enumerate(upload.files)}
        self.store.upload_files(files, f"Inputs for job {job.job_id}")
        self._save(job)
        return self._start(job, lambda: self.client.submit_reconstruction(job.job_id, used))

    def create_object(
        self,
        prompt: str,
        scene_id: str | None = None,
        seed: int = 0,
        image: Path | None = None,
    ) -> Job:
        """Record an object job and start the Modal text-to-3D function.

        With `image`, the photo is uploaded to `<prefix>/<job_id>/input/` and used as the
        reference image (no text-to-image step); `prompt` then only names the object.
        """
        now = self.clock()
        name = " ".join(prompt.split())[:80] or (f"object from {image.name}" if image else "")
        job = Job(new_job_id(now), "object", name[:80], now)
        job.params = {"prompt": prompt, "seed": seed, "scene_id": scene_id}
        reference_path = None
        if image is not None:
            reference_path = f"{self.prefix}/{job.job_id}/input/reference{image.suffix.lower()}"
            self.store.upload_files({reference_path: image}, f"Reference image for {job.job_id}")
            job.params.update(source="image", reference_path=reference_path)
        self._save(job)
        kwargs = {"reference_path": reference_path} if reference_path else {}
        return self._start(
            job, lambda: self.client.submit_object(job.job_id, prompt, seed=seed, **kwargs)
        )

    def create_edit(
        self,
        prompt: str,
        scene_id: str,
        number: int,
        camera: dict,
        region: list[float] | None = None,
        seed: int = 0,
        budget: dict | None = None,
    ) -> Job:
        """Record an edit job for version `number` of a scene and start it on Modal.

        `budget` (validated with `scene_edit.check_budget`) sets how much work the edit does.
        """
        budget = check_budget(budget)
        now = self.clock()
        job = Job(new_job_id(now), "edit", " ".join(prompt.split())[:80], now)
        job.params = {"prompt": prompt, "scene_id": scene_id, "version": number,
                      "camera": camera, "region": region, "seed": seed, "budget": budget,
                      "model": models.IMAGE_EDIT.label}  # fmt: skip
        self._save(job)
        return self._start(
            job,
            lambda: self.client.submit_edit(
                job.job_id,
                scene_id,
                number,
                camera,
                prompt,
                region=region,
                seed=seed,
                budget=budget,
            ),  # fmt: skip
        )

    def create_preview(
        self,
        prompt: str,
        scene_path: str,
        camera: dict,
        seed: int = 0,
        strength: float = 0.7,
        future: dict | None = None,
        input_path: str | None = None,
    ) -> Job:
        """Record a preview job (a 2D restyle of one view) and start it on Modal.

        `future` (`futures.future_params`) makes it a future of a viewpoint: the worker
        starts from the viewpoint's stored images, or renders `camera` and stores them if
        the viewpoint has none yet. `input_path` (a parent future's image) is restyled
        instead of the render.
        """
        now = self.clock()
        future = future or {}
        name = future.get("own_prompt") or (future.get("starter") or {}).get("label") or prompt
        job = Job(new_job_id(now), "preview", " ".join(name.split())[:80], now)
        spec = models.editor_spec(future.get("editor"))
        job.params = {"prompt": prompt, "seed": seed, "strength": strength,
                      "scene_path": scene_path, "camera": camera, "model": spec.label,
                      **future}  # fmt: skip
        kwargs: dict[str, Any] = {}
        if future:
            kwargs["view_prefix"] = future["view_prefix"]
            if future.get("steps"):
                kwargs["steps"] = future["steps"]
            if future.get("object_query"):
                kwargs["object_query"] = future["object_query"]
            if spec.modal_class != "PreviewRenderer":
                kwargs["class_name"] = spec.modal_class
        if input_path:
            job.params["input_path"] = kwargs["input_path"] = input_path
        self._save(job)
        return self._start(
            job,
            lambda: self.client.submit_preview(
                job.job_id, scene_path, camera, prompt, seed=seed, strength=strength, **kwargs
            ),
        )

    def create_image(
        self,
        prompt: str,
        seed: int = 0,
        source: Path | None = None,
        strength: float = 0.7,
        aspect: float = 1.0,
    ) -> Job:
        """Record an image job and start it on Modal: text-to-image, or image-to-image
        when `source` is given (uploaded to `<prefix>/<job_id>/input/`)."""
        now = self.clock()
        job = Job(new_job_id(now), "image", " ".join(prompt.split())[:80], now)
        job.params = {"prompt": prompt, "seed": seed, "mode": "text",
                      "model": models.TEXT_TO_IMAGE.label}  # fmt: skip
        source_path = None
        if source is not None:
            source_path = f"{self.prefix}/{job.job_id}/input/source{source.suffix.lower()}"
            self.store.upload_files({source_path: source}, f"Input image for {job.job_id}")
            job.params.update(mode="image", source_path=source_path, strength=strength)
        else:
            job.params["aspect"] = aspect
        self._save(job)
        return self._start(
            job,
            lambda: self.client.submit_image(
                job.job_id,
                prompt,
                seed=seed,
                source_path=source_path,
                strength=strength,
                aspect=aspect,
            ),  # fmt: skip
        )

    def save_image(self, image: Path) -> Job:
        """Store an uploaded image unchanged as a finished image job (no GPU), so it
        can be picked later as the base for edits."""
        now = self.clock()
        job = Job(new_job_id(now), "image", f"uploaded {image.name}"[:80], now)
        result_file = f"result{image.suffix.lower()}"
        job.scene_path = f"{self.prefix}/{job.job_id}"
        self.store.upload_files({f"{job.scene_path}/{result_file}": image}, f"Base {job.job_id}")
        job.params = {"mode": "upload", "result_file": result_file}
        job.status, job.finished_at = "done", now
        return self._save(job)

    def get(self, job_id: str) -> Job | None:
        data = self.store.read_json(f"{self.prefix}/{job_id}/job.json")
        return Job.from_dict(data) if data else None

    def _apply_result(self, job: Job, result: dict) -> None:
        result = dict(result)
        image = result.pop("image_jpeg", None)
        if isinstance(image, bytes):
            self.images[job.job_id] = image
        mask = result.pop("mask_png", None)
        if isinstance(mask, bytes):
            self.masks[job.job_id] = mask
        job.status = "done" if result.get("status") == "done" else "failed"
        job.error = result.get("error")
        job.scene_path = result.get("scene_path")
        job.run = result
        job.finished_at = self.clock()

    def refresh(self, job: Job, save: bool = True) -> Job:
        """Update a running job from Modal; finished jobs are returned unchanged.

        With `save=False` the change is not written to `job.json`; the caller saves it
        later (`save`), e.g. after showing a returned image first.
        """
        if job.status in FINAL:
            return job
        before = asdict(job)
        try:
            state, result = self.client.poll(job.call_id) if job.call_id else ("running", None)
        except Exception as e:
            run = self.store.read_json(f"{self.prefix}/{job.job_id}/run.json")
            failed = {
                "object": errors.object_failed,
                "preview": errors.preview_failed,
                "edit": errors.edit_failed,
                "image": errors.image_failed,
            }.get(job.kind, errors.training_failed)
            if run and run.get("status") in FINAL:
                self._apply_result(job, run)
            elif type(e).__name__ == "FunctionTimeoutError":
                self._apply_result(
                    job, {"status": "failed", "error": errors.timeout(self.timeout_s).to_dict()}
                )
            else:
                self._apply_result(job, {"status": "failed", "error": failed(repr(e)).to_dict()})
        else:
            if state == "finished":
                self._apply_result(job, result or {})
            else:
                run = self.store.read_json(f"{self.prefix}/{job.job_id}/run.json")
                job.status = "running" if run else "queued"
                job.run = run or {}
        return self._save(job) if save and asdict(job) != before else job

    def save(self, job: Job) -> Job:
        return self._save(job)

    def list(self, refresh: bool = True) -> list[Job]:
        """All jobs, newest first; unfinished jobs are refreshed."""
        jobs = [j for j in (self.get(i) for i in self.store.list_dirs(self.prefix)) if j]
        if refresh:
            jobs = [self.refresh(j) for j in jobs]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
