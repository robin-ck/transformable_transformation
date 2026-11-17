"""Scene details: name, notes and deletion (manage-scenes).

A scene is a finished reconstruction job. Its name and notes live in that job's
`jobs/<id>/job.json` (`Job.name`, `Job.notes`), so listing scenes needs no extra reads.

Deleting a scene removes, in one store commit, the scene and every job made for it:
  jobs/<id>/, scenes/<id>/                          the inputs and all versions
  edits/<job>/     (params.scene_id == id)          its edits
  previews/<job>/  (params.scene_path in scenes/<id>/)  its previews
  objects/<job>/   (params.scene_id == id)          objects generated for it
Image jobs belong to no scene and are kept. Deletion is refused while one of these
jobs is still queued or running (its worker would write into a deleted folder).
On HF Hub the files stay in the dataset's git history.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import prompt_log
from .jobs import FINAL, Job, JobManager, Store

MAX_NAME = 80
MAX_NOTES = 2000


class SceneError(ValueError):
    """A scene change the user asked for cannot be made; the message says why."""


def clean_name(text: str | None) -> str | None:
    """Name on one line, at most `MAX_NAME` characters; empty means no name."""
    name = " ".join((text or "").split())
    if len(name) > MAX_NAME:
        raise SceneError(f"The name is {len(name)} characters long; the limit is {MAX_NAME}.")
    return name or None


def clean_notes(text: str | None) -> str:
    notes = (text or "").strip()
    if len(notes) > MAX_NOTES:
        raise SceneError(f"The notes are {len(notes)} characters long; the limit is {MAX_NOTES}.")
    return notes


def scene_title(job: Job) -> str:
    """How a scene is named in messages: its name, else its input summary."""
    return job.name or job.summary


def choice_label(job: Job) -> str:
    """Label in the scene picker; the job id stays in it, so equal names stay apart."""
    return f"{job.name} · {job.job_id}" if job.name else f"{job.job_id} · {job.summary}"


def set_details(manager: JobManager, scene_id: str, name: str | None, notes: str | None) -> Job:
    """Save a scene's name and notes in its `job.json`. Raises SceneError."""
    job = manager.get(scene_id) if scene_id else None
    if job is None or job.status != "done":
        raise SceneError("Pick a finished scene first.")
    job.name, job.notes = clean_name(name), clean_notes(notes)
    manager.store.write_json(f"{manager.prefix}/{job.job_id}/job.json", asdict(job))
    return job


@dataclass
class DeletePlan:
    """What deleting a scene removes; `running` jobs block it."""

    scene_id: str
    folders: list[str]
    counts: dict[str, int] = field(default_factory=dict)  # versions, edits, previews, objects
    running: list[Job] = field(default_factory=list)

    def check(self) -> None:
        """Raise SceneError naming the scene's jobs that are still queued or running."""
        if self.running:
            what = ", ".join(f'{j.kind} "{j.summary}"' for j in self.running[:3])
            raise SceneError(
                f"Wait until this scene's running jobs finish ({what}); then delete it."
            )

    def summary(self) -> str:
        """e.g. "3 versions, 2 edits, 4 previews and 1 object"."""
        parts = [
            f"{n} {kind[:-1] if n == 1 else kind}"
            for kind, n in self.counts.items()
            if n or kind == "versions"
        ]
        return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def delete_plan(
    scene_id: str,
    n_versions: int,
    objects: list[Job],
    previews: list[Job],
    edits: list[Job],
) -> DeletePlan:
    """Folders to delete for `scene_id` and the jobs among them that are not finished.

    Pass the jobs of each kind as listed (unfinished ones refreshed first, so a job that
    finished while no page watched it does not block deletion).
    """
    mine = {
        "edits": [j for j in edits if j.params.get("scene_id") == scene_id],
        "previews": prompt_log.scene_previews(scene_id, previews),
        "objects": [j for j in objects if j.params.get("scene_id") == scene_id],
    }
    folders = [f"jobs/{scene_id}", f"scenes/{scene_id}"]
    folders += [f"{kind}/{j.job_id}" for kind, jobs in mine.items() for j in jobs]
    counts = {"versions": n_versions, **{k: len(v) for k, v in mine.items()}}
    running = [j for jobs in mine.values() for j in jobs if j.status not in FINAL]
    return DeletePlan(scene_id, folders, counts, running)


def delete_scene(store: Store, plan: DeletePlan) -> None:
    """Delete the plan's folders in one commit and the Space's local copies of the scene.

    Raises SceneError if a job of the scene is still queued or running.
    """
    plan.check()
    store.delete_folders(plan.folders, f"Delete scene {plan.scene_id}")
    remove_local_copies(Path(store.file_dir), plan.scene_id)


def remove_local_copies(file_dir: Path, scene_id: str) -> None:
    """Downloads and view-only files the Space made for a scene (see `versions.py`)."""
    for sub in ("downloads", "hidden"):
        for path in (file_dir / sub).glob(f"{scene_id}-v*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


def details_markdown(job: Job | None, n_versions: int, settings: str = "") -> str:
    """Facts about a scene under its name and notes: id, date, input, settings, versions."""
    if job is None:
        return "Pick a finished scene to see and change its details."
    made = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(job.finished_at or job.created_at))
    lines = [
        f"**Scene** `{job.job_id}` · made {made}",
        f"**Input:** {job.summary}",
    ]
    if settings:
        lines.append(f"**Settings:** {settings}")
    lines.append(f"**Versions:** {n_versions}")
    return "  \n".join(lines)
