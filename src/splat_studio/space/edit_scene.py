"""Edit scene in the Space (task 4.4): start the job, then store its result as a version.

The Modal job writes the edited scene to `edits/<job>/`; it does not touch `scenes/`, so
version numbers are only ever given out by the Space (`versions.add_version`). The new
version records the job, and the job records the version, so a finished edit becomes
exactly one version even if the Space checks it several times.

An edit rewrites the Gaussians, so the new version has no `objects` list: objects added
before the edit become part of the scene and can no longer be hidden on their own.
"""

from __future__ import annotations

from splat_studio.editing.scene_edit import EditError, Region

from .jobs import Job, JobManager, Store
from .versions import Version, add_version, list_versions


class FinishError(Exception):
    """The finished edit could not be stored as a version."""


def parse_region(enabled: bool, region: dict | None) -> dict | None:
    """The region to send with an edit, or None to edit the whole scene.

    Raises EditError if a region is asked for but not (validly) marked.
    """
    if not enabled:
        return None
    if not region:
        raise EditError("Click the scene to mark the region first, or untick 'Limit to a region'.")
    parsed = Region.from_dict(region)
    return parsed.to_dict() if parsed else None


def finish_edit(store: Store, manager: JobManager, job: Job, now: float | None = None) -> Version:
    """The version made from a finished edit job; creates it on the first call."""
    if job.status != "done" or not job.scene_path:
        raise FinishError("The edit is not finished.")
    scene_id = job.params.get("scene_id")
    made = job.params.get("result_version")
    if made:
        existing = next((v for v in list_versions(store, scene_id) if v.number == made), None)
        if existing:
            return existing
    ply = store.download(f"{job.scene_path}/scene.ply")
    splat = store.download(f"{job.scene_path}/scene.splat")
    if ply is None or splat is None:
        raise FinishError("The edited scene files are missing.")
    version = add_version(
        store,
        scene_id,
        {"ply": ply, "splat": splat},
        mode="edit_scene",
        prompt=job.params.get("prompt") or job.summary,
        model=job.run.get("model") or job.params.get("model"),
        parent=job.params.get("version"),
        now=now,
        extra={
            "edit_job": job.job_id,
            "seed": job.params.get("seed"),
            "region": job.params.get("region"),
        },
    )
    job.params["result_version"] = version.number
    manager.save(job)
    return version
