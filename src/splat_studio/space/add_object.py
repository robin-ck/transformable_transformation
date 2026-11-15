"""Add object in the Space (task 4.2): start placement, then merge into a new version.

Nothing is written until the user confirms; cancelling just drops the placement.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from splat_studio.editing.objects import Placement, default_placement, merge_files
from splat_studio.reconstruction.export import ply_vertex_count

from .jobs import Job, Store
from .versions import Version, add_version
from .viewer import floor_height, splat_positions


class PlaceError(Exception):
    """A readable reason why the object could not be placed."""


@dataclass(frozen=True)
class SliderRanges:
    """Slider limits around the scene: +-2 scene radii, size up to 2 radii."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]
    size: tuple[float, float]
    floor: float = 0.0  # height of the scene floor; clicks off the splats land on it


def start_placement(store: Store, version: Version) -> tuple[Placement, SliderRanges]:
    """Start placement for an object in `version`, and slider ranges that fit the scene."""
    splat = store.download(f"{version.path}/scene.splat")
    if splat is None:
        raise PlaceError(f"Version v{version.number} has no scene file.")
    positions = splat_positions(splat)
    placement, radius = default_placement(positions)
    r = 2 * radius

    def around(c: float) -> tuple[float, float]:
        return (round(c - r, 3), round(c + r, 3))

    ranges = SliderRanges(
        around(placement.x),
        around(placement.y),
        around(placement.z),
        (round(radius / 100, 4), round(r, 3)),
        floor_height(positions),
    )
    return placement, ranges


def place_object(
    store: Store, version: Version, job: Job, placement: Placement, now: float | None = None
) -> Version:
    """Merge the object of a finished object `job` into `version`; returns the new version."""
    if job.status != "done" or not job.scene_path:
        raise PlaceError("The object is not ready yet.")
    scene_ply = store.download(f"{version.path}/scene.ply")
    if scene_ply is None:
        raise PlaceError(f"Version v{version.number} has no .ply file to merge into.")
    object_ply = store.download(f"{job.scene_path}/object.ply")
    if object_ply is None:
        raise PlaceError("The generated object file is missing.")
    earlier = list(version.objects)
    with tempfile.TemporaryDirectory() as tmp:
        out_ply, out_splat = Path(tmp) / "scene.ply", Path(tmp) / "scene.splat"
        try:
            total = merge_files(
                scene_ply, object_ply, placement, out_ply, out_splat,
                object_counts=[o["count"] for o in earlier],
            )  # fmt: skip
        except ValueError as e:
            raise PlaceError(f"The scene and its object list do not match ({e}).") from e
        start = ply_vertex_count(scene_ply)  # the object follows all scene Gaussians
        new_object = {
            "id": _next_id(earlier),
            "prompt": job.params.get("prompt") or job.summary,
            "object_job": job.job_id,
            "start": start,
            "count": total - start,
        }
        return add_version(
            store,
            version.scene_id,
            {"ply": out_ply, "splat": out_splat},
            mode="add_object",
            prompt=job.params.get("prompt") or job.summary,
            model=job.run.get("model"),
            parent=version.number,
            now=now,
            extra={
                "object_job": job.job_id,
                "seed": job.params.get("seed"),
                "source": job.params.get("source", "text"),
                "placement": placement.to_dict(),
            },
            objects=[*earlier, new_object],
        )


def _next_id(objects: list[dict]) -> str:
    """Ids o1, o2, ...: one more than the highest id in the inherited list."""
    numbers = [int(o["id"][1:]) for o in objects if str(o.get("id", ""))[1:].isdigit()]
    return f"o{max(numbers, default=0) + 1}"


def object_choices(version: Version | None) -> list[tuple[str, str]]:
    """(label, id) for each object in `version`, in the order they were added."""
    if version is None:
        return []
    return [
        (f"{i}. {o.get('prompt') or 'object'}", o["id"])
        for i, o in enumerate(version.objects, start=1)
    ]


def library_choices(jobs: list[Job], scene_id: str | None) -> list[tuple[str, str]]:
    """(label, job id) for finished objects generated for `scene_id`, newest first.

    These can be placed again (another copy, or one that was cancelled) without
    generating them anew.
    """
    if not scene_id:
        return []
    done = [
        j for j in jobs
        if j.kind == "object" and j.status == "done" and j.scene_path
        and j.params.get("scene_id") == scene_id
    ]  # fmt: skip
    done.sort(key=lambda j: j.created_at, reverse=True)
    return [(f"{j.summary} (seed {j.params.get('seed', 0)}, {j.job_id})", j.job_id) for j in done]
