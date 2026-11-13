"""Add object in the Space (task 4.2): start placement, then merge into a new version.

Nothing is written until the user confirms; cancelling just drops the placement.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from splat_studio.editing.objects import Placement, default_placement, merge_files

from .jobs import Job, Store
from .versions import Version, add_version
from .viewer import splat_positions


class PlaceError(Exception):
    """A readable reason why the object could not be placed."""


@dataclass(frozen=True)
class SliderRanges:
    """Slider limits around the scene: +-2 scene radii, size up to 2 radii."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]
    size: tuple[float, float]


def start_placement(store: Store, version: Version) -> tuple[Placement, SliderRanges]:
    """Start placement for an object in `version`, and slider ranges that fit the scene."""
    splat = store.download(f"{version.path}/scene.splat")
    if splat is None:
        raise PlaceError(f"Version v{version.number} has no scene file.")
    placement, radius = default_placement(splat_positions(splat))
    r = 2 * radius

    def around(c: float) -> tuple[float, float]:
        return (round(c - r, 3), round(c + r, 3))

    ranges = SliderRanges(
        around(placement.x),
        around(placement.y),
        around(placement.z),
        (round(radius / 100, 4), round(r, 3)),
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
    with tempfile.TemporaryDirectory() as tmp:
        out_ply, out_splat = Path(tmp) / "scene.ply", Path(tmp) / "scene.splat"
        merge_files(scene_ply, object_ply, placement, out_ply, out_splat)
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
                "placement": placement.to_dict(),
            },
        )
