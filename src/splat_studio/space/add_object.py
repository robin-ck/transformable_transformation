"""Add object in the Space (task 4.2): start placement, then merge into a new version.

Nothing is written until the user confirms; cancelling just drops the placement.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from splat_studio.editing.objects import Placement, default_placement, merge_files
from splat_studio.reconstruction.export import ply_vertex_count, read_ply, write_ply

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
            "placement": placement.to_dict(),
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


def object_placement(obj: dict) -> Placement | None:
    """Stored placement of a placed object, or None for entries made before it was kept."""
    raw = obj.get("placement")
    if not isinstance(raw, dict):
        return None
    try:
        return Placement(**{k: float(v) for k, v in raw.items() if k in Placement.__annotations__})
    except (TypeError, ValueError):
        return None


def move_object(
    store: Store, version: Version, object_id: str, placement: Placement,
    now: float | None = None,
) -> Version:  # fmt: skip
    """Move (or resize) an object already in `version`; returns the new version.

    The object's Gaussians are cut out of the scene and its generated file is merged
    again at `placement`, after the other objects. Other objects keep their place.
    """
    earlier = list(version.objects)
    obj = next((o for o in earlier if o["id"] == object_id), None)
    if obj is None:
        raise PlaceError("That object is not in this version.")
    job_id = obj.get("object_job")
    object_ply = store.download(f"objects/{job_id}/object.ply") if job_id else None
    if object_ply is None:
        raise PlaceError("The generated object file is missing, so it cannot be moved.")
    scene_ply = store.download(f"{version.path}/scene.ply")
    if scene_ply is None:
        raise PlaceError(f"Version v{version.number} has no .ply file.")
    scene = read_ply(scene_ply)
    start, count = obj["start"], obj["count"]
    if start + count > len(scene):
        raise PlaceError("The scene and its object list do not match.")
    rest = []
    for o in earlier:
        if o["id"] == object_id:
            continue
        shift = count if o["start"] > start else 0
        rest.append({**o, "start": o["start"] - shift})
    kept = np.concatenate([scene[:start], scene[start + count :]])
    with tempfile.TemporaryDirectory() as tmp:
        cut = Path(tmp) / "cut.ply"
        write_ply(cut, kept)
        out_ply, out_splat = Path(tmp) / "scene.ply", Path(tmp) / "scene.splat"
        try:
            total = merge_files(
                cut, object_ply, placement, out_ply, out_splat,
                object_counts=[o["count"] for o in rest],
            )  # fmt: skip
        except ValueError as e:
            raise PlaceError(f"The scene and its object list do not match ({e}).") from e
        moved = {**obj, "start": len(kept), "count": total - len(kept)}
        moved["placement"] = placement.to_dict()
        label = obj.get("prompt") or "object"
        return add_version(
            store,
            version.scene_id,
            {"ply": out_ply, "splat": out_splat},
            mode="move_object",
            prompt=f"move {label}",
            parent=version.number,
            now=now,
            extra={"object_id": object_id, "placement": placement.to_dict()},
            objects=[*rest, moved],
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


# --- objects and edits tables (Scene tab) ------------------------------------------

OBJECT_COLUMNS = ["Object", "Status", "Action"]
EDIT_COLUMNS = ["Version", "Instruction", "Status", "Action"]
JOB_STATE = {"queued": "⏳ waiting for a GPU", "running": "⚙️ generating", "failed": "❌ failed"}


def object_rows(
    version: Version | None,
    hidden: list[str] | None,
    jobs: list[Job],
    scene_id: str | None,
    moving: str | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    """Rows of the objects table and, per row, a key [kind, id].

    Kinds: "placed" (an object in `version`: Hide/Show, click to move), "library" (a
    finished object generated for the scene: Add places a copy) and "job" (still
    generating or failed: no action).
    """
    rows, keys = [], []
    hidden_ids = set(hidden or [])
    for i, o in enumerate(version.objects if version else (), start=1):
        name = f"{i}. {o.get('prompt') or 'object'}"
        if o["id"] == moving:
            status, action = "✋ moving", ""
        elif o["id"] in hidden_ids:
            status, action = "hidden", "👁 Show"
        else:
            status, action = "in scene", "🚫 Hide"
        rows.append([name, status, action])
        keys.append(["placed", o["id"]])
    mine = [
        j for j in jobs
        if j.kind == "object" and scene_id and j.params.get("scene_id") == scene_id
    ]  # fmt: skip
    mine.sort(key=lambda j: j.created_at, reverse=True)
    for j in mine:
        label = f"{j.summary} (seed {j.params.get('seed', 0)})"
        if j.status == "done" and j.scene_path:
            rows.append([label, "generated", "➕ Add"])
            keys.append(["library", j.job_id])
        else:
            rows.append([label, JOB_STATE.get(j.status, j.status), ""])
            keys.append(["job", j.job_id])
    return rows, keys


def edit_rows(
    versions: list[Version], edit_jobs: list[Job], shown: int | None, hidden=None
) -> tuple[list[list[str]], list[list]]:
    """Rows of the edits table (newest first) and, per row, a key [kind, version number].

    Kinds: "line" (an edit the shown version builds on: Hide/Show, view only; a hidden
    edit also hides the edits after it), "other" (an edit on another branch: Open) and
    "job" (still running or failed: no action).
    """
    by_number = {v.number: v for v in versions}
    line, current = [], by_number.get(shown) if shown else None
    while current is not None and current.number not in line:
        line.append(current.number)
        current = by_number.get(current.parent) if current.parent else None
    hidden_edits = {
        int(h[1:]) for h in (hidden or ()) if h[:1] == "e" and h[1:].isdigit()
    }  # fmt: skip
    first_hidden = min((n for n in line if n in hidden_edits), default=None)
    rows: list[list[str]] = []
    keys: list[list] = []
    for j in sorted(edit_jobs, key=lambda j: j.created_at, reverse=True):
        if j.status == "done":
            continue
        state = {"queued": "⏳ waiting", "running": "⚙️ running"}.get(j.status, "❌ failed")
        rows.append(["", j.params.get("prompt") or j.summary, state, ""])
        keys.append(["job", None])
    for v in versions:
        if v.mode != "edit_scene":
            continue
        if v.number not in line:
            status, action, kind = "", "Open", "other"
        elif v.number in hidden_edits:
            status, action, kind = "hidden", "👁 Show", "line"
        elif first_hidden is not None and v.number > first_hidden:
            status, action, kind = f"hidden (with v{first_hidden})", "", "line"
        else:
            status, action, kind = "in scene", "🚫 Hide", "line"
        rows.append([f"v{v.number}", v.prompt or "", status, action])
        keys.append([kind, v.number])
    return rows, keys
