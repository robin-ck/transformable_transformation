"""Splat Studio Hugging Face Space: upload, validate, start reconstruction, track jobs.

Run locally:  python space/app.py
Secrets (Space settings): MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, HF_TOKEN, SPLAT_STUDIO_DATASET.
Without SPLAT_STUDIO_DATASET the app stores jobs in ./local_data (UI work only; the
Modal job reads from HF Hub, so reconstructions cannot finish in that mode).
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import gradio as gr

# On the Space, splat_studio is uploaded next to this file; locally use src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splat_studio import models  # noqa: E402
from splat_studio.editing import annotate, area, instructions, masking  # noqa: E402
from splat_studio.editing.objects import Placement  # noqa: E402
from splat_studio.editing.preview import CameraError, camera_from_viewer  # noqa: E402
from splat_studio.editing.scene_edit import (  # noqa: E402
    BUDGET_LIMITS,
    BUDGET_PRESETS,
    COLD_S,
    DEFAULT_PRESET,
    DOWNLOAD_S,
    RegionError,
    estimate_edit_breakdown,
    preset_of,
    region_from_viewer,
)
from splat_studio.reconstruction import settings as scene_settings  # noqa: E402
from splat_studio.space import (  # noqa: E402
    futures,
    prompt_log,
    scenes,
    tool_options,
    viewpoints,
)
from splat_studio.space.add_object import (  # noqa: E402
    EDIT_COLUMNS,
    OBJECT_COLUMNS,
    PlaceError,
    edit_rows,
    move_object,
    object_placement,
    object_rows,
    place_object,
    start_placement,
)
from splat_studio.space.jobs import (  # noqa: E402
    EDIT_TIMEOUT_S,
    FINAL,
    OBJECT_TIMEOUT_S,
    PREVIEW_TIMEOUT_S,
    HubStore,
    Job,
    JobManager,
    LocalStore,
    format_elapsed,
)
from splat_studio.space.validation import (  # noqa: E402
    HELP,
    UploadError,
    validate_object_image,
    validate_upload,
)
from splat_studio.space.versions import (  # noqa: E402
    HISTORY_COLUMNS,
    MODE_LABEL,
    Version,
    download_file,
    filtered_splat,
    history_rows,
    list_versions,
    version_from_row,
    version_numbers,
    view_splat,
)
from splat_studio.space.viewer import Camera, camera_from_json, scene_html  # noqa: E402


def load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines from a local `.env` (git-ignored); real env vars win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(Path(__file__).resolve().parent.parent / ".env")  # local runs only
DATASET = os.environ.get("SPLAT_STUDIO_DATASET")
STORE = HubStore(DATASET, os.environ.get("HF_TOKEN")) if DATASET else LocalStore(Path("local_data"))
JOBS = JobManager(STORE)
OBJECTS = JobManager(STORE, prefix="objects", timeout_s=OBJECT_TIMEOUT_S)
PREVIEWS = JobManager(STORE, prefix="previews", timeout_s=PREVIEW_TIMEOUT_S)
EDITS = JobManager(STORE, prefix="edits", timeout_s=EDIT_TIMEOUT_S)
STATUS_LABEL = {
    "queued": "⏳ queued",
    "running": "⚙️ running",
    "done": "✅ done",
    "failed": "❌ failed",
}
COLUMNS = ["Job", "Input", "Settings", "Status", "Elapsed", "Details"]


def on_upload(files: list[str] | None):
    """Check the upload; also returns the number of photos (None for a video or nothing),
    which the estimate uses instead of the frames setting."""
    if not files:
        return gr.update(value=HELP), gr.update(interactive=False), None
    try:
        upload = validate_upload(files)
    except UploadError as e:
        return gr.update(value=f"**Upload rejected.** {e}"), gr.update(interactive=False), None
    photos = len(upload.files) if upload.kind == "photos" else None
    return gr.update(value=f"**Ready.** {upload.summary}"), gr.update(interactive=True), photos


def settings_label(job: Job) -> str:
    """Jobs table: the scene settings a job ran with; empty for jobs from before them."""
    p = job.params
    if not all(k in p for k in SCENE_KEYS):
        return ""
    parts = [f"{p['steps']:,} steps"]
    if job.kind == "video":
        parts.append(f"{p['frames']} frames")
    parts.append(f"{p['max_side']:,} px")
    try:
        name = scene_settings.preset_of({k: p[k] for k in SCENE_KEYS})
    except ValueError:
        name = None
    return f"{name or CUSTOM}: " + " · ".join(parts)


def jobs_table(jobs: list[Job] | None = None) -> list[list[str]]:
    now = JOBS.clock()
    rows = []
    for j in JOBS.list() if jobs is None else jobs:
        if j.status == "failed" and j.error:
            details = f"{j.error.get('message', '')} {j.error.get('tip', '')}".strip()
        elif j.status == "done":
            details = f"Scene ready: {j.scene_path}"
        else:
            details = ""
        what = f"{j.name} · {j.summary}" if j.name else j.summary
        rows.append(
            [j.job_id, what, settings_label(j), STATUS_LABEL.get(j.status, j.status),
             format_elapsed(j.elapsed(now)), details]
        )  # fmt: skip
    return rows


# --- new scene settings (scene-settings) --------------------------------------------

SCENE_KEYS = list(scene_settings.LIMITS)  # steps, frames, max_side
SCENE_SLIDERS = {  # label, slider step, hint
    "steps": (
        "Training steps",
        500,
        "More steps give a sharper scene. About 5 s per 100 steps at 1,600 px, "
        "and more after 7,000 steps, as the scene keeps growing until 15,000.",
    ),
    "frames": (
        "Frames from the video",
        10,
        "Video only; a photo set uses every photo. More frames cover more views, but "
        "finding the camera positions (the longest part of the job) gets slower.",
    ),
    "max_side": (
        "Max image size (px)",
        100,
        "Long side of the training images, for video frames and photos. Smaller trains "
        "faster but gives a blurrier scene.",
    ),
}
STAGE_NAMES = {"frames": "frames", "poses": "camera positions", "training": "training",
               "saving": "saving"}  # fmt: skip


def scene_estimate(photos, steps, frames, max_side) -> str:
    """Markdown: rough GPU time and cost of a new scene, with the parts that make it up."""
    settings = dict(zip(SCENE_KEYS, (steps, frames, max_side), strict=True))
    try:
        parts = scene_settings.estimate_breakdown(settings, photos)
        scene_settings.check_time(settings, photos)
    except ValueError as e:
        return f"⚠️ {e}"
    total = sum(parts.values())
    detail = ", ".join(f"{STAGE_NAMES[k]} {_minutes(v)}" for k, v in parts.items() if v)
    note = (
        " Photo sets are not measured yet and may take longer: every photo is matched "
        "with every other."
        if photos
        else ""
    )
    return (
        f"⏱️ Up to about **{_minutes(total)}** on the GPU ({detail}), up to about "
        f"${scene_settings.estimate_usd(total):.2f}, plus any wait for a free GPU. Based on "
        "the slowest of three Standard video runs, which took 10 to 15 min; training takes "
        f"longer for scenes that grow more Gaussians.{note}"
    )


def on_scene_preset(name, photos):
    """Preset picked: set the sliders to it and update the estimate."""
    if name not in scene_settings.PRESETS:
        return [gr.skip()] * len(SCENE_KEYS) + [gr.skip()]
    values = [scene_settings.PRESETS[name][k] for k in SCENE_KEYS]
    return [*values, scene_estimate(photos, *values)]


def on_scene_slider(photos, *values):
    """A slider moved: show the matching preset (or Custom) and update the estimate."""
    try:
        name = scene_settings.preset_of(dict(zip(SCENE_KEYS, values, strict=True))) or CUSTOM
    except ValueError:
        name = CUSTOM
    return name, scene_estimate(photos, *values)


def scene_choices(jobs: list[Job]) -> list[tuple[str, str]]:
    """Finished scenes, oldest first (`jobs` is newest first).

    New scenes go at the end: Gradio's dropdown marks its selection by position, so
    putting a new scene first made it look selected and picking it did nothing.
    """
    done = [j for j in jobs if j.status == "done"]
    return [(scenes.choice_label(j), j.job_id) for j in reversed(done)]


def refresh_all(current: str | None):
    """Jobs table and scene picker from one listing (each listing reads the Hub)."""
    jobs = JOBS.list()
    choices = scene_choices(jobs)
    value = current if current in {v for _, v in choices} else None
    return jobs_table(jobs), gr.update(choices=choices, value=value)


def _scene(job_id: str | None) -> tuple[Job | None, list[Version]]:
    job = JOBS.get(job_id) if job_id else None
    if not job or job.status != "done":
        return job, []
    return job, list_versions(STORE, job_id, v1_created_at=job.finished_at)


def _versions(job_id: str | None) -> list[Version]:
    return _scene(job_id)[1]


def _version(job_id: str | None, number: int | None) -> Version | None:
    return next((v for v in _versions(job_id) if v.number == number), None)


def _end_placement(obj_job: str | None, placing: bool):
    """Outputs that leave placement mode: (place_group, placing, obj_job, moving).

    A finished object is dropped (nothing was saved); a job still generating is kept,
    and its object is shown on whichever version is picked when it is done.
    """
    return gr.update(visible=False), False, (None if placing else obj_job), None


def _shown(version: Version | None) -> str:
    if version is None:
        return ""
    what = MODE_LABEL.get(version.mode, version.mode)
    prompt = f': "{version.prompt}"' if version.prompt else ""
    return f"Showing **v{version.number}** · {what}{prompt}"


def _hidden_splat(version: Version | None, hidden: list[str] | None):
    """Local `.splat` without the hidden objects and edits, or None to use the version's
    own file."""
    if version is None or not hidden:
        return None
    try:
        if any(str(h).startswith("e") for h in hidden):
            return view_splat(STORE, version, _versions(version.scene_id), hidden)
        return filtered_splat(STORE, version, hidden)
    except Exception:  # fall back to the full scene rather than break the viewer
        return None


def scene_view(job_id: str | None, number: int | None, obj_job: str | None, placing: bool):
    """Open a version of a scene; outputs `SCENE_OUTPUTS`.

    With `number` None the base (v1, the reconstruction without edits) is shown;
    later versions stay one click away in the History table.

    (version, hidden, viewer, download, shown, history table, place_group, placing,
    obj_job, moving)
    """
    job, versions = _scene(job_id)
    version = next((v for v in versions if v.number == number), None) if number else None
    version = version or min(versions, key=lambda v: v.number, default=None)
    rows = history_rows(versions, v1_input=job.summary if job else None)
    return (
        version.number if version else None,
        [],
        scene_html(STORE, version.path if version else None),
        gr.update(value=None, visible=False),
        _shown(version),
        gr.update(value=rows),
        *_end_placement(obj_job, placing),
    )


def log_outputs(job_id: str | None, number: int | None):
    """History of the shown version and the scene's prompt log (task 4.5)."""
    versions = _versions(job_id)
    if not versions:
        return "", gr.update(value=[])
    previews = [j for j in PREVIEWS.list(refresh=False) if j.kind == "preview"]
    edits = [j for j in EDITS.list(refresh=False) if j.params.get("scene_id") == job_id]
    names = {v.id: v.name for v in _setup(job_id).viewpoints}
    rows = [e.row() for e in prompt_log.scene_log(job_id, versions, previews, edits, names)]
    return prompt_log.history_markdown(versions, number), gr.update(value=rows)


def on_pick_scene(job_id: str | None, obj_job: str | None, placing: bool):
    """Show the base version of the picked scene and list all its versions."""
    return scene_view(job_id, None, obj_job, placing)


def on_pick_row(job_id: str | None, table, obj_job: str | None, placing: bool, evt: gr.SelectData):
    """A click on a History row opens that version."""
    rows = table.values.tolist() if hasattr(table, "values") else (table or [])
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    number = version_from_row(rows[row]) if 0 <= row < len(rows) else None
    if number is None:
        return tuple(gr.skip() for _ in range(10))
    return scene_view(job_id, number, obj_job, placing)


# --- start tab and finished-job notice (redesign-space-gui 1.2, 1.3) -----------------


def _done_ids(jobs: list[Job]) -> list[str]:
    return [j.job_id for j in jobs if j.status == "done"]


def on_load(obj_job: str | None, placing: bool):
    """Open on the Scene tab with the newest finished scene (its base version), or on
    New scene if none.

    Outputs: tabs, jobs table, picker, seen, then `SCENE_OUTPUTS`.
    """
    jobs = JOBS.list()
    choices = scene_choices(jobs)
    newest = choices[-1][1] if choices else None
    return (
        gr.Tabs(selected="scene" if newest else "new"),
        jobs_table(jobs),
        gr.update(choices=choices, value=newest),
        _done_ids(jobs),
        *scene_view(newest, None, obj_job, placing),
    )


def on_tick(current: str | None, seen: list[str] | None):
    """Refresh jobs and picker; announce scenes that finished since the last listing.

    The notice never changes the open scene; picking it in the scene list opens it.
    """
    jobs = JOBS.list()
    done = _done_ids(jobs)
    if seen is not None and set(done) == set(seen):
        picker = gr.skip()  # unchanged: do not redraw the list while it may be open
    else:
        choices = scene_choices(jobs)
        value = current if current in {v for _, v in choices} else None
        picker = gr.update(choices=choices, value=value)
    if seen is not None:
        for job in jobs:
            if job.job_id in set(done) - set(seen):
                gr.Info(
                    f"Scene {job.job_id} ({scenes.scene_title(job)}) is ready. "
                    "Pick it in the 'Finished scene' list of the Scene tab to view it.",
                    duration=15,
                )
    return jobs_table(jobs), picker, done


# --- scene details: name, notes, delete (manage-scenes) -----------------------------


def on_details(scene_id: str | None):
    """Scene details of the picked scene; closes a pending delete question.

    Outputs: facts, name, notes, confirm group, status.
    """
    job = JOBS.get(scene_id) if scene_id else None
    if job is None or job.status != "done":
        return scenes.details_markdown(None, 0), "", "", gr.update(visible=False), ""
    n = len(version_numbers(STORE, job.job_id))
    facts = scenes.details_markdown(job, n, settings_label(job))
    return facts, job.name or "", job.notes, gr.update(visible=False), ""


def on_save_details(scene_id: str | None, name: str | None, notes: str | None):
    """Save name and notes. Outputs: status, picker, name, notes, scene id.

    The picker gets the new labels with no value: Gradio's dropdown keeps showing the
    old label while its value stays the same, so the next step sets it to the scene id.
    """
    try:
        job = scenes.set_details(JOBS, scene_id, name, notes)
    except scenes.SceneError as e:
        raise gr.Error(str(e)) from e
    choices = scene_choices(JOBS.list(refresh=False))
    return (
        f"✅ Saved. The scene is now listed as **{scenes.choice_label(job)}**.",
        gr.update(choices=choices, value=None),
        job.name or "",
        job.notes,
        job.job_id,
    )


def _delete_plan(scene_id: str | None) -> scenes.DeletePlan:
    """What deleting the scene removes; unfinished jobs are updated from Modal first."""
    job = JOBS.get(scene_id) if scene_id else None
    if job is None or job.status != "done":
        raise scenes.SceneError("Pick a finished scene first.")

    def updated(manager: JobManager, jobs: list[Job]) -> list[Job]:
        return [j if j.status in FINAL else manager.refresh(j) for j in jobs]

    def of_scene(manager: JobManager) -> list[Job]:
        jobs = manager.list(refresh=False)
        return updated(manager, [j for j in jobs if j.params.get("scene_id") == job.job_id])

    previews = prompt_log.scene_previews(job.job_id, PREVIEWS.list(refresh=False))
    return scenes.delete_plan(
        job.job_id,
        len(version_numbers(STORE, job.job_id)),
        objects=of_scene(OBJECTS),
        previews=updated(PREVIEWS, previews),
        edits=of_scene(EDITS),
        viewpoints=job.job_id in STORE.list_dirs("viewpoints"),
    )


def on_delete_ask(scene_id: str | None):
    """First click on Delete: say what goes and ask again. Outputs: confirm group, text."""
    try:
        plan = _delete_plan(scene_id)
        plan.check()
    except scenes.SceneError as e:
        raise gr.Error(str(e)) from e
    job = JOBS.get(scene_id)
    title = scenes.choice_label(job) if job else scene_id
    return gr.update(visible=True), (
        f"⚠️ Delete **{title}** for good? This removes its {plan.summary()} from the "
        "dataset, and the Space cannot bring them back."
    )


def on_delete(scene_id: str | None):
    """Delete the scene and open the newest remaining one.

    Outputs: status, confirm group, jobs table, picker, seen.
    """
    try:
        plan = _delete_plan(scene_id)
        scenes.delete_scene(STORE, plan)
        SETUPS.pop(scene_id, None)
    except scenes.SceneError as e:
        raise gr.Error(str(e)) from e
    except Exception as e:  # e.g. the Hub refused the commit
        raise gr.Error(f"The scene could not be deleted ({type(e).__name__}: {e})."[:300]) from e
    jobs = JOBS.list()
    choices = scene_choices(jobs)
    newest = choices[-1][1] if choices else None
    return (
        f"🗑️ Deleted scene `{scene_id}` ({plan.summary()}).",
        gr.update(visible=False),
        jobs_table(jobs),
        gr.update(choices=choices, value=newest),
        _done_ids(jobs),
    )


# --- objects and edits tables ------------------------------------------------------


def _placement_cfg(placement: Placement, floor: float | None, visible: bool) -> dict:
    cfg = placement.to_viewer()
    cfg["visible"] = bool(visible)
    if floor is not None:
        cfg["floor"] = floor
    return cfg


def on_tables(scene_id, number, hidden, moving):
    """Objects and edits tables of the open version. Outputs: objects, keys, edits, numbers."""
    version = _version(scene_id, number)
    jobs = OBJECTS.list(refresh=False)
    rows, keys = object_rows(version, hidden, jobs, scene_id, moving)
    edits = [j for j in EDITS.list(refresh=False) if j.params.get("scene_id") == scene_id]
    e_rows, numbers = edit_rows(_versions(scene_id), edits, number, hidden)
    return gr.update(value=rows), keys, gr.update(value=e_rows), numbers


def _reload(camera_json, version, hidden, obj_job, placing, moving, floor, show_obj, values):
    """Viewer of `version` without the hidden objects (and the one being moved), keeping
    the camera and any object being placed (at `values`, from the viewer)."""
    drop = [*hidden, moving] if moving else list(hidden)
    obj_path = cfg = None
    job = OBJECTS.get(obj_job) if (placing and obj_job) else None
    if job is not None and job.scene_path and values is not None:
        obj_path = job.scene_path
        cfg = _placement_cfg(Placement(*values), floor, show_obj)
    return scene_html(
        STORE, version.path, object_path=obj_path, placement=cfg,
        splat=_hidden_splat(version, drop), camera=camera_from_json(camera_json),
    )  # fmt: skip


def _slider_updates(placement: Placement, ranges) -> list:
    """Sliders set to `placement`, with ranges widened so the value always fits."""

    def fit(lo_hi, v):
        lo, hi = lo_hi
        return gr.update(minimum=min(lo, v), maximum=max(hi, v), value=v)

    p = placement
    return [
        fit(ranges.x, p.x), fit(ranges.y, p.y), fit(ranges.z, p.z),
        gr.update(value=p.turn_deg), fit(ranges.size, p.size),
        gr.update(value=p.tilt_x_deg), gr.update(value=p.tilt_y_deg),
    ]  # fmt: skip


def on_object_select(
    camera_json, scene_id, show_obj, number, hidden, obj_job, placing, moving, floor, keys,
    evt: gr.SelectData,
):  # fmt: skip
    """A click in the objects table.

    Action cell of a placed object: hide or show it (view only). Any other cell of a
    placed object: move or resize it. A generated object: add a copy to the scene.
    Outputs: status, hidden, viewer, obj_job, placing, moving, place_group, reference,
    floor, 7 sliders.
    """
    skip = [gr.skip()] * 16
    row, col = (evt.index if isinstance(evt.index, (list, tuple)) else (evt.index, 0))[:2]
    keys = keys or []
    if not 0 <= row < len(keys):
        return skip
    kind, key = keys[row]
    hidden = list(hidden or [])
    version = _version(scene_id, number)
    if version is None:
        raise gr.Error("Pick a finished scene first.")
    try:
        values = parse_placement(json.dumps(json.loads(camera_json or "{}").get("placement")))
    except (ValueError, AttributeError):
        values = None
    if kind == "job":
        return ["This object is still generating (or failed); it can be added when done.",
                *skip[1:]]  # fmt: skip
    if kind == "placed" and key == moving:
        return ["Move it in the viewer or with the sliders, then save or cancel.", *skip[1:]]
    if kind == "placed" and col == len(OBJECT_COLUMNS) - 1:
        # Follow the clicked label, not a toggle: one click can fire `select` twice.
        if "Show" in str(evt.value):
            hidden = [h for h in hidden if h != key]
        elif key not in hidden:
            hidden = [*hidden, key]
        viewer = _reload(
            camera_json, version, hidden, obj_job, placing, moving, floor, show_obj, values
        )
        return [gr.skip(), hidden, viewer, *skip[3:]]
    if placing and kind == "library" and key == obj_job and not moving:
        return skip  # the same click again (see above)
    if placing:
        raise gr.Error("Save or cancel the current object first.")
    if kind == "library":
        out = on_object_tick(key, False, scene_id, number, hidden, show_obj)
        out[1] = key
        # status, hidden, viewer, obj_job, placing, moving, group, ref, floor, sliders
        return [out[0], gr.skip(), out[3], out[1], out[2], None, *out[4:]]
    obj = next((o for o in version.objects if o["id"] == key), None)
    placement = object_placement(obj) if obj else None
    if obj is None or placement is None or not obj.get("object_job"):
        raise gr.Error("This object was placed before positions were saved; it can only be hidden.")
    job = OBJECTS.get(obj["object_job"])
    if job is None or not job.scene_path:
        raise gr.Error("The generated object for this entry is missing; it can only be hidden.")
    try:
        _, ranges = start_placement(STORE, version)
    except PlaceError as e:
        raise gr.Error(str(e)) from e
    reference = STORE.download(f"{job.scene_path}/reference.png")
    viewer = scene_html(
        STORE, version.path, object_path=job.scene_path,
        placement=_placement_cfg(placement, ranges.floor, show_obj),
        splat=_hidden_splat(version, [*hidden, key]), camera=camera_from_json(camera_json),
    )  # fmt: skip
    name = obj.get("prompt") or "object"
    return [
        f"✋ Moving **{name}**. Drag it in the viewer (Shift-drag turns, Alt-drag tilts) or "
        "use the sliders, then save it as a new version or cancel.",
        gr.skip(), viewer, job.job_id, True, key, gr.update(visible=True),
        str(reference) if reference else None, ranges.floor,
        *_slider_updates(placement, ranges),
    ]  # fmt: skip


# Like GET_CAMERA_JS, and also asks the viewer for the placement of an object being
# placed (only then does it answer): the sliders are not inputs, since a hidden group
# has no values.
OBJECT_SELECT_JS = """async (...args) => {
  const f = document.querySelector("iframe.splat-viewer");
  const ask = (type, reply, ms) => new Promise((resolve) => {
    if (!f || !f.contentWindow) return resolve(null);
    const id = Math.random().toString(36).slice(2);
    const on = (e) => { const d = e.data; if (d && d.type === reply && d.id === id) done(d); };
    const done = (v) => { window.removeEventListener("message", on); resolve(v); };
    window.addEventListener("message", on);
    f.contentWindow.postMessage({type, id}, "*");
    setTimeout(() => done(null), ms);
  });
  const camera = await ask("get_camera", "camera", 3000);
  const placement = await ask("get_placement", "placement", 400);
  const out = camera ? {...camera, placement} : (placement ? {placement} : null);
  return [out ? JSON.stringify(out) : "", ...args.slice(1)];
}"""


def on_edit_select(
    camera_json, scene_id, show_obj, number, hidden, keys, obj_job, placing, moving, floor,
    evt: gr.SelectData,
):  # fmt: skip
    """A click in the edits table. Outputs: `SCENE_OUTPUTS`.

    Hide/Show on an edit the shown version builds on: view the scene without it (its
    parent plus the objects placed later; view only). Open on another edit, or a click
    elsewhere in its row: open that edit's version.
    """
    skip = [gr.skip()] * 10
    row, col = (evt.index if isinstance(evt.index, (list, tuple)) else (evt.index, 0))[:2]
    kind, edit = keys[row] if keys and 0 <= row < len(keys) else ("job", None)
    if edit is None:
        return skip
    action = str(evt.value)
    if kind == "line" and col == len(EDIT_COLUMNS) - 1 and ("Hide" in action or "Show" in action):
        version = _version(scene_id, number)
        if version is None:
            return skip
        key = f"e{edit}"
        hidden = list(hidden or [])
        # Follow the clicked label, not a toggle: one click can fire `select` twice.
        if "Show" in action:
            hidden = [h for h in hidden if h != key]
        elif key not in hidden:
            hidden = [*hidden, key]
        try:
            values = parse_placement(json.dumps(json.loads(camera_json or "{}").get("placement")))
        except (ValueError, AttributeError):
            values = None
        viewer = _reload(
            camera_json, version, hidden, obj_job, placing, moving, floor, show_obj, values
        )
        out = list(skip)
        out[1], out[2] = hidden, viewer
        return out
    if placing:
        raise gr.Error("Save or cancel the current object first.")
    if edit == number and not hidden:
        return skip  # already shown
    return scene_view(scene_id, edit, obj_job, placing)


# --- add object (task 4.2; click and drag from redesign-space-gui 3) ------------------

# Sends slider values to the viewer iframe, which moves the object without reloading.
PLACEMENT_JS = """(x, y, z, turn, size, tiltX, tiltY) => {
  const f = document.querySelector("iframe.splat-viewer");
  if (f && f.contentWindow) {
    f.contentWindow.postMessage({type: "placement", x, y, z, turn, size, tiltX, tiltY}, "*");
  }
  return [x, y, z, turn, size, tiltX, tiltY];
}"""

# Shows or hides the object being placed; the placement itself is kept.
SHOW_OBJECT_JS = """(visible) => {
  const f = document.querySelector("iframe.splat-viewer");
  if (f && f.contentWindow) {
    f.contentWindow.postMessage({type: "object_visible", visible: !!visible}, "*");
  }
  return [visible];
}"""

# Gradio maps a js hook's return values onto the inputs that are not `gr.State`, in
# order, so every hooked event lists its State inputs last.

# "Save as new version" asks the viewer for its placement first, so the merged object
# lands where the user last saw it even if a drag never reached the sliders.
# Inputs: x, y, z, turn, size, tilt x, tilt y, then scene and States.
PLACE_JS = """async (...args) => {
  const f = document.querySelector("iframe.splat-viewer");
  if (!f || !f.contentWindow) return args;
  const id = Math.random().toString(36).slice(2);
  const p = await new Promise((resolve) => {
    const on = (e) => {
      const d = e.data;
      if (d && d.type === "placement" && d.id === id) { done(d); }
    };
    const done = (v) => { window.removeEventListener("message", on); resolve(v); };
    window.addEventListener("message", on);
    f.contentWindow.postMessage({type: "get_placement", id}, "*");
    setTimeout(() => done(null), 2000);
  });
  const vals = p && [p.x, p.y, p.z, p.turn, p.size, p.tiltX, p.tiltY];
  if (!vals || !vals.every(Number.isFinite)) return args;  // keep the sliders' values
  return [...vals, ...args.slice(7)];
}"""

# Page-level listener: the viewer posts `placement_changed` after a click or drag; the
# values are written into a hidden textbox whose change event updates the sliders.
BRIDGE_HEAD = """<script>
window.addEventListener("message", (e) => {
  const d = e.data;
  if (!d || d.type !== "placement_changed") return;
  const box = document.querySelector("#placement-bridge textarea, #placement-bridge input");
  if (!box) return;
  box.value = JSON.stringify({x: d.x, y: d.y, z: d.z, turn: d.turn, size: d.size,
                              tiltX: d.tiltX, tiltY: d.tiltY, t: Date.now()});
  box.dispatchEvent(new Event("input", {bubbles: true}));
});
// Futures gallery: a trash button on every tile but the first (the plain render). A click
// asks first, then sends "<tile index>:<time>" to the hidden #futures-delete-bridge.
(() => {
  const mark = (gallery) => {
    gallery.querySelectorAll(".gallery-item").forEach((item, i) => {
      if (i === 0 || item.querySelector(".future-trash")) return;
      const b = document.createElement("button");
      b.type = "button"; b.className = "future-trash";
      b.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#fff"'
        + ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        + '<path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14M10 11v6M14 11v6"/></svg>';
      b.title = "Delete this future"; b.setAttribute("aria-label", "Delete this future");
      b.addEventListener("click", (e) => {
        e.preventDefault(); e.stopPropagation();
        const items = [...gallery.querySelectorAll(".gallery-item")];
        const idx = items.indexOf(item);
        const label = item.querySelector(".caption-label");
        const cap = label ? label.textContent.trim() : "this future";
        if (idx < 1 || !confirm(`Delete the future "${cap}"? This cannot be undone.`)) return;
        const box = document.querySelector(
          "#futures-delete-bridge textarea, #futures-delete-bridge input");
        if (!box) return;
        box.value = idx + ":" + Date.now();
        box.dispatchEvent(new Event("input", {bubbles: true}));
      });
      item.appendChild(b);
    });
  };
  new MutationObserver(() => {
    const g = document.getElementById("futures-gallery");
    if (g) mark(g);
  }).observe(document.documentElement, {childList: true, subtree: true});
})();
</script>"""


# Scene tool switches in Gradio's own Settings dialog (add-tool-visibility-options). Gradio
# has no API for that dialog, so its markup is matched here (SEL) and anything missing just
# leaves all tools shown. The global choice travels as JSON through #tool-options-bridge.
TOOLS_HEAD = """<script>
(() => {
  const TOOLS = __TOOLS__;
  const SEL = {
    dialog: ".api-docs .api-docs-wrap", title: ".title",
    tabs: "#scene-tool-tabs", column: "#scene-tools",
    bridge: "#tool-options-bridge textarea, #tool-options-bridge input",
  };
  const idOf = Object.fromEntries(Object.entries(TOOLS).map(([id, l]) => [l, id]));
  let hidden = [], seen = null, state = "";
  const openTool = () => Object.keys(TOOLS).find((id) => {
    const panel = document.getElementById("tool-" + id);
    return panel && panel.style.display !== "none";
  });
  const apply = () => {
    const tabs = document.querySelector(SEL.tabs), col = document.querySelector(SEL.column);
    if (!tabs || !col) return;
    const off = new Set(hidden);
    tabs.querySelectorAll(".tab-wrapper button").forEach((b) => {
      const id = idOf[b.textContent.trim()];
      if (id) b.classList.toggle("tool-off", off.has(id));
    });
    const all = Object.keys(TOOLS).every((id) => off.has(id));
    col.classList.toggle("tool-off", all);
    const open = openTool();
    if (!all && open && off.has(open)) {
      const first = Object.keys(TOOLS).find((id) => !off.has(id));
      const btn = [...tabs.querySelectorAll(".tab-wrapper button")].find(
        (b) => !b.closest("[aria-hidden=true]") && idOf[b.textContent.trim()] === first);
      if (btn) btn.click();
    }
    const now = hidden.join(",");
    if (now !== state) {  // let Gradio re-measure the tab row and the viewer resize
      state = now;
      setTimeout(() => window.dispatchEvent(new Event("resize")), 50);
    }
  };
  const save = () => {
    const box = document.querySelector(SEL.bridge);
    if (!box) return;
    box.value = seen = JSON.stringify(hidden);
    box.dispatchEvent(new Event("input", {bubbles: true}));
  };
  const inject = () => {
    const dlg = document.querySelector(SEL.dialog);
    const title = dlg && dlg.querySelector(SEL.title);
    if (!title || !/^\\s*Settings/.test(title.textContent)) return;
    let sec = dlg.querySelector("#scene-tool-options");
    if (!sec) {
      sec = document.createElement("div");
      sec.id = "scene-tool-options"; sec.className = "banner-wrap";
      sec.style.padding = "var(--size-4, 16px)";
      const h = document.createElement("h2");
      h.textContent = "Scene tools";
      h.style.cssText = "font-weight: 600; margin-bottom: 4px;";
      const note = document.createElement("p");
      note.textContent = "Tools shown next to the viewer, for everyone using this Space.";
      note.style.cssText = "opacity: .75; font-size: .9em; margin-bottom: 6px;";
      sec.append(h, note);
      for (const [id, l] of Object.entries(TOOLS)) {
        const lab = document.createElement("label");
        const cb = document.createElement("input");
        cb.type = "checkbox"; cb.dataset.tool = id;
        cb.addEventListener("change", () => {
          hidden = [...sec.querySelectorAll("input[data-tool]")]
            .filter((c) => !c.checked).map((c) => c.dataset.tool);
          apply(); save();
        });
        lab.append(cb, document.createTextNode(l));
        sec.append(lab);
      }
      const after = dlg.children[0];
      after ? after.after(sec) : dlg.prepend(sec);
    }
    sec.querySelectorAll("input[data-tool]").forEach((c) => {
      c.checked = !hidden.includes(c.dataset.tool);
    });
  };
  const run = () => { try { apply(); inject(); } catch (e) { /* leave all tools shown */ } };
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; run(); });
  }).observe(document.documentElement, {childList: true, subtree: true});
  setInterval(() => {  // the saved choice arrives from the server on page load
    const box = document.querySelector(SEL.bridge);
    if (!box || box.value === seen) return;
    seen = box.value;
    try {
      const v = JSON.parse(box.value || "[]");
      hidden = Array.isArray(v) ? v.filter((id) => id in TOOLS) : [];
    } catch (e) { hidden = []; }
    run();
  }, 300);
})();
</script>""".replace("__TOOLS__", json.dumps(tool_options.TOOLS))
HIDDEN_TOOLS: list[list[str]] = []  # cached global choice (one entry once read)


def on_tool_options() -> str:
    """Page load: the global hidden-tools choice as JSON for the page script."""
    if not HIDDEN_TOOLS:
        HIDDEN_TOOLS.append(tool_options.load(STORE))
    return json.dumps(HIDDEN_TOOLS[0])


def on_save_tool_options(text: str | None) -> None:
    """Save the hidden tools the page script sent; other pages get them on their next load."""
    try:
        hidden = tool_options.save(STORE, tool_options.parse(text))
    except Exception as e:  # keep the page usable; the choice applies to this page only
        gr.Warning(f"Could not save the scene tool settings: {e}")
        return
    HIDDEN_TOOLS[:] = [hidden]


def on_bridge(text: str | None):
    """Placement from the viewer (JSON) into the seven sliders; skips anything malformed."""
    values = parse_placement(text)
    if values is None:
        return [gr.skip()] * 7
    return [gr.update(value=v) for v in values]


def parse_placement(text: str | None) -> tuple[float, ...] | None:
    """(x, y, z, turn, size, tilt x, tilt y); missing tilts (older pages) read as 0."""
    try:
        d = json.loads(text or "")
        values = tuple(float(d[k]) for k in ("x", "y", "z", "turn", "size"))
        values += tuple(float(d.get(k) or 0) for k in ("tiltX", "tiltY"))
    except (ValueError, TypeError, KeyError):
        return None
    if not all(math.isfinite(v) for v in values) or values[4] <= 0:
        return None
    return values  # type: ignore[return-value]


def on_generate(
    scene_id: str | None,
    number: int | None,
    prompt: str,
    seed: float | None,
    image: str | None,
    placing: bool,
):
    """Start an object job from the text, or from an uploaded photo if there is one."""
    if _version(scene_id, number) is None:
        raise gr.Error("Pick a finished scene first.")
    if placing:
        raise gr.Error("Save or cancel the current object first.")
    prompt = (prompt or "").strip()
    reference = None
    if image:
        try:
            reference = validate_object_image(image)
        except UploadError as e:
            raise gr.Error(str(e)) from e
    elif not prompt:
        raise gr.Error("Describe the object, e.g. 'a mature oak tree', or upload a photo of it.")
    job = OBJECTS.create_object(prompt, scene_id=scene_id, seed=int(seed or 0), image=reference)
    if job.status == "failed":
        return (
            f"**Could not start.** {job.error['message']} {job.error.get('tip', '')}",
            None,
            False,
            gr.update(visible=False),
        )
    return (
        f"⚙️ Generating **{job.summary}** (job {job.job_id}); this takes a few minutes.",
        job.job_id,
        False,
        gr.update(visible=False),
    )


def on_object_tick(obj_job, placing, scene_id, number, hidden, show_obj):
    """Poll the object job; when it is done, show the object in the viewer for placement.

    Outputs: status, obj_job, placing, viewer, place_group, reference image, floor,
    7 sliders.
    """
    rest = [gr.skip()] * 12  # placing, viewer, group, image, floor, sliders: unchanged
    if not obj_job or placing:
        return [gr.skip(), gr.skip(), *rest]
    job = OBJECTS.get(obj_job)
    job = OBJECTS.refresh(job) if job else None
    if job is None:
        return ["The object job was not found.", None, *rest]
    if job.status == "failed":
        err = job.error or {}
        msg = f"❌ **Object failed.** {err.get('message', '')} {err.get('tip', '')}".strip()
        return [msg, None, *rest]
    if job.status != "done":
        label = "⏳ Waiting for a GPU" if job.status == "queued" else "⚙️ Generating"
        elapsed = format_elapsed(job.elapsed(OBJECTS.clock()))
        return [f"{label}: **{job.summary}** ({elapsed})", gr.skip(), *rest]
    version = _version(scene_id, number)
    if version is None:
        return ["✅ Object ready. Pick a scene version to place it in.", gr.skip(), *rest]
    try:
        placement, ranges = start_placement(STORE, version)
    except PlaceError as e:
        return [f"**Cannot place the object.** {e}", gr.skip(), *rest]
    reference = STORE.download(f"{job.scene_path}/reference.png")
    viewer = scene_html(
        STORE,
        version.path,
        object_path=job.scene_path,
        placement=_placement_cfg(placement, ranges.floor, show_obj),
        splat=_hidden_splat(version, hidden),
    )
    return [
        f"✅ **{job.summary}** is ready. Click the scene to put it there, drag it to move "
        "(Shift-drag turns it, Alt-drag tilts it), or use the sliders. Then save it as a "
        "new version or cancel.",
        gr.skip(),
        True,
        viewer,
        gr.update(visible=True),
        str(reference) if reference else None,
        ranges.floor,
        gr.update(minimum=ranges.x[0], maximum=ranges.x[1], value=placement.x),
        gr.update(minimum=ranges.y[0], maximum=ranges.y[1], value=placement.y),
        gr.update(minimum=ranges.z[0], maximum=ranges.z[1], value=placement.z),
        gr.update(value=0),
        gr.update(minimum=ranges.size[0], maximum=ranges.size[1], value=placement.size),
        gr.update(value=0),
        gr.update(value=0),
    ]


def on_place(
    x, y, z, turn, size, tilt_x, tilt_y, scene_id, number, obj_job, moving,
    progress=gr.Progress(),  # noqa: B008 - Gradio's progress idiom
):  # fmt: skip
    """Merge the object (or move a placed one); opens the new version.

    Outputs: status, then `SCENE_OUTPUTS`.
    """
    version = _version(scene_id, number)
    job = OBJECTS.get(obj_job) if obj_job else None
    if version is None or job is None:
        raise gr.Error("Nothing to place: generate an object for a picked scene first.")
    placement = Placement(x, y, z, turn, size, tilt_x, tilt_y)
    progress(0.1, desc="Saving the scene as a new version…")
    try:
        if moving:
            new = move_object(STORE, version, moving, placement)
        else:
            new = place_object(STORE, version, job, placement)
    except PlaceError as e:
        raise gr.Error(str(e)) from e
    view = list(scene_view(scene_id, new.number, None, False))
    view[-4:] = [gr.update(visible=False), False, None, None]
    done = "Moved" if moving else "Added"
    return (f"✅ {done} **{job.summary}** as v{new.number}.", *view)


def on_cancel(camera_json, scene_id, number, hidden):
    """Outputs: status, viewer, place_group, placing, obj_job, moving."""
    version = _version(scene_id, number)
    return (
        "Cancelled; no new version was made.",
        scene_html(
            STORE,
            version.path if version else None,
            splat=_hidden_splat(version, hidden),
            camera=camera_from_json(camera_json),
        ),  # fmt: skip
        gr.update(visible=False),
        False,
        None,
        None,
    )


# --- edit scene (task 4.4) -----------------------------------------------------------

REGION_JS = """(on) => {
  const f = document.querySelector("iframe.splat-viewer");
  if (f && f.contentWindow) f.contentWindow.postMessage({type: "region_mode", on}, "*");
  return on;
}"""


BUDGET_KEYS = list(BUDGET_LIMITS)  # n_views, steps, edit_every, denoise_steps
BUDGET_HINTS = {
    "n_views": (
        "Views around the camera that are edited first. More views keep the edit "
        "consistent further around the scene; each costs about one image edit."
    ),
    "steps": (
        "Training steps on the edited views. More steps follow the edits more closely; "
        "about 5 s per 100 steps, plus the re-edits they bring."
    ),
    "edit_every": (
        "One view is re-edited every this many steps. Lower re-edits more often: "
        "views agree better, but this is the largest share of the time."
    ),
    "denoise_steps": (
        "Image-editor steps per edit. Fewer is faster and a bit rougher; "
        "every edit (first and re-edits) costs about 0.06 s per step."
    ),
}
CUSTOM = "Custom"


def _minutes(seconds: float) -> str:
    m, s = divmod(int(round(seconds / 10) * 10), 60)
    return (f"{m} min {s:02d} s" if s else f"{m} min") if m else f"{s} s"


def edit_estimate(n_views, steps, edit_every, denoise_steps) -> str:
    """Markdown: rough job time for these settings, with the parts that make it up."""
    budget = dict(zip(BUDGET_KEYS, (n_views, steps, edit_every, denoise_steps), strict=True))
    try:
        parts = estimate_edit_breakdown(budget)
    except ValueError as e:
        return f"⚠️ {e}"
    names = {"first_edits": "first edits", "training": "training",
             "re_edits": "re-edits", "saving": "loading and saving"}  # fmt: skip
    detail = ", ".join(f"{names[k]} {_minutes(v)}" for k, v in parts.items())
    return (
        f"⏱️ About **{_minutes(sum(parts.values()))}** on the GPU ({detail}). "
        f"Add about {int(COLD_S)} s if no edit ran in the last 5 minutes, and "
        f"{int(DOWNLOAD_S)} s for a version the GPU has not loaded before. Rough times "
        "for the 1.5 M Gaussian test scene; larger scenes take longer."
    )


def on_budget_preset(name):
    """Preset picked: set the sliders to it and update the estimate."""
    if name not in BUDGET_PRESETS:
        return [gr.skip()] * len(BUDGET_KEYS) + [gr.skip()]
    values = [BUDGET_PRESETS[name][k] for k in BUDGET_KEYS]
    return [*values, edit_estimate(*values)]


def on_budget_slider(*values):
    """A slider moved: show the matching preset (or Custom) and update the estimate."""
    budget = dict(zip(BUDGET_KEYS, values, strict=True))
    try:
        name = preset_of(budget) or CUSTOM
    except ValueError:
        name = CUSTOM
    return name, edit_estimate(*values)


def on_edit(
    camera_json, scene_id, prompt, region_on, seed,
    n_views, steps, edit_every, denoise_steps, number, placing,
):  # fmt: skip
    version = _version(scene_id, number)
    if version is None:
        raise gr.Error("Pick a finished scene first.")
    if placing:
        raise gr.Error("Place or cancel the object first.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Write an instruction first, e.g. 'make the table green'.")
    try:
        camera = camera_from_viewer(camera_json)
        region = region_from_viewer(camera_json) if region_on else None
    except (CameraError, RegionError) as e:
        raise gr.Error(str(e)) from e
    if region_on and region is None:
        raise gr.Error("Drag a box in the viewer first, or switch off 'Limit to a region'.")
    budget = dict(zip(BUDGET_KEYS, (n_views, steps, edit_every, denoise_steps), strict=True))
    try:
        job = EDITS.create_edit(
            prompt,
            scene_id,
            version.number,
            camera.to_dict(),
            region=region.to_list() if region else None,
            seed=int(seed or 0),
            budget=budget,
        )
    except ValueError as e:
        raise gr.Error(str(e)) from e
    if job.status == "failed":
        return f"**Could not start.** {job.error['message']} {job.error.get('tip', '')}", None
    where = "inside the marked box" if region else "around this view"
    took = _minutes(sum(estimate_edit_breakdown(job.params["budget"]).values()))
    return (
        f"⚙️ Editing v{version.number} {where} (job {job.job_id}). This takes about {took} "
        "on the GPU, plus any wait for one; the result is saved as a new version even if "
        "you close the page.",
        job.job_id,
    )


# The last stage in the run log -> what the job does now. The first three are load stages
# of a fresh container (`SceneEditor.load`); a warm container skips them.
EDIT_STAGES = {
    "imports": "loading the image editor",
    "warm_gsplat": "loading the image editor",
    "load_editor": "loading the scene",
    "download_scene": "loading the scene",
    "load_scene": "rendering views",
    "render_views": "editing the views",
    "encode_prompt": "editing the views",
    "first_edits": "training on the edited views",
    "train": "saving",
}


def on_edit_tick(edit_job, scene_id, obj_job, placing):
    """Poll the edit job; when done, open its version if its scene is shown.

    Outputs: status, edit_job, then `SCENE_OUTPUTS`. (The Modal job saves the version.)
    """
    keep = [gr.skip()] * 10
    if not edit_job:
        return [gr.skip(), gr.skip(), *keep]
    job = EDITS.get(edit_job)
    job = EDITS.refresh(job) if job else None
    if job is None:
        return ["The edit job was not found.", None, *keep]
    if job.status == "failed":
        err = job.error or {}
        msg = f"❌ **Edit failed.** {err.get('message', '')} {err.get('tip', '')}".strip()
        return [msg, None, *keep]
    elapsed = format_elapsed(job.elapsed(EDITS.clock()))
    if job.status != "done":
        if job.status == "queued":
            return [f"⏳ Waiting for a GPU: **{job.summary}** ({elapsed})", gr.skip(), *keep]
        stages = list((job.run or {}).get("stages", {}))
        doing = EDIT_STAGES.get(stages[-1], "starting") if stages else "starting"
        return [f"⚙️ Editing: **{job.summary}**, {doing} ({elapsed})", gr.skip(), *keep]
    number = (job.run or {}).get("version")
    msg = f"✅ **{job.summary}** saved as v{number} ({elapsed})."
    if scene_id != job.params.get("scene_id"):
        return [msg + " Pick its scene to see it.", None, *keep]
    if placing:
        gr.Info(f"Edit saved as v{number}. Open it from the History tab after placing.")
        return [msg + " Open it from the History tab after placing.", None, *keep]
    gr.Info(f"Edit saved as v{number}; it is now shown.")
    return [msg, None, *scene_view(scene_id, number, obj_job, placing)]


# --- preview (task 4.3) --------------------------------------------------------------

# Asks the viewer iframe for its camera and passes it on as the first input (JSON).
# Gradio hands the js hook the input values followed by the output values, so the
# camera goes first and everything else is passed through unchanged.
GET_CAMERA_JS = """async (...args) => {
  const f = document.querySelector("iframe.splat-viewer");
  let camera = "";
  if (f && f.contentWindow) {
    const id = Math.random().toString(36).slice(2);
    camera = await new Promise((resolve) => {
      const on = (e) => {
        const d = e.data;
        if (d && d.type === "camera" && d.id === id) { done(JSON.stringify(d)); }
      };
      const done = (v) => { window.removeEventListener("message", on); resolve(v); };
      window.addEventListener("message", on);
      f.contentWindow.postMessage({type: "get_camera", id}, "*");
      setTimeout(() => done(""), 3000);
    });
  }
  return [camera, ...args.slice(1)];
}"""


# Moves the viewer to a saved view (`viewer.py`, "set_camera"); the input is JSON. A
# future laid over the viewer is removed, since it belongs to the view before.
SET_CAMERA_JS = """(text) => {
  const f = document.querySelector("iframe.splat-viewer");
  if (!text || !f || !f.contentWindow) return;
  try {
    f.contentWindow.postMessage({type: "hide_image"}, "*");
    f.contentWindow.postMessage({type: "set_camera", ...JSON.parse(text)}, "*");
  } catch (err) { console.warn("set_camera", err); }
}"""

# After a delete: remove the overlay if it showed the deleted future ({"hide": true}).
HIDE_FUTURE_JS = """(text) => {
  const f = document.querySelector("iframe.splat-viewer");
  if (!text || !f || !f.contentWindow || !text.includes('"hide"')) return;
  f.contentWindow.postMessage({type: "hide_image"}, "*");
}"""

# A gallery tile in the viewer: move to its view, then lay the future over it (`src`),
# or remove the overlay for the plain render. The input is JSON (`_viewer_overlay`).
SHOW_FUTURE_JS = """(text) => {
  const f = document.querySelector("iframe.splat-viewer");
  if (!text || !f || !f.contentWindow) return;
  try {
    const d = JSON.parse(text), w = f.contentWindow;
    w.postMessage({type: "set_camera", position: d.position, target: d.target}, "*");
    w.postMessage(d.src ? {type: "show_image", src: d.src, caption: d.caption}
                        : {type: "hide_image"}, "*");
  } catch (err) { console.warn("show_future", err); }
}"""


def on_preview(camera_json, scene_id, prompt, strength, seed, number, placing, negative=None):
    version = _version(scene_id, number)
    if version is None:
        raise gr.Error("Pick a finished scene first.")
    setup = _setup(scene_id)
    if negative is not None:  # the Preview tool's negative prompt, kept per scene
        try:
            custom = instructions.clean({**setup.instructions, "preview_negative": negative})
        except instructions.InstructionError as e:
            raise gr.Error(str(e)) from e
        if custom != setup.instructions:
            setup.instructions = custom
            _save_setup(scene_id, setup)
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Describe the look first, e.g. 'same street in autumn with rain gardens'.")
    try:
        camera = camera_from_viewer(camera_json)
    except CameraError as e:
        raise gr.Error(str(e)) from e
    job = PREVIEWS.create_preview(
        prompt, version.path, camera.to_dict(), seed=int(seed or 0), strength=float(strength),
        instructions=instructions.resolve(setup.instructions),
    )  # fmt: skip
    if job.status == "failed":
        gr.Warning(f"The preview could not start. {job.error['message']}")
        return f"**Could not start.** {job.error['message']} {job.error.get('tip', '')}", None
    note = " The object being placed is not part of it." if placing else ""
    gr.Info("Preview started. It takes a few minutes; the image appears below the button.")
    return (
        f"⚙️ Rendering this view of v{version.number} and restyling it (job {job.job_id}).{note}",
        job.job_id,
    )


def _preview_outputs(job: Job | None, notify: bool = True) -> list:
    """Outputs for a preview's current state: status, preview_job, column, preview, render.

    `preview_job` is cleared once the job is final; while it runs it is left unchanged.
    """
    rest = [gr.skip()] * 3
    if job is None:
        return ["The preview job was not found.", None, *rest]
    if job.status == "failed":
        err = job.error or {}
        msg = f"❌ **Preview failed.** {err.get('message', '')} {err.get('tip', '')}".strip()
        if notify:
            gr.Warning(f"Preview failed. {err.get('message', '')}".strip())
        return [msg, None, *rest]
    if job.status != "done":
        label = "⏳ Waiting for a GPU" if job.status == "queued" else "⚙️ Restyling"
        elapsed = format_elapsed(job.elapsed(PREVIEWS.clock()))
        return [f"{label}: **{job.summary}** ({elapsed})", gr.skip(), *rest]
    try:
        styled = STORE.download(f"{job.scene_path}/preview.png")
        render = STORE.download(f"{job.scene_path}/render.png")
    except Exception as e:
        return [f"**The preview could not be fetched** ({type(e).__name__}).", None, *rest]
    if styled is None:
        return ["**The preview image is missing.**", None, *rest]
    seconds = job.run.get("seconds")
    took = f" in {format_elapsed(seconds)}" if seconds else ""
    if notify:
        gr.Info("Preview ready: see the Preview tab.")
    return [
        f"✅ Preview of **{job.summary}** ready{took}. The scene is unchanged.",
        None,
        gr.update(visible=True),
        str(styled),
        str(render) if render else None,
    ]


def on_preview_tick(preview_job: str | None):
    """Poll the preview job. Outputs: status, preview_job, column, preview, render."""
    if not preview_job:
        return [gr.skip()] * 5
    job = PREVIEWS.get(preview_job)
    return _preview_outputs(PREVIEWS.refresh(job) if job else None)


def _scene_previews(scene_id: str | None) -> list[Job]:
    """The scene's previews, newest first; unfinished ones are updated from Modal, so a
    preview that finished after its page was closed shows as done (reopen-previews)."""
    if not scene_id:
        return []
    jobs = prompt_log.scene_previews(scene_id, PREVIEWS.list(refresh=False))
    return [PREVIEWS.refresh(j) if j.status not in ("done", "failed") else j for j in jobs]


def on_previews(scene_id: str | None):
    """The Preview sub-tab's list of earlier previews. Outputs: table, row job ids."""
    rows, ids = prompt_log.preview_rows(scene_id or "", _scene_previews(scene_id))
    return gr.update(value=rows), ids


def on_pick_preview(ids: list[str] | None, evt: gr.SelectData):
    """A click on an earlier preview: show a finished one, follow a running one.

    Outputs: status, preview_job, column, preview, render.
    """
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not ids or not 0 <= row < len(ids):
        return [gr.skip()] * 5
    job = PREVIEWS.get(ids[row])
    job = PREVIEWS.refresh(job) if job else None
    out = _preview_outputs(job, notify=False)
    if job is not None and job.status != "done":
        out[2] = gr.update(visible=False)  # the images shown belong to another preview
    if job is not None and job.status not in ("done", "failed"):
        out[1] = job.job_id  # the 5 s poll follows it from here
    return out


# --- Futures (add-viewpoint-futures) ---------------------------------------------------

FUTURES = futures.FutureIndex(PREVIEWS)
SETUPS: dict[str, viewpoints.Setup] = {}  # per scene; this Space is the only writer
RUNNING: dict[str, Job] = {}  # futures this Space started, polled every second
WARM_EVERY_S = 4 * 60  # a warm-up call keeps a container up for WARM_S (5 min) after it
_last_warm: dict[str, float] = {}  # Modal class -> last warm-up
FUTURE_HINT = (
    "Futures are restyled images made from saved views. Move the viewer to a spot you "
    "want to change and click **Save current view**, or just type a prompt and click "
    "**Make future**: the current view is then saved first. Click a future in the gallery "
    "to see it in the viewer, laid over the 3D scene at its view."
)


AREA_COLOUR = "#ff3b30"
AREA_HINT = (
    "Paint where something new should go or what should change (a willow here, a wetland "
    "on this meadow). Kontext edits a crop around the area and only the area changes. "
    "Leave it empty to change the whole view; not together with 'Object to change'."
)


def _local(*parts: str) -> Path:
    return Path(STORE.file_dir).joinpath(*parts)


def _setup(scene_id: str) -> viewpoints.Setup:
    if scene_id not in SETUPS:
        SETUPS[scene_id] = viewpoints.load_setup(STORE, scene_id)
    return SETUPS[scene_id]


def _save_setup(scene_id: str, setup: viewpoints.Setup) -> None:
    viewpoints.save_setup(STORE, scene_id, setup)
    SETUPS[scene_id] = setup


def _placeholder_image() -> Path:
    """Grey tile for a viewpoint whose render is not stored yet."""
    path = _local("thumbs", "pending.png")
    if not path.is_file():
        from PIL import Image, ImageDraw

        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (512, 320), (31, 36, 48))
        ImageDraw.Draw(img).text((180, 150), "rendering this view…", fill=(201, 207, 219))
        img.save(path)
    return path


def _view_render(scene_id: str, vp_id: str) -> Path | None:
    """Local copy of a viewpoint's stored render, or None if it has none (yet)."""
    try:
        return STORE.download(f"{viewpoints.view_prefix(scene_id, vp_id)}/render.png")
    except Exception:
        return None


def _view_thumb(scene_id: str, vp_id: str) -> Path:
    dest = _local("thumbs", "views", scene_id, f"{vp_id}.jpg")
    if dest.is_file():
        return dest
    src = _view_render(scene_id, vp_id)
    return futures.thumbnail(src, dest) if src else _placeholder_image()


def _future_image(job: Job) -> Path | None:
    """The future's full image: the JPEG the worker returned, else preview.png."""
    path = _local("futures", f"{job.job_id}.jpg")
    if path.is_file():
        return path
    if job.job_id in PREVIEWS.images:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PREVIEWS.images[job.job_id])
        return path
    try:
        return STORE.download(f"previews/{job.job_id}/preview.png")
    except Exception:
        return None


def _changed_area(job: Job | None, image: Path | None):
    """The "Changed area" image of a masked future (add-object-masking): the future with
    the unchanged part dimmed. A hidden, empty update for other tiles."""
    hidden = gr.update(value=None, visible=False)
    if job is None or image is None or not futures.is_limited(job.params):
        return hidden
    dest = _local("thumbs", "changed", f"{job.job_id}.jpg")
    if not dest.is_file():
        import io

        from PIL import Image

        try:
            raw = PREVIEWS.masks.get(job.job_id)
            if raw is None:
                raw = STORE.download(f"previews/{job.job_id}/mask.png").read_bytes()
            with Image.open(image) as img, Image.open(io.BytesIO(raw)) as mask:
                out = masking.changed_area(img, mask)
        except Exception as e:  # an old or failed future without mask.png
            print(f"changed area of {job.job_id}: {e!r}")
            return hidden
        out.thumbnail((futures.THUMB_SIDE, futures.THUMB_SIDE))
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.save(dest, format="JPEG", quality=85)
    return gr.update(value=str(dest), visible=True)


def _detections_image(job: Job | None):
    """The labelled detection image of a masked future (add-detection-preview), captioned
    "Detected: 3 × chair, 1 × stool". A hidden, empty update for other tiles and for masked
    futures made before it existed."""
    hidden = gr.update(value=None, visible=False)
    if job is None or not job.params.get("object_query"):
        return hidden
    path = _local("futures", "detections", f"{job.job_id}.jpg")
    if not path.is_file():
        try:
            raw = PREVIEWS.detections.get(job.job_id)
            if raw is None:
                raw = STORE.download(f"previews/{job.job_id}/detections.jpg").read_bytes()
        except Exception as e:  # made before add-detection-preview
            print(f"detections of {job.job_id}: {e!r}")
            return hidden
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    caption = futures.detections_caption(job) or "Detected objects"
    return gr.update(value=str(path), visible=True, label=caption)


def _future_thumb(job: Job) -> Path | None:
    dest = _local("thumbs", "futures", f"{job.job_id}.jpg")
    if dest.is_file():
        return dest
    src = _future_image(job)
    return futures.thumbnail(src, dest) if src else None


def _future_job(job_id: str | None) -> Job | None:
    if not job_id:
        return None
    return FUTURES.final.get(job_id) or RUNNING.get(job_id) or PREVIEWS.get(job_id)


def _strip(scene_id: str | None):
    """The viewpoint strip: (gallery value, viewpoint ids in the same order)."""
    if not scene_id:
        return [], []
    vps = _setup(scene_id).ordered()
    items = [(str(_view_thumb(scene_id, v.id)), f"{v.name} · v{v.version}") for v in vps]
    return items, [v.id for v in vps]


def _gallery(scene_id: str | None, vp_id: str | None):
    """A viewpoint's gallery: (value, tile keys, label); the plain render comes first."""
    vp = _setup(scene_id).get(vp_id) if scene_id else None
    if vp is None:
        return [], [], "Futures (pick a saved view)"
    items = [(str(_view_thumb(scene_id, vp.id)), "plain render")]
    keys = ["render"]
    for job in futures.viewpoint_futures(FUTURES.jobs(), vp.id):
        thumb = _future_thumb(job)
        if thumb:
            items.append((str(thumb), futures.caption(job)))
            keys.append(job.job_id)
    n = len(keys) - 1
    return items, keys, f'Futures of "{vp.name}" ({n}; click to show, pick two to compare)'


def _last_used(scene_id: str, setup: viewpoints.Setup) -> str | None:
    """The viewpoint of the newest future of the scene, else the first viewpoint."""
    ids = {v.id for v in setup.viewpoints}
    for job in sorted(FUTURES.jobs(), key=lambda j: j.created_at, reverse=True):
        if futures.is_future(job) and job.params["viewpoint"] in ids:
            return job.params["viewpoint"]
    ordered = setup.ordered()
    return ordered[0].id if ordered else None


def _starter_choices(setup: viewpoints.Setup) -> list[str]:
    return ["none", *(s.label for s in setup.starters)]


def _instruction_texts(custom: dict | None) -> list[str]:
    """The Futures settings' instruction texts, in `instructions.FUTURE_KEYS` order."""
    texts = instructions.resolve(custom)
    return [texts[k] for k in instructions.FUTURE_KEYS]


def on_preview_negative(scene_id):
    """The scene's negative prompt for the Preview tool (the default without a scene)."""
    custom = _setup(scene_id).instructions if scene_id else None
    return instructions.resolve(custom)["preview_negative"]


def on_sent_preview(scene_id, starter_label, prompt, object_query, area_png):
    """What the editor will get for the current inputs (the "Sent to the editor" line)."""
    if not scene_id:
        return ""
    setup = _setup(scene_id)
    starter = next((s for s in setup.starters if s.label == starter_label), None)
    has_area = area_png is not None and bool(area.from_png(area_png).any())
    return futures.sent_preview(starter, prompt, setup.editor, object_query, has_area,
                                setup.instructions)  # fmt: skip


def _editor_class(scene_id: str | None) -> str:
    """Modal class of the scene's editor (the default editor without a scene)."""
    key = _setup(scene_id).editor if scene_id else None
    return models.editor_spec(key).modal_class


def _warm_up(class_name: str = "PreviewRenderer") -> None:
    """Start the editor's container in the background (at most every WARM_EVERY_S each)."""
    now = time.time()
    if now - _last_warm.get(class_name, 0.0) < WARM_EVERY_S:
        return
    _last_warm[class_name] = now

    def call() -> None:
        try:
            PREVIEWS.client.warm_preview(class_name)
        except Exception as e:  # no Modal locally, or a deploy without `warm`
            print(f"warm-up skipped: {e!r}")

    threading.Thread(target=call, daemon=True).start()


def on_futures_open(scene_id: str | None, vp_id: str | None):
    """Fill the Futures sub-tab; the viewer keeps its camera.

    Outputs: intro, strip, vp ids, selected vp, name, starter radio, starters text,
    editor, gallery, tile keys, picked, then one text per editor instruction.
    """
    _warm_up(_editor_class(scene_id))
    if not scene_id:
        empty = [[], [], None, "", gr.update(choices=["none"], value="none"), "", gr.skip()]
        return ["Pick a finished scene first.", *empty, gr.update(value=[]), [], [],
                *_instruction_texts(None)]  # fmt: skip
    setup = _setup(scene_id)
    if setup.get(vp_id) is None:
        vp_id = _last_used(scene_id, setup)
    strip, ids = _strip(scene_id)
    items, keys, label = _gallery(scene_id, vp_id)
    vp = setup.get(vp_id)
    intro = FUTURE_HINT if not ids else (
        f"Selected view: **{vp.name}** (v{vp.version}). Pick a view to move the viewer "
        "there. **Make future** uses the selected view while the viewer is at it, and "
        "saves the current view as a new one when you have moved away."
    )  # fmt: skip
    choices = _starter_choices(setup)
    return [
        intro, strip, ids, vp_id, vp.name if vp else "",
        gr.update(choices=choices, value="none"),
        viewpoints.starters_text(setup.starters), setup.editor,
        gr.update(value=items, label=label), keys, [],
        *_instruction_texts(setup.instructions),
    ]  # fmt: skip


def _viewer_camera(vp: viewpoints.Viewpoint) -> dict:
    m = vp.camera["c2w"]
    return {"position": [m[0][3], m[1][3], m[2][3]], "target": viewpoints.viewer_target(vp)}


def on_pick_viewpoint(scene_id, number, ids, obj_job, placing, evt: gr.SelectData):
    """A click on a saved view: select it, show its gallery and move the viewer there
    (reopening the viewer on the view's version if another version is shown).

    Outputs: selected vp, name, intro, gallery, tile keys, picked, camera message, then
    `SCENE_OUTPUTS`.
    """
    i = evt.index if isinstance(evt.index, int) else evt.index[0]
    vp = _setup(scene_id).get(ids[i]) if scene_id and ids and 0 <= i < len(ids) else None
    if vp is None:
        return [gr.skip()] * 17
    items, keys, label = _gallery(scene_id, vp.id)
    move = json.dumps(_viewer_camera(vp))
    head = [vp.id, vp.name, f"Selected view: **{vp.name}** (v{vp.version}).",
            gr.update(value=items, label=label), keys, []]  # fmt: skip
    if vp.version == number:
        return [*head, move, *[gr.skip()] * 10]
    out = list(scene_view(scene_id, vp.version, obj_job, placing))
    version = _version(scene_id, vp.version)
    cam = _viewer_camera(vp)
    out[2] = scene_html(STORE, version.path if version else None, camera=Camera(
        tuple(cam["position"]), tuple(cam["target"])))  # fmt: skip
    return [*head, "", *out]


def _camera_and_target(camera_json: str | None) -> tuple[dict, list[float] | None]:
    try:
        camera = camera_from_viewer(camera_json).to_dict()
    except CameraError as e:
        raise gr.Error(str(e)) from e
    target = json.loads(camera_json).get("target")
    return camera, target if isinstance(target, list) and len(target) == 3 else None


def _version_path(scene_id: str, number: int) -> str:
    """Dataset folder of a scene version (no Hub call, unlike `_version`)."""
    return f"scenes/{scene_id}/v{int(number)}"


def _start_view_render(scene_id: str, vp: viewpoints.Viewpoint) -> dict | None:
    """Render a new viewpoint on Modal; returns the pending call, or None if it failed."""
    try:
        call = PREVIEWS.client.submit_view(
            _version_path(scene_id, vp.version),
            vp.camera,
            viewpoints.view_prefix(scene_id, vp.id),
            class_name=_editor_class(scene_id),
        )
    except Exception as e:
        print(f"view render not started: {e!r}")
        return None
    return {"call": call, "scene": scene_id, "vp": vp.id}


def on_save_view(camera_json, scene_id, name, number):
    """Save the viewer's camera as a viewpoint of the shown version and render it once.

    Outputs: status, strip, vp ids, selected vp, name, intro, gallery, tile keys, pending
    view render, timer.
    """
    if not scene_id or not number:
        raise gr.Error("Pick a finished scene first.")
    camera, target = _camera_and_target(camera_json)
    setup = _setup(scene_id)
    try:
        vp = viewpoints.add_viewpoint(setup, name, number, camera, target)
    except viewpoints.ViewpointError as e:
        raise gr.Error(str(e)) from e
    _save_setup(scene_id, setup)
    pending = _start_view_render(scene_id, vp)
    strip, ids = _strip(scene_id)
    items, keys, label = _gallery(scene_id, vp.id)
    msg = f'Saved this view as **{vp.name}** (v{number}). ' + (
        "Rendering it on the GPU for its thumbnail…" if pending else
        "The GPU could not be reached; the view is rendered with its first future."
    )  # fmt: skip
    return [
        msg, strip, ids, vp.id, vp.name, f"Selected view: **{vp.name}** (v{number}).",
        gr.update(value=items, label=label), keys, pending, gr.Timer(active=True),
    ]  # fmt: skip


def on_manage_viewpoint(action: str, scene_id, vp_id, name):
    """Rename, move up/down or remove the selected viewpoint.

    Outputs: status, strip, vp ids, selected vp, name, gallery, tile keys.
    """
    if not scene_id:
        raise gr.Error("Pick a finished scene first.")
    setup = _setup(scene_id)
    try:
        if action == "rename":
            vp = viewpoints.rename_viewpoint(setup, vp_id, name)
            msg = f"Renamed to **{vp.name}**."
        elif action in ("up", "down"):
            viewpoints.move_viewpoint(setup, vp_id, -1 if action == "up" else 1)
            msg = "Moved."
        else:
            vp = viewpoints.remove_viewpoint(setup, vp_id)
            vp_id = None
            msg = f"Removed **{vp.name}** from the list; its futures are kept in the dataset."
    except viewpoints.ViewpointError as e:
        raise gr.Error(str(e)) from e
    _save_setup(scene_id, setup)
    strip, ids = _strip(scene_id)
    items, keys, label = _gallery(scene_id, vp_id)
    vp = setup.get(vp_id)
    return [msg, strip, ids, vp_id, vp.name if vp else "",
            gr.update(value=items, label=label), keys]  # fmt: skip


def _area_key(vp_id: str | None, parent_id: str | None) -> str | None:
    """Which start image an area is drawn on: the parent future's, else the view's."""
    if parent_id:
        return f"future:{parent_id}"
    return f"view:{vp_id}" if vp_id else None


def on_area_load(scene_id, vp_id, parent_id):
    """Show the next future's start image in the area editor, with no area drawn
    (add-area-layout). Outputs: editor, start key, area PNG, note."""
    empty = [None, None, None, AREA_HINT]
    if not scene_id:
        return empty
    job = _future_job(parent_id)
    if job is not None:
        image, what = _future_image(job), f"the future *{futures.caption(job)}*"
    elif vp_id and _setup(scene_id).get(vp_id) is not None:
        image, what = _view_render(scene_id, vp_id), f"**{_setup(scene_id).get(vp_id).name}**"
    else:
        return empty
    if image is None:
        return [None, None, None, "This view has no stored image yet; make one future first."]
    value = {"background": str(image), "layers": [], "composite": str(image)}
    note = f"Paint on {what}. {AREA_HINT}"
    return [value, _area_key(vp_id, parent_id), None, note]


def on_area_change(value, area_on):
    """The painted area as a PNG (None when nothing is painted). Outputs: area PNG, note."""
    if not value or not area_on:
        return None, AREA_HINT
    layers = list(value.get("layers") or [])
    background = value.get("background")
    size = None
    if background is not None and getattr(background, "ndim", 0) >= 2:
        size = (background.shape[1], background.shape[0])
    hard = area.from_layers(layers, size)
    if not hard.any():
        return None, AREA_HINT
    note = f"Area: {hard.mean():.0%} of the view. Only it changes (FLUX.1 Kontext editor)."
    if hard.mean() < area.MIN_COVER:
        note = "The area is too small; paint a larger part of the view."
    return area.to_png(hard), note


def on_make_future(
    camera_json, scene_id, starter_label, prompt, object_query, seed, speed, vp_id, parent_id,
    number, area_png=None, area_on=None,
):  # fmt: skip
    """Start a future. From the selected view while the viewer is at it (or from the
    parent future's view when building on one); otherwise the current view is saved as
    a new viewpoint first, in the same GPU job.

    Outputs: status, running future, selected vp, parent, parent note, strip, vp ids,
    name, gallery, tile keys, timer.
    """
    if not scene_id or not number:
        raise gr.Error("Pick a finished scene first.")
    setup = _setup(scene_id)
    parent = _future_job(parent_id)
    note = ""
    if parent is not None:
        vp = setup.get(parent.params.get("viewpoint"))
        if vp is None:
            raise gr.Error("The view of that future was removed; pick another future.")
    else:
        camera, target = _camera_and_target(camera_json)
        vp = setup.get(vp_id)
        if vp is None or not viewpoints.same_view(vp.camera, camera):
            vp = viewpoints.add_viewpoint(setup, None, number, camera, target)
            _save_setup(scene_id, setup)
            note = f" Saved the current view as **{vp.name}** (v{number})."
    hard = area.from_png(area_png) if area_png else None
    if hard is not None and hard.any() and area_on != _area_key(vp.id, parent_id):
        raise gr.Error(
            "The area was drawn on another image than this future starts from. Click the "
            "saved view (or 'Start from this') again and redraw the area, or clear it."
        )
    starter = next((s for s in setup.starters if s.label == starter_label), None)
    try:
        text = futures.compose_prompt(starter, prompt)
        params = futures.future_params(
            vp, viewpoints.view_prefix(scene_id, vp.id), starter, prompt, parent, setup.editor,
            speed, object_query, area_mask=hard, texts=setup.instructions,
        )  # fmt: skip
    except futures.FutureError as e:
        raise gr.Error(str(e)) from e
    job = PREVIEWS.create_preview(
        text, _version_path(scene_id, vp.version), vp.camera, seed=int(seed or 0), future=params,
        input_path=f"previews/{parent.job_id}/preview.png" if parent else None,
        area_png=area_png if params.get("area") else None,
    )  # fmt: skip
    strip, ids = _strip(scene_id)
    items, keys, label = _gallery(scene_id, vp.id)
    gallery = gr.update(value=items, label=label)
    if job.status == "failed":
        err = job.error or {}
        return [f"❌ **Could not start.** {err.get('message', '')}{note}", None, vp.id,
                parent_id, gr.skip(), strip, ids, vp.name, gallery, keys,
                gr.Timer(active=False)]  # fmt: skip
    RUNNING[job.job_id] = job
    built = f" on future `{parent.job_id}`" if parent else ""
    return [
        f"⏳ Making a future of **{vp.name}**{built}: *{job.summary}* (job {job.job_id}).{note}",
        job.job_id, vp.id, None, "", strip, ids, vp.name, gallery, keys,
        gr.Timer(active=True),
    ]  # fmt: skip


DETECT_TIMEOUT_S = 240  # a cold Kontext container takes about 90 s to load


def on_detect_objects(scene_id, vp_id, object_query):
    """ "Detect objects": find the object in the selected view's plain render, without
    making a future (add-detection-preview). Outputs: status, detected objects image."""
    if not scene_id:
        raise gr.Error("Pick a finished scene first.")
    setup = _setup(scene_id)
    vp = setup.get(vp_id)
    if vp is None:
        raise gr.Error("Pick a saved view first: detection runs on its plain render.")
    try:
        query = futures.detect_query(setup.editor, object_query)
    except futures.FutureError as e:
        raise gr.Error(str(e)) from e
    class_name = models.editor_spec(setup.editor).modal_class
    try:
        call = PREVIEWS.client.submit_detect(
            viewpoints.view_prefix(scene_id, vp.id), query, class_name
        )
    except Exception as e:
        raise gr.Error(f"Could not start the detection: {e}") from e
    hidden = gr.update(value=None, visible=False)
    yield f"⏳ Looking for **{query}** in the plain render of **{vp.name}**…", hidden
    start, result = time.time(), None
    while True:
        try:
            state, result = PREVIEWS.client.poll(call)
        except Exception as e:
            state, result = "finished", {"status": "failed", "error": {"message": repr(e)}}
        if state == "finished":
            break
        if time.time() - start > DETECT_TIMEOUT_S:
            yield "❌ The detection took too long (no free GPU?). Try again.", hidden
            return
        time.sleep(0.5)
    result = result or {}
    if result.get("status") != "done" or not result.get("detections_jpeg"):
        err = result.get("error") or {}
        msg = f"{err.get('message', 'The detection failed.')} {err.get('tip', '')}".strip()
        yield f"❌ {msg}", hidden
        return
    # A new file name each time, so the browser does not show an earlier image.
    path = _local("detect", scene_id, f"{vp.id}-{int(time.time() * 1000)}.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result["detections_jpeg"])
    caption = annotate.detections_caption((result.get("mask") or {}).get("labels") or [])
    yield (
        f"🔎 {caption} in the plain render of **{vp.name}** ({time.time() - start:.0f} s). "
        "**Make future** from this view uses the same detector on the same image "
        "(not when building on a future).",
        gr.update(value=str(path), visible=True, label=caption),
    )


def _save_in_background(job: Job) -> None:
    def save() -> None:
        try:
            PREVIEWS.save(job)
        except Exception as e:  # the next listing refreshes it from Modal again
            print(f"saving job.json of {job.job_id} failed: {e!r}")

    threading.Thread(target=save, daemon=True).start()


def on_futures_tick(job_id, pending, scene_id, vp_id):
    """Poll the running future and a new viewpoint's render (every second).

    Outputs: status, running future, pending view render, result, result details,
    "Start from this", gallery, tile keys, strip, timer, shown tile, changed area,
    detections.
    """
    out = [gr.skip()] * 13
    view_pending, running = bool(pending), bool(job_id)
    if pending:
        try:
            state, result = PREVIEWS.client.poll(pending["call"])
        except Exception as e:
            state, result = "finished", {"status": "failed", "error": {"message": repr(e)}}
        if state == "finished":
            out[2], view_pending = None, False
            result = result or {}
            if result.get("status") == "done" and result.get("image_jpeg"):
                dest = _local("thumbs", "views", pending["scene"], f"{pending['vp']}.jpg")
                raw = dest.with_suffix(".full.jpg")
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_bytes(result["image_jpeg"])
                futures.thumbnail(raw, dest)
            else:
                msg = (result.get("error") or {}).get("message", "")
                gr.Warning(f"The view could not be rendered now; its first future will. {msg}")
            if pending["scene"] == scene_id:
                out[8] = _strip(scene_id)[0]
                if pending["vp"] == vp_id:
                    items, keys, label = _gallery(scene_id, vp_id)
                    out[6], out[7] = gr.update(value=items, label=label), keys
    if job_id:
        job = _future_job(job_id)
        if job is None:
            out[0], out[1], running = "The future's job was not found.", None, False
        else:
            job = PREVIEWS.refresh(job, save=False)
            RUNNING[job.job_id] = job
            if job.status in ("done", "failed"):
                running = False
            if job.status == "failed":
                RUNNING.pop(job.job_id, None)
                _save_in_background(job)
                err = job.error or {}
                gr.Warning(f"The future failed. {err.get('message', '')}")
                out[0] = (f"❌ **The future failed.** {err.get('message', '')} "
                          f"{err.get('tip', '')} Click **Make future** to try again.")  # fmt: skip
                out[1] = None
            elif job.status == "done":
                RUNNING.pop(job.job_id, None)
                FUTURES.put(job)
                image = _future_image(job)  # the returned JPEG: no Hub round trip
                _save_in_background(job)
                took = format_elapsed(job.elapsed(PREVIEWS.clock()))
                setup = _setup(scene_id) if scene_id else viewpoints.Setup()
                vp = setup.get(job.params.get("viewpoint"))
                out[0] = f"✅ Future ready {took} after the click."
                out[1] = None
                out[3] = str(image) if image else None
                out[4] = futures.details_markdown(job, vp.name if vp else None)
                out[5] = gr.update(interactive=True)
                out[10] = job.job_id  # "Start from this" builds on the image just shown
                out[11] = _changed_area(job, image)
                out[12] = _detections_image(job)
                if scene_id and vp:
                    items, keys, label = _gallery(scene_id, vp.id)
                    out[6], out[7] = gr.update(value=items, label=label), keys
                    out[8] = _strip(scene_id)[0]  # a new view's render now exists
            else:
                label = "⏳ Waiting for a GPU" if job.status == "queued" else "⚙️ Restyling"
                elapsed = format_elapsed(job.elapsed(PREVIEWS.clock()))
                out[0] = f"{label}: **{job.summary}** ({elapsed})"
    out[9] = gr.Timer(active=running or view_pending)
    return out


def _tile_image(scene_id: str, vp_id: str, key: str) -> tuple[Path | None, str]:
    """Full image and details of a gallery tile ("render" or a future's job id)."""
    vp = _setup(scene_id).get(vp_id)
    if key == "render":
        what = f"**Plain render** of **{vp.name}** (v{vp.version})" if vp else "Plain render"
        return _view_render(scene_id, vp_id), what
    job = _future_job(key)
    if job is None:
        return None, "This future was not found."
    return _future_image(job), futures.details_markdown(job, vp.name if vp else None)


def _viewer_overlay(scene_id: str, vp_id: str, key: str, image: Path | None) -> str:
    """Message for SHOW_FUTURE_JS: move the viewer to the view and lay the future over it
    (no image for the plain render, which is the 3D view itself)."""
    vp = _setup(scene_id).get(vp_id) if scene_id and vp_id else None
    if vp is None:
        return ""
    msg = _viewer_camera(vp)
    job = _future_job(key) if key != "render" else None
    if job is not None and image is not None:
        dest = _local("thumbs", "viewer", f"{job.job_id}.jpg")
        msg["src"] = futures.data_url(futures.thumbnail(image, dest, futures.VIEWER_SIDE))
        msg["caption"] = f"{vp.name}: {futures.caption(job, 60)}"
    return json.dumps(msg)


def on_pick_tile(scene_id, vp_id, keys, picked, evt: gr.SelectData):
    """A click on a gallery tile: show it large and in the viewer, and keep it as one of
    the two to compare.

    Outputs: result, result details, "Start from this", shown tile, picked, compare note,
    viewer message, changed area, detections.
    """
    i = evt.index if isinstance(evt.index, int) else evt.index[0]
    if not keys or not 0 <= i < len(keys):
        return [gr.skip()] * 9
    key = keys[i]
    image, details = _tile_image(scene_id, vp_id, key)
    overlay = _viewer_overlay(scene_id, vp_id, key, image)
    picked = [k for k in (picked or []) if k != key][-1:] + [key]
    names = ["plain render" if k == "render" else f"`{k}`" for k in picked]
    note = "Picked to compare: " + " and ".join(names)
    if len(picked) < 2:
        note += " (pick one more)"
    job = _future_job(key) if key != "render" else None
    return [str(image) if image else None, details,
            gr.update(interactive=key != "render"), key, picked, note, overlay,
            _changed_area(job, image), _detections_image(job)]  # fmt: skip


_DELETE_LOCK = threading.Lock()
_DELETE_SEEN: set[str] = set()  # delete requests ("<index>:<time>") already handled


def on_delete_future(text, last, scene_id, vp_id, keys, picked, shown, parent):
    """Trash button on a gallery tile ("<index>:<time>" from the page): delete that future.
    `last` is the request handled before; the same request again does nothing.

    Outputs: gallery, tile keys, picked, compare note, result, result details, "Start from
    this", shown tile, viewer message, parent, parent note, parent group, last request.
    """
    try:
        i = int(str(text).split(":", 1)[0])
    except ValueError:
        return [gr.skip()] * 13
    with _DELETE_LOCK:  # also catches a duplicate sent before `last` reached the page
        if text == last or text in _DELETE_SEEN:
            return [gr.skip()] * 13
        _DELETE_SEEN.add(text)
    if not keys or not 0 < i < len(keys):
        raise gr.Error("This future is no longer in the gallery; pick the view again.")
    job_id = keys[i]
    try:
        job = futures.delete_future(FUTURES, job_id)
    except futures.FutureError as e:
        raise gr.Error(str(e)) from e
    PREVIEWS.images.pop(job_id, None)
    for path in (_local("futures", f"{job_id}.jpg"), _local("thumbs", "futures", f"{job_id}.jpg"),
                 _local("thumbs", "viewer", f"{job_id}.jpg")):  # fmt: skip
        path.unlink(missing_ok=True)
    gr.Info(f'Deleted the future "{futures.caption(job)}".')
    items, keys, label = _gallery(scene_id, vp_id)
    picked = [k for k in (picked or []) if k != job_id]
    names = ["plain render" if k == "render" else f"`{k}`" for k in picked]
    note = "Picked to compare: " + " and ".join(names) if picked else ""
    out = [gr.update(value=items, label=label), keys, picked, note]
    if shown == job_id:
        out += [None, "", gr.update(interactive=False), None, json.dumps({"hide": True})]
    else:
        out += [gr.skip()] * 4 + [""]
    if parent == job_id:
        out += [None, "", gr.update(visible=False)]
    else:
        out += [gr.skip()] * 3
    return [*out, text]


def on_start_from(shown: str | None):
    """ "Start from this": the next Make restyles the shown future's image.

    Outputs: parent, parent note, clear button.
    """
    job = _future_job(shown) if shown and shown != "render" else None
    if job is None:
        raise gr.Error("Pick a future in the gallery first.")
    note = (f"**Building on** *{futures.caption(job)}* (`{job.job_id}`): the next **Make "
            "future** restyles that image with your new prompt.")  # fmt: skip
    return job.job_id, note, gr.update(visible=True)


def on_compare(scene_id, vp_id, picked):
    """Show the two picked tiles side by side. Outputs: row, image A, text A, detections A,
    image B, text B, detections B."""
    if not picked or len(picked) < 2:
        raise gr.Error("Pick two tiles in the gallery first (a future, or the plain render).")
    out = [gr.update(visible=True)]
    for k in picked[-2:]:
        image, details = _tile_image(scene_id, vp_id, k)
        job = _future_job(k) if k != "render" else None
        out += [str(image) if image else None, details, _detections_image(job)]
    return out


def on_save_future_settings(scene_id, text, editor, *texts):
    """Save the scene's starter directions, editor and editor instructions (`texts`, in
    `instructions.FUTURE_KEYS` order). Outputs: status, starter radio."""
    if not scene_id:
        raise gr.Error("Pick a finished scene first.")
    setup = _setup(scene_id)
    try:
        setup.starters = viewpoints.parse_starters(text, setup.starters)
        viewpoints.set_editor(setup, editor)
        given = dict(zip(instructions.FUTURE_KEYS, texts, strict=True))
        setup.instructions = instructions.clean({**setup.instructions, **given})
    except (viewpoints.ViewpointError, instructions.InstructionError) as e:
        raise gr.Error(str(e)) from e
    _save_setup(scene_id, setup)
    _warm_up(_editor_class(scene_id))
    n = len(setup.starters)
    msg = (
        f"Saved {n} starter direction{'s' if n != 1 else ''}; editor "
        f"{models.editor_spec(setup.editor).name}; "
        f"{len(setup.instructions) or 'no'} editor instruction"
        f"{'s' if len(setup.instructions) != 1 else ''} changed from the defaults. Earlier "
        "futures keep the text they were made with."
    )
    return msg, gr.update(choices=_starter_choices(setup), value="none")


def on_download(job_id: str | None, number: int | None, kind: str):
    version = _version(job_id, number)
    if version is None:
        raise gr.Error("Pick a scene version first.")
    try:
        path = download_file(STORE, version, kind)
    except Exception as e:
        raise gr.Error(f"The file could not be fetched ({type(e).__name__}).") from e
    if path is None:
        raise gr.Error(f"Version v{version.number} has no .{kind} file.")
    return gr.update(value=str(path), visible=True)


def on_create(files: list[str] | None, steps, frames, max_side):
    try:
        upload = validate_upload(files)
    except UploadError as e:
        return f"**Upload rejected.** {e}", jobs_table()
    settings = dict(zip(SCENE_KEYS, (steps, frames, max_side), strict=True))
    try:
        job = JOBS.create(upload, settings)
    except ValueError as e:
        return f"**Not started.** {e}", jobs_table()
    if job.status == "failed":
        return (
            f"**Could not start the job.** {job.error['message']} {job.error.get('tip', '')}",
            jobs_table(),
        )
    return f"**Job {job.job_id} started.** It appears in the list below.", jobs_table()


def clicked(button: gr.Button, busy: str, fn, inputs, outputs, **kwargs):
    """Wire ``button`` so it reacts at once: it shows ``busy`` and is disabled while the
    action runs, then gets its label back (also after an error). Returns the action's
    event, so callers can chain ``.then`` on it."""
    label = button.value
    quick = {"queue": False, "show_progress": "hidden"}
    action = button.click(lambda: gr.update(value=busy, interactive=False), None, button, **quick)
    action = action.then(fn, inputs, outputs, **kwargs)
    action.then(lambda: gr.update(value=label, interactive=True), None, button, **quick)
    return action


CSS = """
button { transition: transform 60ms ease, filter 60ms ease; }
button:active:not(:disabled) { transform: scale(0.97); filter: brightness(0.9); }
button:disabled { cursor: progress !important; }
#placement-bridge, #camera-bridge, #futures-camera-bridge,
#futures-overlay-bridge, #futures-delete-bridge,
#tool-options-bridge { display: none !important; }
#futures-gallery .gallery-item { position: relative; }
#futures-gallery .future-trash { position: absolute; top: 8px; right: 8px; z-index: 5;
  width: 30px; height: 30px; padding: 0; border-radius: 6px; display: flex;
  align-items: center; justify-content: center; background: rgba(20,24,32,.85);
  border: 1px solid #6b7385; cursor: pointer; }
#futures-gallery .future-trash:hover { opacity: 1; background: #b42318; }
#history-table td, #history-table th { font-size: 13px; }
#history-table tbody tr, #objects-table tbody tr, #edits-table tbody tr,
#previews-table tbody tr { cursor: pointer; }
#objects-table td:last-child, #edits-table td:last-child { font-weight: 600; white-space: nowrap; }
.tool-off { display: none !important; }
#scene-tool-options label { display: flex; gap: 8px; align-items: center; margin: 4px 0; }
#scene-tool-options input[type="checkbox"] { appearance: auto !important; width: 16px;
  height: 16px; margin: 0; accent-color: var(--color-accent, #f97316); }
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Splat Studio", css=CSS, head=BRIDGE_HEAD + TOOLS_HEAD) as demo:
        gr.Markdown("# Splat Studio\nTurn a video or photo set into a 3D Gaussian splat scene.")
        if not DATASET:
            gr.Markdown("⚠️ `SPLAT_STUDIO_DATASET` is not set: jobs are stored locally only.")
        version, hidden, seen = gr.State(None), gr.State([]), gr.State(None)
        obj_job, placing, floor = gr.State(None), gr.State(False), gr.State(None)
        moving = gr.State(None)  # id of the placed object being moved, if any
        obj_keys, edit_numbers = gr.State([]), gr.State([])
        preview_job, edit_job = gr.State(None), gr.State(None)
        n_photos = gr.State(None)  # photos in the current upload; None for a video

        # Global scene tool choice for the page script; outside the tabs so it always exists.
        tools_json = gr.Textbox(elem_id="tool-options-bridge", show_label=False, container=False)
        with gr.Tabs(selected="new") as tabs:
            # --- New scene: upload, create, jobs -----------------------------------
            with gr.Tab("New scene", id="new"):
                files = gr.File(
                    label="Video or photos",
                    file_count="multiple",
                    file_types=[".mp4", ".mov", ".jpg", ".jpeg", ".png"],
                )
                message = gr.Markdown(HELP)
                sc_default = scene_settings.PRESETS[scene_settings.DEFAULT_PRESET]
                sc_preset = gr.Radio(
                    [*scene_settings.PRESETS, CUSTOM],
                    value=scene_settings.DEFAULT_PRESET,
                    label="Speed and quality",
                    info="Standard is what every scene used so far. Fast and High are not "
                    "yet measured or compared for quality.",
                )
                with gr.Accordion("Scene settings", open=False):
                    sc_settings = [
                        gr.Slider(
                            *scene_settings.LIMITS[k],
                            value=sc_default[k],
                            step=SCENE_SLIDERS[k][1],
                            label=SCENE_SLIDERS[k][0],
                            info=SCENE_SLIDERS[k][2],
                        )
                        for k in SCENE_KEYS
                    ]
                sc_estimate = gr.Markdown(
                    scene_estimate(None, *(sc_default[k] for k in SCENE_KEYS))
                )
                create = gr.Button("Create scene", variant="primary", interactive=False)
                gr.Markdown("## Jobs")
                table = gr.Dataframe(headers=COLUMNS, interactive=False, wrap=True)
                refresh = gr.Button("Refresh")

            # --- Scene: viewer left, tools right ------------------------------------
            with gr.Tab("Scene", id="scene"):
                with gr.Row(equal_height=True):
                    picker = gr.Dropdown(
                        label="Finished scene",
                        choices=[],
                        value=None,
                        allow_custom_value=True,  # choices are filled in after load
                    )
                with gr.Accordion("Scene details (name, notes, delete)", open=False):
                    sc_facts = gr.Markdown(scenes.details_markdown(None, 0))
                    with gr.Row():
                        sc_name = gr.Textbox(
                            label="Name",
                            max_lines=1,
                            max_length=scenes.MAX_NAME,
                            placeholder="e.g. Main street, before the rain gardens",
                            info="Shown in the scene list; empty uses the job id and input.",
                            scale=2,
                        )
                        sc_notes = gr.Textbox(
                            label="Notes",
                            lines=2,
                            max_length=scenes.MAX_NOTES,
                            placeholder="Where and when it was filmed, what it is for, ...",
                            scale=3,
                        )
                    with gr.Row():
                        sc_save = gr.Button("Save details", variant="primary")
                        sc_delete = gr.Button("Delete scene…", variant="stop")
                    with gr.Group(visible=False) as sc_confirm:
                        sc_confirm_text = gr.Markdown()
                        with gr.Row():
                            sc_delete_yes = gr.Button("Yes, delete it", variant="stop")
                            sc_delete_no = gr.Button("Cancel")
                    sc_status = gr.Markdown()
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        viewer = gr.HTML(scene_html(STORE, None))
                        shown = gr.Markdown()
                        with gr.Row():
                            get_ply = gr.Button("Download .ply")
                            get_splat = gr.Button("Download .splat (compressed)")
                        download = gr.File(label="Download", visible=False, interactive=False)
                    with gr.Column(scale=2, elem_id="scene-tools"):
                        with gr.Tabs(elem_id="scene-tool-tabs"):
                            with gr.Tab("History", elem_id="tool-history"):
                                history_table = gr.Dataframe(
                                    headers=HISTORY_COLUMNS, interactive=False, wrap=True,
                                    type="array", elem_id="history-table",
                                    label="Versions (newest first; click a row to open it)",
                                )  # fmt: skip
                                history = gr.Markdown()
                                with gr.Accordion("Prompt log (incl. previews)", open=False):
                                    log_table = gr.Dataframe(
                                        headers=prompt_log.COLUMNS, interactive=False, wrap=True
                                    )
                            with gr.Tab("Objects", elem_id="tool-objects"):
                                objects_table = gr.Dataframe(
                                    headers=OBJECT_COLUMNS, interactive=False, wrap=True,
                                    type="array", elem_id="objects-table",
                                    label="Objects (click a placed one to move or resize it; "
                                    "Hide is view only, downloads keep all)",
                                )  # fmt: skip
                                gr.Markdown("**New object**")
                                obj_prompt = gr.Textbox(
                                    label="Object", placeholder="a mature oak tree"
                                )
                                obj_image = gr.Image(
                                    label="Or start from a photo (optional; the text then "
                                    "only names the object)",
                                    type="filepath", sources=["upload"], height=160,
                                )  # fmt: skip
                                with gr.Row():
                                    obj_seed = gr.Number(label="Seed", value=0, precision=0)
                                    obj_go = gr.Button("Generate object")
                                obj_status = gr.Markdown()
                                with gr.Group(visible=False) as place_group:
                                    obj_ref = gr.Image(
                                        label="Reference image", interactive=False, height=180
                                    )
                                    show_obj = gr.Checkbox(value=True, label="Show object")
                                    px = gr.Slider(-1, 1, value=0, step=0.001, label="Move x")
                                    py = gr.Slider(-1, 1, value=0, step=0.001, label="Move y")
                                    pz = gr.Slider(-1, 1, value=0, step=0.001, label="Height")
                                    pturn = gr.Slider(-180, 180, value=0, step=1, label="Turn (°)")
                                    ptilt_x = gr.Slider(
                                        -180, 180, value=0, step=1, label="Tilt about x (°)"
                                    )
                                    ptilt_y = gr.Slider(
                                        -180, 180, value=0, step=1, label="Tilt about y (°)"
                                    )
                                    psize = gr.Slider(0.01, 1, value=0.25, step=0.001, label="Size")
                                    with gr.Row():
                                        place_ok = gr.Button(
                                            "Save as new version", variant="primary"
                                        )
                                        place_cancel = gr.Button("Cancel")
                            with gr.Tab("Edit scene", elem_id="tool-edit"):
                                edits_table = gr.Dataframe(
                                    headers=EDIT_COLUMNS, interactive=False, wrap=True,
                                    type="array", elem_id="edits-table",
                                    label="Edits (Hide: view the scene before it; Open: show it)",
                                )  # fmt: skip
                                gr.Markdown(
                                    "Change the scene itself from an instruction; the result "
                                    "is a new version. The edit is strongest around the "
                                    "current view, so aim the camera first."
                                )
                                ed_prompt = gr.Textbox(
                                    label="Instruction", placeholder="make the table green"
                                )
                                ed_region = gr.Checkbox(
                                    label="Limit to a region (drag a box in the viewer)",
                                    value=False,
                                )
                                ed_preset = gr.Radio(
                                    [*BUDGET_PRESETS, CUSTOM],
                                    value=DEFAULT_PRESET,
                                    label="Speed and quality",
                                    info="Fast and Balanced are not yet checked against Best "
                                    "for quality; compare on your scene.",
                                )
                                default = BUDGET_PRESETS[DEFAULT_PRESET]
                                with gr.Accordion("Edit settings", open=False):
                                    ed_budget = [
                                        gr.Slider(
                                            *BUDGET_LIMITS[k],
                                            value=default[k],
                                            step=1,
                                            label=label,
                                            info=BUDGET_HINTS[k],
                                        )
                                        for k, label in [
                                            ("n_views", "Views"),
                                            ("steps", "Training steps"),
                                            ("edit_every", "Re-edit every … steps"),
                                            ("denoise_steps", "Editor steps per edit"),
                                        ]
                                    ]
                                ed_estimate = gr.Markdown(
                                    edit_estimate(*(default[k] for k in BUDGET_KEYS))
                                )
                                with gr.Row():
                                    ed_seed = gr.Number(label="Seed", value=0, precision=0)
                                    ed_go = gr.Button("Edit scene", variant="primary")
                                ed_status = gr.Markdown()
                            with gr.Tab("Preview", elem_id="tool-preview") as pv_tab:
                                gr.Markdown(
                                    "Restyle the current view as a 2D image. Aim the camera "
                                    "first; the scene is not changed."
                                )
                                pv_prompt = gr.Textbox(
                                    label="Look",
                                    placeholder="same street in autumn with rain gardens",
                                )
                                pv_strength = gr.Slider(
                                    0.3, 0.95, value=0.7, step=0.05, label="Change (low keeps more)"
                                )
                                pv_negative = gr.Textbox(
                                    value=instructions.DEFAULTS["preview_negative"],
                                    label=instructions.LABELS["preview_negative"][0],
                                    info=instructions.LABELS["preview_negative"][1],
                                    max_length=instructions.MAX_TEXT,
                                )  # fmt: skip
                                with gr.Row():
                                    pv_seed = gr.Number(label="Seed", value=0, precision=0)
                                    pv_go = gr.Button("Preview this view")
                                pv_status = gr.Markdown()
                                with gr.Column(visible=False) as preview_col:
                                    preview_img = gr.Image(
                                        label="Preview (download with the arrow)",
                                        interactive=False, format="png",
                                    )  # fmt: skip
                                    render_img = gr.Image(
                                        label="Rendered view it started from", interactive=False
                                    )
                                pv_table = gr.Dataframe(
                                    headers=prompt_log.PREVIEW_COLUMNS, interactive=False,
                                    wrap=True, type="array", elem_id="previews-table",
                                    label="Earlier previews of this scene (click one to show it)",
                                )  # fmt: skip
                                pv_ids = gr.State([])
                            with gr.Tab("Futures", elem_id="tool-futures") as fu_tab:
                                fu_intro = gr.Markdown(FUTURE_HINT)
                                fu_strip = gr.Gallery(
                                    label="Saved views (click one to go there)",
                                    columns=4, height=200, object_fit="cover",
                                    allow_preview=False, elem_id="futures-views",
                                )  # fmt: skip
                                with gr.Row(equal_height=True):
                                    fu_name = gr.Textbox(
                                        label="View name", max_lines=1,
                                        max_length=viewpoints.MAX_NAME,
                                        placeholder="bridge, east bank, bird's-eye…", scale=3,
                                    )  # fmt: skip
                                    fu_save = gr.Button("Save current view", scale=2)
                                with gr.Accordion("Change the selected view", open=False):
                                    with gr.Row():
                                        fu_rename = gr.Button("Rename (to the name above)")
                                        fu_up = gr.Button("Move up", min_width=80)
                                        fu_down = gr.Button("Move down", min_width=80)
                                        fu_remove = gr.Button("Remove", variant="stop")
                                fu_starter = gr.Radio(
                                    ["none"], value="none", label="Starter direction",
                                    info="Its text goes before yours.",
                                )  # fmt: skip
                                fu_prompt = gr.Textbox(
                                    label="What changes (English)",
                                    placeholder="wide reed beds and a wooden footbridge",
                                    max_length=futures.MAX_PROMPT, lines=2,
                                )  # fmt: skip
                                with gr.Row(equal_height=True):
                                    fu_object = gr.Textbox(
                                        label="Object to change (optional)", max_lines=1,
                                        max_length=masking.MAX_QUERY, placeholder="chairs",
                                        info="Only that object changes; the rest of the view "
                                        "stays exactly as it is. FLUX.1 Kontext editor only.",
                                        scale=3,
                                    )  # fmt: skip
                                    fu_detect = gr.Button(
                                        "Detect objects", scale=1, min_width=120
                                    )  # fmt: skip
                                with gr.Accordion(
                                    "Area to change (optional): paint where the change goes",
                                    open=False,
                                ) as fu_area_box:
                                    fu_area = gr.ImageEditor(
                                        label="Paint over the area; only it changes",
                                        type="numpy", image_mode="RGBA", sources=(),
                                        transforms=(), layers=False, height=360,
                                        brush=gr.Brush(default_size=80, colors=[AREA_COLOUR],
                                                       color_mode="fixed"),
                                        eraser=gr.Eraser(default_size=80),
                                        show_download_button=False,
                                    )  # fmt: skip
                                    with gr.Row(equal_height=True):
                                        fu_area_note = gr.Markdown(AREA_HINT)
                                        fu_area_load = gr.Button(
                                            "Load start image (clears the area)", scale=0,
                                            min_width=160,
                                        )  # fmt: skip
                                with gr.Group(visible=False) as fu_parent_group:
                                    fu_parent_note = gr.Markdown()
                                    fu_parent_clear = gr.Button("Start fresh instead", size="sm")
                                with gr.Row():
                                    fu_seed = gr.Number(label="Seed", value=0, precision=0)
                                    fu_speed = gr.Radio(
                                        [("Fast", "fast"), ("Quality", "quality")],
                                        value=futures.DEFAULT_SPEED, label="Speed",
                                        info="Fast: fewer steps, about a third quicker.",
                                    )  # fmt: skip
                                    fu_make = gr.Button("Make future", variant="primary")
                                fu_sent = gr.Markdown()
                                fu_status = gr.Markdown()
                                fu_result = gr.Image(
                                    label="Future (download with the arrow)",
                                    interactive=False, format="jpeg", height=360,
                                )  # fmt: skip
                                with gr.Row(equal_height=True):
                                    fu_changed = gr.Image(
                                        label="Changed area (the dimmed part is unchanged)",
                                        interactive=False, format="jpeg", height=240,
                                        visible=False, min_width=120,
                                    )  # fmt: skip
                                    fu_detected = gr.Image(
                                        label="Detected objects", interactive=False,
                                        format="jpeg", height=240, visible=False,
                                        min_width=120,
                                    )  # fmt: skip
                                fu_details = gr.Markdown()
                                fu_from = gr.Button("Start from this", interactive=False)
                                fu_gallery = gr.Gallery(
                                    label="Futures (pick a saved view)", columns=3,
                                    height=320, object_fit="cover", allow_preview=False,
                                    elem_id="futures-gallery",
                                )  # fmt: skip
                                with gr.Row(equal_height=True):
                                    fu_pick_note = gr.Markdown()
                                    fu_compare = gr.Button("Compare", scale=0, min_width=120)
                                with gr.Row(visible=False, equal_height=False) as fu_cmp:
                                    with gr.Column(min_width=120):
                                        fu_cmp_a = gr.Image(
                                            interactive=False, height=300, show_label=False
                                        )
                                        fu_cmp_a_md = gr.Markdown()
                                        fu_cmp_a_det = gr.Image(
                                            interactive=False, height=160, visible=False
                                        )
                                    with gr.Column(min_width=120):
                                        fu_cmp_b = gr.Image(
                                            interactive=False, height=300, show_label=False
                                        )
                                        fu_cmp_b_md = gr.Markdown()
                                        fu_cmp_b_det = gr.Image(
                                            interactive=False, height=160, visible=False
                                        )
                                with gr.Accordion(
                                    "Settings: starters, editor and editor instructions",
                                    open=False,
                                ):
                                    fu_starters = gr.Textbox(
                                        label="Starter directions (one per line, "
                                        '"label: prompt text")', lines=6,
                                    )  # fmt: skip
                                    fu_editor = gr.Dropdown(
                                        [(e.name, k) for k, e in models.EDITORS.items()],
                                        value=models.DEFAULT_EDITOR, label="Editor",
                                        info="SDXL restyles the whole view. Kontext follows "
                                        'instructions ("remove the chairs", "make the chairs '
                                        'red") and keeps the rest; slower, larger GPU.',
                                    )  # fmt: skip
                                    gr.Markdown(
                                        "**Editor instructions.** Fixed text the Space adds "
                                        "to your prompt before the editor gets it. Change or "
                                        "empty any of them; the line above **Make future** "
                                        "shows exactly what will be sent."
                                    )  # fmt: skip
                                    fu_instr = [
                                        gr.Textbox(
                                            value=instructions.DEFAULTS[k],
                                            label=instructions.LABELS[k][0],
                                            info=instructions.LABELS[k][1], lines=2,
                                            max_length=instructions.MAX_TEXT,
                                        )
                                        for k in instructions.FUTURE_KEYS
                                    ]  # fmt: skip
                                    with gr.Row():
                                        fu_settings_save = gr.Button("Save settings")
                                        fu_instr_reset = gr.Button(
                                            "Reset instructions to defaults (then save)"
                                        )
                                fu_vp, fu_vp_ids, fu_job = (
                                    gr.State(None),
                                    gr.State([]),
                                    gr.State(None),
                                )
                                fu_pending, fu_parent = gr.State(None), gr.State(None)
                                # The painted area as a 1-bit PNG, and the start image
                                # it was drawn on ("view:<id>" or "future:<job id>").
                                fu_area_png, fu_area_on = gr.State(None), gr.State(None)
                                fu_keys, fu_picked = gr.State([]), gr.State([])
                                fu_shown = gr.State(None)
                                fu_timer = gr.Timer(1, active=False)
                                fu_move = gr.Textbox(
                                    elem_id="futures-camera-bridge", show_label=False,
                                    container=False,
                                )  # fmt: skip
                                fu_overlay = gr.Textbox(
                                    elem_id="futures-overlay-bridge", show_label=False,
                                    container=False,
                                )  # fmt: skip
                                fu_deleted = gr.State(None)
                                fu_delete = gr.Textbox(
                                    elem_id="futures-delete-bridge", show_label=False,
                                    container=False,
                                )  # fmt: skip
                # Hidden helpers for the iframe messages (kept in the DOM, hidden by CSS).
                bridge = gr.Textbox(elem_id="placement-bridge", show_label=False, container=False)
                camera_json = gr.Textbox(elem_id="camera-bridge", show_label=False, container=False)

        sliders = [px, py, pz, pturn, psize, ptilt_x, ptilt_y]  # order of Placement's fields
        end = [place_group, placing, obj_job, moving]
        scene_outputs = [version, hidden, viewer, download, shown, history_table, *end]
        log = (log_outputs, [picker, version], [history, log_table])
        # Earlier previews; runs before `log`, so the prompt log reads their updated status.
        pv_list = (on_previews, picker, [pv_table, pv_ids])
        lib = (
            on_tables, [picker, version, hidden, moving],
            [objects_table, obj_keys, edits_table, edit_numbers],
        )  # fmt: skip
        details = (on_details, picker, [sc_facts, sc_name, sc_notes, sc_confirm, sc_status])

        # New scene
        sc_quick = {"queue": False, "show_progress": "hidden"}
        files.change(on_upload, files, [message, create, n_photos]).then(
            scene_estimate, [n_photos, *sc_settings], sc_estimate, **sc_quick
        )
        sc_preset.input(
            on_scene_preset, [sc_preset, n_photos], [*sc_settings, sc_estimate], **sc_quick
        )
        for slider in sc_settings:
            slider.input(
                on_scene_slider, [n_photos, *sc_settings], [sc_preset, sc_estimate], **sc_quick
            )
        clicked(create, "⏳ Starting…", on_create, [files, *sc_settings], [message, table])
        clicked(refresh, "⏳ Refreshing…", refresh_all, picker, [table, picker])

        # Start tab, finished-job notice, opening scenes and versions
        demo.load(on_tool_options, None, tools_json, queue=False, show_progress="hidden")
        tools_json.input(
            on_save_tool_options, tools_json, None, queue=False, show_progress="hidden"
        )
        demo.load(on_load, [obj_job, placing], [tabs, table, picker, seen, *scene_outputs]).then(
            *pv_list
        ).then(*log).then(*lib).then(*details)
        gr.Timer(10).tick(on_tick, [picker, seen], [table, picker, seen])
        open_scene = (on_pick_scene, [picker, obj_job, placing], scene_outputs)
        picker.input(*open_scene).then(*pv_list).then(*log).then(*lib).then(*details)

        # Scene details: save name and notes; delete after a second click
        renamed = gr.State(None)  # scene id, kept while the picker is briefly emptied
        clicked(
            sc_save, "⏳ Saving…", on_save_details, [picker, sc_name, sc_notes],
            [sc_status, picker, sc_name, sc_notes, renamed],
        ).success(lambda s: gr.update(value=s), renamed, picker, queue=False)  # fmt: skip
        clicked(sc_delete, "⏳ Checking…", on_delete_ask, picker, [sc_confirm, sc_confirm_text])
        sc_delete_no.click(lambda: gr.update(visible=False), None, sc_confirm, queue=False)
        clicked(
            sc_delete_yes, "⏳ Deleting…", on_delete, picker,
            [sc_status, sc_confirm, table, picker, seen],
        ).success(*open_scene).then(*pv_list).then(*log).then(*lib).then(
            # keeps the "Deleted" line in the status
            lambda s: on_details(s)[:4], picker, [sc_facts, sc_name, sc_notes, sc_confirm]
        )  # fmt: skip
        history_table.select(
            on_pick_row, [picker, history_table, obj_job, placing], scene_outputs
        ).then(*log).then(*lib)

        # Objects table: hide/show, add a generated one, move a placed one
        objects_table.select(
            on_object_select,
            [camera_json, picker, show_obj, version, hidden, obj_job, placing, moving, floor,
             obj_keys],
            [obj_status, hidden, viewer, obj_job, placing, moving, place_group, obj_ref, floor,
             *sliders],
            js=OBJECT_SELECT_JS,
            trigger_mode="once",
        ).then(*lib)  # fmt: skip
        # Edits table: open one edit's version
        edits_table.select(
            on_edit_select,
            [camera_json, picker, show_obj, version, hidden, edit_numbers, obj_job, placing,
             moving, floor],
            scene_outputs,
            js=OBJECT_SELECT_JS,
            trigger_mode="once",
        ).then(*log).then(*lib)  # fmt: skip

        # Add object
        clicked(
            obj_go,
            "⏳ Starting…",
            on_generate,
            [picker, version, obj_prompt, obj_seed, obj_image, placing],
            [obj_status, obj_job, placing, place_group],
        ).then(*lib)
        gr.Timer(5).tick(
            on_object_tick,
            [obj_job, placing, picker, version, hidden, show_obj],
            [obj_status, obj_job, placing, viewer, place_group, obj_ref, floor, *sliders],
        )
        for slider in sliders:
            slider.input(None, sliders, None, js=PLACEMENT_JS)
        bridge.change(on_bridge, bridge, sliders)
        show_obj.input(None, show_obj, None, js=SHOW_OBJECT_JS)
        # A finished object enters placement: list it among the generated objects.
        placing.change(*lib)
        # A generating object's progress also shows in the table.
        obj_status.change(*lib)
        clicked(
            place_ok,
            "⏳ Placing…",
            on_place,
            [*sliders, picker, version, obj_job, moving],
            [obj_status, *scene_outputs],
            js=PLACE_JS,
        ).then(*log).then(*lib)
        clicked(
            place_cancel,
            "⏳ Cancelling…",
            on_cancel,
            [camera_json, picker, version, hidden],
            [obj_status, viewer, place_group, placing, obj_job, moving],
            js=GET_CAMERA_JS,
        ).then(*lib)

        # Edit scene
        ed_region.input(None, ed_region, ed_region, js=REGION_JS)
        # A new viewer page starts without the box overlay, so switch the region off.
        viewer.change(lambda: False, None, ed_region)
        quick = {"queue": False, "show_progress": "hidden"}
        ed_preset.input(on_budget_preset, ed_preset, [*ed_budget, ed_estimate], **quick)
        for slider in ed_budget:
            slider.input(on_budget_slider, ed_budget, [ed_preset, ed_estimate], **quick)
        clicked(
            ed_go,
            "⏳ Starting…",
            on_edit,
            # States (version, placing) go last: GET_CAMERA_JS never sees them, so any input
            # after them would be shifted.
            [camera_json, picker, ed_prompt, ed_region, ed_seed, *ed_budget, version, placing],
            [ed_status, edit_job],
            js=GET_CAMERA_JS,
        )
        gr.Timer(10).tick(
            on_edit_tick,
            [edit_job, picker, obj_job, placing],
            [ed_status, edit_job, *scene_outputs],
        )
        ed_status.change(*log).then(*lib)

        # Preview
        clicked(
            pv_go,
            "⏳ Starting…",
            on_preview,
            [camera_json, picker, pv_prompt, pv_strength, pv_seed, version, placing, pv_negative],
            [pv_status, preview_job],
            js=GET_CAMERA_JS,
        )
        gr.Timer(5).tick(
            on_preview_tick,
            preview_job,
            [pv_status, preview_job, preview_col, preview_img, render_img],
        )
        # A started or finished preview adds a line to the prompt log.
        pv_status.change(*log)
        # Earlier previews (reopen-previews): the list follows starts and finishes, and a
        # click shows a finished preview or follows a running one.
        preview_job.change(*pv_list)
        pv_tab.select(*pv_list)
        pv_tab.select(on_preview_negative, picker, pv_negative)
        picker.input(on_preview_negative, picker, pv_negative)
        pv_table.select(
            on_pick_preview,
            pv_ids,
            [pv_status, preview_job, preview_col, preview_img, render_img],
        )

        # Futures (add-viewpoint-futures)
        fu_open = (
            on_futures_open, [picker, fu_vp],
            [fu_intro, fu_strip, fu_vp_ids, fu_vp, fu_name, fu_starter, fu_starters, fu_editor,
             fu_gallery, fu_keys, fu_picked, *fu_instr],
        )  # fmt: skip
        area_load = (
            on_area_load, [picker, fu_vp, fu_parent],
            [fu_area, fu_area_on, fu_area_png, fu_area_note],
        )  # fmt: skip
        sent = (
            on_sent_preview, [picker, fu_starter, fu_prompt, fu_object, fu_area_png], fu_sent,
        )  # fmt: skip
        fu_tab.select(*fu_open).then(*sent)
        picker.input(*fu_open).then(*sent)
        fu_strip.select(
            on_pick_viewpoint,
            [picker, version, fu_vp_ids, obj_job, placing],
            [fu_vp, fu_name, fu_intro, fu_gallery, fu_keys, fu_picked, fu_move, *scene_outputs],
        ).then(None, fu_move, None, js=SET_CAMERA_JS).then(*area_load)
        fu_strip_out = [fu_status, fu_strip, fu_vp_ids, fu_vp, fu_name]
        clicked(
            fu_save, "⏳ Saving…", on_save_view, [camera_json, picker, fu_name, version],
            [*fu_strip_out, fu_intro, fu_gallery, fu_keys, fu_pending, fu_timer],
            js=GET_CAMERA_JS,
        )  # fmt: skip
        for button, action in ((fu_rename, "rename"), (fu_up, "up"), (fu_down, "down"),
                               (fu_remove, "remove")):  # fmt: skip
            button.click(
                lambda s, v, n, a=action: on_manage_viewpoint(a, s, v, n),
                [picker, fu_vp, fu_name], [*fu_strip_out, fu_gallery, fu_keys],
            )  # fmt: skip
        clicked(
            fu_make, "⏳ Starting…", on_make_future,
            # States go last: GET_CAMERA_JS only replaces the first input.
            [camera_json, picker, fu_starter, fu_prompt, fu_object, fu_seed, fu_speed, fu_vp,
             fu_parent, version, fu_area_png, fu_area_on],
            [fu_status, fu_job, fu_vp, fu_parent, fu_parent_note, fu_strip, fu_vp_ids, fu_name,
             fu_gallery, fu_keys, fu_timer],
            js=GET_CAMERA_JS,
        ).then(lambda p: gr.update(visible=bool(p)), fu_parent, fu_parent_group)  # fmt: skip
        fu_timer.tick(
            on_futures_tick, [fu_job, fu_pending, picker, fu_vp],
            [fu_status, fu_job, fu_pending, fu_result, fu_details, fu_from, fu_gallery,
             fu_keys, fu_strip, fu_timer, fu_shown, fu_changed, fu_detected],
            concurrency_limit=1, show_progress="hidden",
        )  # fmt: skip
        # A started or finished future adds a line to the prompt log.
        fu_job.change(*log)
        fu_gallery.select(
            on_pick_tile, [picker, fu_vp, fu_keys, fu_picked],
            [fu_result, fu_details, fu_from, fu_shown, fu_picked, fu_pick_note, fu_overlay,
             fu_changed, fu_detected],
        ).then(None, fu_overlay, None, js=SHOW_FUTURE_JS)  # fmt: skip
        # `input` only (a programmatic value change must not delete); `fu_deleted` keeps
        # the last request, so an event that arrives twice deletes one future, not two.
        fu_delete.input(
            on_delete_future,
            [fu_delete, fu_deleted, picker, fu_vp, fu_keys, fu_picked, fu_shown, fu_parent],
            [fu_gallery, fu_keys, fu_picked, fu_pick_note, fu_result, fu_details, fu_from,
             fu_shown, fu_overlay, fu_parent, fu_parent_note, fu_parent_group, fu_deleted],
        ).then(None, fu_overlay, None, js=HIDE_FUTURE_JS).then(
            # The shown future was deleted (no tile shown): hide its two small images too.
            lambda shown: [gr.skip()] * 2 if shown else [gr.update(value=None, visible=False)] * 2,
            fu_shown, [fu_changed, fu_detected],
        )  # fmt: skip
        fu_from.click(on_start_from, fu_shown, [fu_parent, fu_parent_note, fu_parent_group]).then(
            *area_load
        )
        fu_detect.click(
            on_detect_objects, [picker, fu_vp, fu_object], [fu_status, fu_detected],
            concurrency_limit=1,
        )  # fmt: skip
        fu_parent_clear.click(
            lambda: (None, "", gr.update(visible=False)), None,
            [fu_parent, fu_parent_note, fu_parent_group],
        ).then(*area_load)  # fmt: skip
        # "Area to change" (add-area-layout): the editor shows the start image of the next
        # future; a new start image clears the area drawn on the old one.
        fu_area_load.click(*area_load)
        fu_area_box.expand(*area_load)
        fu_area.change(
            on_area_change, [fu_area, fu_area_on], [fu_area_png, fu_area_note],
            show_progress="hidden",
        )  # fmt: skip
        fu_compare.click(
            on_compare, [picker, fu_vp, fu_picked],
            [fu_cmp, fu_cmp_a, fu_cmp_a_md, fu_cmp_a_det, fu_cmp_b, fu_cmp_b_md, fu_cmp_b_det],
        )  # fmt: skip
        clicked(
            fu_settings_save, "⏳ Saving…", on_save_future_settings,
            [picker, fu_starters, fu_editor, *fu_instr], [fu_status, fu_starter],
        ).then(*sent)  # fmt: skip
        fu_instr_reset.click(lambda: _instruction_texts(None), None, fu_instr)
        for comp in (fu_starter, fu_prompt, fu_object, fu_area_png):
            comp.change(*sent, show_progress="hidden")

        # Downloads (always the full files, whatever is hidden)
        clicked(
            get_ply, "⏳ Fetching…", lambda j, n: on_download(j, n, "ply"), [picker, version],
            download,
        )  # fmt: skip
        clicked(
            get_splat, "⏳ Fetching…", lambda j, n: on_download(j, n, "splat"),
            [picker, version], download,
        )  # fmt: skip
    return demo


if __name__ == "__main__":
    # Scene files are served from the store's download folder to the viewer.
    STORE.file_dir.mkdir(parents=True, exist_ok=True)
    build().launch(allowed_paths=[str(STORE.file_dir.resolve())])
