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
from pathlib import Path

import gradio as gr

# On the Space, splat_studio is uploaded next to this file; locally use src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splat_studio.editing import images  # noqa: E402
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
from splat_studio.space import prompt_log  # noqa: E402
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
    IMAGE_TIMEOUT_S,
    OBJECT_TIMEOUT_S,
    PREVIEW_TIMEOUT_S,
    HubStore,
    Job,
    JobManager,
    LocalStore,
    format_elapsed,
    image_result_path,
)
from splat_studio.space.validation import (  # noqa: E402
    HELP,
    UploadError,
    validate_object_image,
    validate_source_image,
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
    view_splat,
)
from splat_studio.space.viewer import camera_from_json, scene_html  # noqa: E402


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
IMAGES = JobManager(STORE, prefix="images", timeout_s=IMAGE_TIMEOUT_S)
STATUS_LABEL = {
    "queued": "⏳ queued",
    "running": "⚙️ running",
    "done": "✅ done",
    "failed": "❌ failed",
}
COLUMNS = ["Job", "Input", "Status", "Elapsed", "Details"]


def on_upload(files: list[str] | None):
    if not files:
        return gr.update(value=HELP), gr.update(interactive=False)
    try:
        upload = validate_upload(files)
    except UploadError as e:
        return gr.update(value=f"**Upload rejected.** {e}"), gr.update(interactive=False)
    return gr.update(value=f"**Ready.** {upload.summary}"), gr.update(interactive=True)


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
        rows.append(
            [j.job_id, j.summary, STATUS_LABEL.get(j.status, j.status),
             format_elapsed(j.elapsed(now)), details]
        )  # fmt: skip
    return rows


def scene_choices(jobs: list[Job]) -> list[tuple[str, str]]:
    """Finished scenes, oldest first (`jobs` is newest first).

    New scenes go at the end: Gradio's dropdown marks its selection by position, so
    putting a new scene first made it look selected and picking it did nothing.
    """
    done = [j for j in jobs if j.status == "done"]
    return [(f"{j.job_id} · {j.summary}", j.job_id) for j in reversed(done)]


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
    """Open a version (newest if `number` is None) of a scene; outputs `SCENE_OUTPUTS`.

    (version, hidden, viewer, download, shown, history table, place_group, placing,
    obj_job, moving)
    """
    job, versions = _scene(job_id)
    version = next((v for v in versions if v.number == number), None) if number else None
    version = version or (versions[0] if versions else None)
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
    rows = [e.row() for e in prompt_log.scene_log(job_id, versions, previews, edits)]
    return prompt_log.history_markdown(versions, number), gr.update(value=rows)


def on_pick_scene(job_id: str | None, obj_job: str | None, placing: bool):
    """Show the newest version of the picked scene and list all its versions."""
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
    """Open on the Scene tab with the newest finished scene, or on New scene if none.

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
                    f"Scene {job.job_id} ({job.summary}) is ready. "
                    "Pick it in the 'Finished scene' list of the Scene tab to view it.",
                    duration=15,
                )
    return jobs_table(jobs), picker, done


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
</script>"""


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


def on_preview(camera_json, scene_id, prompt, strength, seed, number, placing):
    version = _version(scene_id, number)
    if version is None:
        raise gr.Error("Pick a finished scene first.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Describe the look first, e.g. 'same street in autumn with rain gardens'.")
    try:
        camera = camera_from_viewer(camera_json)
    except CameraError as e:
        raise gr.Error(str(e)) from e
    job = PREVIEWS.create_preview(
        prompt, version.path, camera.to_dict(), seed=int(seed or 0), strength=float(strength)
    )
    if job.status == "failed":
        gr.Warning(f"The preview could not start. {job.error['message']}")
        return f"**Could not start.** {job.error['message']} {job.error.get('tip', '')}", None
    note = " The object being placed is not part of it." if placing else ""
    gr.Info("Preview started. It takes a few minutes; the image appears below the button.")
    return (
        f"⚙️ Rendering this view of v{version.number} and restyling it (job {job.job_id}).{note}",
        job.job_id,
    )


def on_preview_tick(preview_job: str | None):
    """Poll the preview job. Outputs: status, preview_job, column, preview, render."""
    rest = [gr.skip()] * 3
    if not preview_job:
        return [gr.skip(), gr.skip(), *rest]
    job = PREVIEWS.get(preview_job)
    job = PREVIEWS.refresh(job) if job else None
    if job is None:
        return ["The preview job was not found.", None, *rest]
    if job.status == "failed":
        err = job.error or {}
        msg = f"❌ **Preview failed.** {err.get('message', '')} {err.get('tip', '')}".strip()
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
    gr.Info("Preview ready: see the Preview tab.")
    return [
        f"✅ Preview of **{job.summary}** ready{took}. The scene is unchanged.",
        None,
        gr.update(visible=True),
        str(styled),
        str(render) if render else None,
    ]


# --- Image editing tab ----------------------------------------------------------------


def on_image_mode(mode: str):
    """Show the controls of the chosen mode. Outputs: source, aspect, strength, save-base."""
    with_image = mode == images.IMAGE_TO_IMAGE
    return (
        gr.update(visible=with_image),
        gr.update(visible=not with_image),
        gr.update(visible=with_image),
        gr.update(visible=with_image),
    )


def on_image(mode, prompt, source, aspect, strength, seed):
    prompt = (prompt or "").strip()
    with_image = mode == images.IMAGE_TO_IMAGE
    if not prompt:
        example = "the same roof covered in wildflowers" if with_image else "a green roof"
        raise gr.Error(f"Describe the image first, e.g. '{example}'.")
    kwargs = {"seed": int(seed or 0)}
    if with_image:
        try:
            kwargs.update(source=validate_source_image(source), strength=float(strength))
        except UploadError as e:
            raise gr.Error(str(e)) from e
    else:
        kwargs["aspect"] = images.aspect_of(aspect)
    job = IMAGES.create_image(prompt, **kwargs)
    if job.status == "failed":
        return f"**Could not start.** {job.error['message']} {job.error.get('tip', '')}", None
    what = "Changing your image" if with_image else "Making the image"
    return f"⚙️ {what} (job {job.job_id}).", job.job_id


def on_image_tick(image_job: str | None):
    """Poll the image job. Outputs: status, image_job, result, use-as-input button."""
    rest = [gr.skip()] * 2
    if not image_job:
        return [gr.skip(), gr.skip(), *rest]
    job = IMAGES.get(image_job)
    job = IMAGES.refresh(job) if job else None
    if job is None:
        return ["The image job was not found.", None, *rest]
    if job.status == "failed":
        err = job.error or {}
        msg = f"❌ **Image failed.** {err.get('message', '')} {err.get('tip', '')}".strip()
        return [msg, None, *rest]
    if job.status != "done":
        label = "⏳ Waiting for a GPU" if job.status == "queued" else "⚙️ Generating"
        elapsed = format_elapsed(job.elapsed(IMAGES.clock()))
        return [f"{label}: **{job.summary}** ({elapsed})", gr.skip(), *rest]
    try:
        result = STORE.download(image_result_path(job))
    except Exception as e:
        return [f"**The image could not be fetched** ({type(e).__name__}).", None, *rest]
    if result is None:
        return ["**The image is missing.**", None, *rest]
    seconds = job.run.get("seconds")
    took = f" in {format_elapsed(seconds)}" if seconds else ""
    return [
        f"✅ **{job.summary}** ready{took}.",
        None,
        str(result),
        gr.update(interactive=True),
    ]


def on_use_result(result: str | None):
    """Continue from the last result. Outputs: mode, source, aspect, strength."""
    if not result:
        raise gr.Error("Generate an image first.")
    return images.IMAGE_TO_IMAGE, result, *on_image_mode(images.IMAGE_TO_IMAGE)[1:]


def on_save_base(source: str | None):
    """Store the input image unchanged. Outputs: status, image_job, result, use-as-input."""
    try:
        image = validate_source_image(source)
    except UploadError as e:
        raise gr.Error(str(e)) from e
    job = IMAGES.save_image(image)
    return (
        f"✅ Saved unchanged as a base (job {job.job_id}). Find it under "
        '"Earlier images" below, or write a prompt and generate from it.',
        None,
        str(image),
        gr.update(interactive=True),
    )


IMAGE_MODE_LABEL = {"image": "edited", "text": "from text", "upload": "base, unchanged"}


def on_image_history():
    """Finished images, newest first. Outputs: gallery items (image, caption), paths, note."""
    items, missing, errors = [], 0, []
    try:
        finished = [j for j in IMAGES.list(refresh=False) if j.status == "done" and j.scene_path]
    except Exception as e:
        return [], [], f"⚠️ The image list could not be read ({type(e).__name__}: {str(e)[:150]})."
    for job in finished[:24]:
        try:
            path = STORE.download(image_result_path(job))
        except Exception as e:
            errors.append(f"{type(e).__name__}: {str(e)[:150]}")
            continue
        if not path:
            missing += 1
        else:
            mode = job.params.get("mode", "text")
            label = IMAGE_MODE_LABEL.get(mode, mode)
            seed = "" if mode == "upload" else f", seed {job.params.get('seed', 0)}"
            items.append((str(path), f"{job.summary} ({label}{seed})"))
    note = f"{len(items)} of {len(finished)} finished images shown (newest first)."
    if not finished:
        note = "No finished images yet."
    if missing:
        note += f" {missing} file(s) missing in the dataset."
    if errors:
        note += f" {len(errors)} could not be fetched, e.g. {errors[0]}"
    return items, [p for p, _ in items], note


def on_pick_earlier(paths: list[str] | None, evt: gr.SelectData):
    """Use a clicked earlier image as the input. Outputs: mode, source, aspect, strength."""
    if not paths or evt.index is None or evt.index >= len(paths):
        raise gr.Error("Show the earlier images again, then pick one.")
    gr.Info("Picked as input image. Write a prompt and generate.")
    return on_use_result(paths[evt.index])


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


def on_create(files: list[str] | None):
    try:
        upload = validate_upload(files)
    except UploadError as e:
        return f"**Upload rejected.** {e}", jobs_table()
    job = JOBS.create(upload)
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
#placement-bridge, #camera-bridge { display: none !important; }
#history-table td, #history-table th { font-size: 13px; }
#history-table tbody tr, #objects-table tbody tr, #edits-table tbody tr { cursor: pointer; }
#objects-table td:last-child, #edits-table td:last-child { font-weight: 600; white-space: nowrap; }
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Splat Studio", css=CSS, head=BRIDGE_HEAD) as demo:
        gr.Markdown("# Splat Studio\nTurn a video or photo set into a 3D Gaussian splat scene.")
        if not DATASET:
            gr.Markdown("⚠️ `SPLAT_STUDIO_DATASET` is not set: jobs are stored locally only.")
        version, hidden, seen = gr.State(None), gr.State([]), gr.State(None)
        obj_job, placing, floor = gr.State(None), gr.State(False), gr.State(None)
        moving = gr.State(None)  # id of the placed object being moved, if any
        obj_keys, edit_numbers = gr.State([]), gr.State([])
        preview_job, edit_job, image_job = gr.State(None), gr.State(None), gr.State(None)
        image_paths = gr.State([])

        with gr.Tabs(selected="new") as tabs:
            # --- New scene: upload, create, jobs -----------------------------------
            with gr.Tab("New scene", id="new"):
                files = gr.File(
                    label="Video or photos",
                    file_count="multiple",
                    file_types=[".mp4", ".mov", ".jpg", ".jpeg", ".png"],
                )
                message = gr.Markdown(HELP)
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
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        viewer = gr.HTML(scene_html(STORE, None))
                        shown = gr.Markdown()
                        with gr.Row():
                            get_ply = gr.Button("Download .ply")
                            get_splat = gr.Button("Download .splat (compressed)")
                        download = gr.File(label="Download", visible=False, interactive=False)
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("History"):
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
                            with gr.Tab("Objects"):
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
                            with gr.Tab("Edit scene"):
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
                            with gr.Tab("Preview"):
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
                # Hidden helpers for the iframe messages (kept in the DOM, hidden by CSS).
                bridge = gr.Textbox(elem_id="placement-bridge", show_label=False, container=False)
                camera_json = gr.Textbox(elem_id="camera-bridge", show_label=False, container=False)

            # --- Image editing: 2D images, no scene needed ----------------------------
            with gr.Tab("Image editing", id="image") as image_tab:
                gr.Markdown(
                    "Make an image from a text, or change an image you upload. "
                    "This works without a scene and changes no scene."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        im_mode = gr.Radio(
                            list(images.MODES), value=images.TEXT_TO_IMAGE, label="Mode"
                        )
                        im_source = gr.Image(
                            label="Input image (PNG, JPG or WebP, up to 20 MB)",
                            type="filepath", sources=["upload", "clipboard"], height=260,
                            visible=False,
                        )  # fmt: skip
                        im_prompt = gr.Textbox(
                            label="Prompt",
                            lines=3,
                            placeholder="Text to image: describe the whole picture. "
                            "With an image: describe how it should look afterwards.",
                        )
                        im_aspect = gr.Dropdown(
                            list(images.ASPECTS), value=images.DEFAULT_ASPECT, label="Format"
                        )
                        im_strength = gr.Slider(
                            0.2, 0.95, value=0.6, step=0.05, visible=False,
                            label="Change (low keeps more of the input)",
                        )  # fmt: skip
                        with gr.Row():
                            im_seed = gr.Number(label="Seed", value=0, precision=0)
                            im_go = gr.Button("Generate image", variant="primary")
                        im_keep = gr.Button("Save as base (no change)", visible=False)
                        im_status = gr.Markdown()
                    with gr.Column(scale=3):
                        im_result = gr.Image(
                            label="Result (download with the arrow)",
                            interactive=False, format="png", height=520,
                        )  # fmt: skip
                        im_again = gr.Button("Use result as input", interactive=False)
                gr.Markdown("### Earlier images")
                with gr.Row(equal_height=True):
                    im_history_note = gr.Markdown()
                    im_history_go = gr.Button("Refresh list", scale=0, min_width=140)
                im_history = gr.Gallery(
                    label="Generated and saved images (click one to use it as the input)",
                    columns=6, height=320, object_fit="cover", allow_preview=False,
                )  # fmt: skip
        sliders = [px, py, pz, pturn, psize, ptilt_x, ptilt_y]  # order of Placement's fields
        end = [place_group, placing, obj_job, moving]
        scene_outputs = [version, hidden, viewer, download, shown, history_table, *end]
        log = (log_outputs, [picker, version], [history, log_table])
        lib = (
            on_tables, [picker, version, hidden, moving],
            [objects_table, obj_keys, edits_table, edit_numbers],
        )  # fmt: skip

        # New scene
        files.change(on_upload, files, [message, create])
        clicked(create, "⏳ Starting…", on_create, files, [message, table])
        clicked(refresh, "⏳ Refreshing…", refresh_all, picker, [table, picker])

        # Start tab, finished-job notice, opening scenes and versions
        demo.load(on_load, [obj_job, placing], [tabs, table, picker, seen, *scene_outputs]).then(
            *log
        ).then(*lib)
        gr.Timer(10).tick(on_tick, [picker, seen], [table, picker, seen])
        picker.input(on_pick_scene, [picker, obj_job, placing], scene_outputs).then(*log).then(*lib)
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
            [camera_json, picker, pv_prompt, pv_strength, pv_seed, version, placing],
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

        # Image editing
        mode_outputs = [im_source, im_aspect, im_strength, im_keep]
        im_mode.change(on_image_mode, im_mode, mode_outputs)
        clicked(
            im_go,
            "⏳ Starting…",
            on_image,
            [im_mode, im_prompt, im_source, im_aspect, im_strength, im_seed],
            [im_status, image_job],
        )
        gr.Timer(5).tick(on_image_tick, image_job, [im_status, image_job, im_result, im_again])
        im_again.click(on_use_result, im_result, [im_mode, im_source, *mode_outputs[1:]])
        clicked(
            im_keep, "⏳ Saving…", on_save_base, im_source,
            [im_status, image_job, im_result, im_again],
        )  # fmt: skip
        history_out = [im_history, image_paths, im_history_note]
        clicked(im_history_go, "⏳ Loading…", on_image_history, None, history_out)
        # Load the list when the tab opens, and again whenever a new result arrives.
        image_tab.select(on_image_history, None, history_out)
        im_result.change(on_image_history, None, history_out)
        im_history.select(on_pick_earlier, image_paths, [im_mode, im_source, *mode_outputs[1:]])

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
