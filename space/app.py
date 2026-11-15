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

from splat_studio.editing.objects import Placement  # noqa: E402
from splat_studio.editing.preview import CameraError, camera_from_viewer  # noqa: E402
from splat_studio.editing.scene_edit import EditError  # noqa: E402
from splat_studio.space import edit_scene, prompt_log  # noqa: E402
from splat_studio.space.add_object import PlaceError, place_object, start_placement  # noqa: E402
from splat_studio.space.jobs import (  # noqa: E402
    EDIT_TIMEOUT_S,
    OBJECT_TIMEOUT_S,
    PREVIEW_TIMEOUT_S,
    HubStore,
    Job,
    JobManager,
    LocalStore,
    format_elapsed,
)
from splat_studio.space.validation import HELP, UploadError, validate_upload  # noqa: E402
from splat_studio.space.versions import (  # noqa: E402
    HISTORY_COLUMNS,
    MODE_LABEL,
    Version,
    download_file,
    filtered_splat,
    history_rows,
    list_versions,
    version_from_row,
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
    return [(f"{j.job_id} · {j.summary}", j.job_id) for j in jobs if j.status == "done"]


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
    """Outputs that leave placement mode: (place_group, placing, obj_job).

    A finished object is dropped (nothing was saved); a job still generating is kept,
    and its object is shown on whichever version is picked when it is done.
    """
    return gr.update(visible=False), False, (None if placing else obj_job)


def _shown(version: Version | None) -> str:
    if version is None:
        return ""
    what = MODE_LABEL.get(version.mode, version.mode)
    prompt = f': "{version.prompt}"' if version.prompt else ""
    return f"Showing **v{version.number}** · {what}{prompt}"


def _objects_control(version: Version | None):
    """Show/hide checkboxes for the objects of a version; hidden if it lists none."""
    if version is None or not version.objects:
        return gr.update(choices=[], value=[], visible=False)
    choices = [(f"{o.get('prompt') or 'object'} ({o['id']})", o["id"]) for o in version.objects]
    return gr.update(choices=choices, value=[o["id"] for o in version.objects], visible=True)


def _hidden_splat(version: Version | None, hidden: list[str] | None):
    """Local `.splat` without the hidden objects, or None to use the version's own file."""
    if version is None or not hidden:
        return None
    try:
        return filtered_splat(STORE, version, hidden)
    except Exception:  # fall back to the full scene rather than break the viewer
        return None


def scene_view(job_id: str | None, number: int | None, obj_job: str | None, placing: bool):
    """Open a version (newest if `number` is None) of a scene; outputs `SCENE_OUTPUTS`.

    (version, hidden, viewer, download, shown, objects, history table, place_group,
    placing, obj_job)
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
        _objects_control(version),
        gr.update(value=rows),
        *_end_placement(obj_job, placing),
    )


def log_outputs(job_id: str | None, number: int | None):
    """History of the shown version and the scene's prompt log (task 4.5)."""
    versions = _versions(job_id)
    if not versions:
        return "", gr.update(value=[])
    previews = [j for j in PREVIEWS.list(refresh=False) if j.kind == "preview"]
    rows = [e.row() for e in prompt_log.scene_log(job_id, versions, previews)]
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
    newest = choices[0][1] if choices else None  # the listing is newest first
    return (
        gr.Tabs(selected="scene" if newest else "new"),
        jobs_table(jobs),
        gr.update(choices=choices, value=newest),
        _done_ids(jobs),
        *scene_view(newest, None, obj_job, placing),
    )


def on_tick(current: str | None, seen: list[str] | None):
    """Refresh jobs and picker; announce scenes that finished since the last listing.

    The notice never changes the open scene; "Open newest scene" does that on request.
    """
    jobs = JOBS.list()
    choices = scene_choices(jobs)
    value = current if current in {v for _, v in choices} else None
    done = _done_ids(jobs)
    if seen is not None:
        for job in jobs:
            if job.job_id in set(done) - set(seen):
                gr.Info(
                    f"Scene {job.job_id} ({job.summary}) is ready. "
                    "Click 'Open newest scene' in the Scene tab to view it.",
                    duration=15,
                )
    return jobs_table(jobs), gr.update(choices=choices, value=value), done


def on_open_newest(obj_job: str | None, placing: bool):
    """Outputs: tabs, picker, then `SCENE_OUTPUTS`."""
    choices = scene_choices(JOBS.list())
    if not choices:
        raise gr.Error("No finished scene yet.")
    newest = choices[0][1]
    return (
        gr.Tabs(selected="scene"),
        gr.update(choices=choices, value=newest),
        *scene_view(newest, None, obj_job, placing),
    )


# --- hide objects (redesign-space-gui 4.3) -------------------------------------------


def _placement_cfg(placement: Placement, floor: float | None, visible: bool) -> dict:
    cfg = placement.to_viewer()
    cfg["visible"] = bool(visible)
    if floor is not None:
        cfg["floor"] = floor
    return cfg


def on_objects_shown(
    camera_json, scene_id, number, shown, obj_job, placing, floor, show_obj, x, y, z, turn, size
):
    """Reload the viewer without the unticked objects; keeps the camera and placement.

    Outputs: hidden, viewer.
    """
    version = _version(scene_id, number)
    if version is None:
        return [], gr.skip()
    hidden = [o["id"] for o in version.objects if o["id"] not in set(shown or [])]
    camera = camera_from_json(camera_json)
    obj_path = cfg = None
    job = OBJECTS.get(obj_job) if (placing and obj_job) else None
    if job is not None and job.scene_path:
        obj_path = job.scene_path
        cfg = _placement_cfg(Placement(x, y, z, turn, size), floor, show_obj)
    viewer = scene_html(
        STORE, version.path, object_path=obj_path, placement=cfg,
        splat=_hidden_splat(version, hidden), camera=camera,
    )  # fmt: skip
    return hidden, viewer


# --- add object (task 4.2; click and drag from redesign-space-gui 3) ------------------

# Sends slider values to the viewer iframe, which moves the object without reloading.
PLACEMENT_JS = """(x, y, z, turn, size) => {
  const f = document.querySelector("iframe.splat-viewer");
  if (f && f.contentWindow) {
    f.contentWindow.postMessage({type: "placement", x, y, z, turn, size}, "*");
  }
  return [x, y, z, turn, size];
}"""

# Shows or hides the object being placed; the placement itself is kept.
SHOW_OBJECT_JS = """(visible) => {
  const f = document.querySelector("iframe.splat-viewer");
  if (f && f.contentWindow) {
    f.contentWindow.postMessage({type: "object_visible", visible: !!visible}, "*");
  }
  return [visible];
}"""

# "Place in new version" asks the viewer for its placement first, so the merged object
# lands where the user last saw it even if a drag never reached the sliders.
# Inputs: scene, version, obj_job, x, y, z, turn, size.
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
  if (!p) return args;
  const out = args.slice();
  [out[3], out[4], out[5], out[6], out[7]] = [p.x, p.y, p.z, p.turn, p.size];
  return out;
}"""

# Page-level listener: the viewer posts `placement_changed` after a click or drag; the
# values are written into a hidden textbox whose change event updates the sliders.
# The page also keeps the Edit scene region (window.splatRegion), so a reloaded viewer gets
# it back: the viewer posts `viewer_ready`, the page answers with `region_state`.
BRIDGE_HEAD = """<script>
window.splatRegion = {on: false, frac: 0.2, center: null};
const splatBridge = (id, data) => {
  const box = document.querySelector(`#${id} textarea, #${id} input`);
  if (!box) return;
  box.value = JSON.stringify({...data, t: Date.now()});
  box.dispatchEvent(new Event("input", {bubbles: true}));
};
window.addEventListener("message", (e) => {
  const d = e.data;
  if (!d) return;
  if (d.type === "placement_changed") {
    splatBridge("placement-bridge", {x: d.x, y: d.y, z: d.z, turn: d.turn, size: d.size});
  } else if (d.type === "region_changed") {
    window.splatRegion.center = d.center;
    splatBridge("region-bridge", {center: d.center, radius: d.radius});
  } else if (d.type === "viewer_ready" && e.source) {
    e.source.postMessage({type: "region_state", ...window.splatRegion}, "*");
  }
});
</script>"""


def on_bridge(text: str | None):
    """Placement from the viewer (JSON) into the five sliders; skips anything malformed."""
    values = parse_placement(text)
    if values is None:
        return [gr.skip()] * 5
    return [gr.update(value=v) for v in values]


def parse_placement(text: str | None) -> tuple[float, float, float, float, float] | None:
    try:
        d = json.loads(text or "")
        values = tuple(float(d[k]) for k in ("x", "y", "z", "turn", "size"))
    except (ValueError, TypeError, KeyError):
        return None
    if not all(math.isfinite(v) for v in values) or values[4] <= 0:
        return None
    return values  # type: ignore[return-value]


def on_generate(scene_id: str | None, number: int | None, prompt: str, seed: float | None):
    if _version(scene_id, number) is None:
        raise gr.Error("Pick a finished scene first.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Describe the object first, e.g. 'a mature oak tree'.")
    job = OBJECTS.create_object(prompt, scene_id=scene_id, seed=int(seed or 0))
    if job.status == "failed":
        return f"**Could not start.** {job.error['message']}", None, False, gr.update(visible=False)
    return (
        f"⚙️ Generating **{job.summary}** (job {job.job_id}); this takes a few minutes.",
        job.job_id,
        False,
        gr.update(visible=False),
    )


def on_object_tick(obj_job, placing, scene_id, number, hidden, show_obj):
    """Poll the object job; when it is done, show the object in the viewer for placement.

    Outputs: status, obj_job, placing, viewer, place_group, reference image, floor,
    5 sliders.
    """
    rest = [gr.skip()] * 10  # placing, viewer, group, image, floor, sliders: unchanged
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
        "(Shift-drag turns it), or use the sliders. Then place it or cancel.",
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
    ]


def on_place(scene_id, number, obj_job, x, y, z, turn, size, progress=gr.Progress()):  # noqa: B008 - Gradio's progress idiom
    """Merge the object; opens the new version. Outputs: status, then `SCENE_OUTPUTS`."""
    version = _version(scene_id, number)
    job = OBJECTS.get(obj_job) if obj_job else None
    if version is None or job is None:
        raise gr.Error("Nothing to place: generate an object for a picked scene first.")
    progress(0.1, desc="Merging the object into a new version…")
    try:
        new = place_object(STORE, version, job, Placement(x, y, z, turn, size))
    except PlaceError as e:
        raise gr.Error(str(e)) from e
    view = list(scene_view(scene_id, new.number, None, False))
    view[-3:] = [gr.update(visible=False), False, None]
    return (f"✅ Added **{job.summary}** as v{new.number}.", *view)


def on_cancel(camera_json, scene_id, number, hidden):
    version = _version(scene_id, number)
    return (
        "Placement cancelled; no new version was made.",
        scene_html(
            STORE,
            version.path if version else None,
            splat=_hidden_splat(version, hidden),
            camera=camera_from_json(camera_json),
        ),  # fmt: skip
        gr.update(visible=False),
        False,
        None,
    )


# --- edit scene (task 4.4) -----------------------------------------------------------

# Region mode on/off and size go to the page state and the viewer. Turning it off forgets
# the marked centre. Inputs: region_on, region_frac.
REGION_JS = """(on, frac) => {
  const R = window.splatRegion || (window.splatRegion = {on: false, frac: 0.2, center: null});
  R.on = !!on; R.frac = Number(frac) || R.frac;
  if (!R.on) R.center = null;
  const f = document.querySelector("iframe.splat-viewer");
  if (f && f.contentWindow) f.contentWindow.postMessage({type: "region_state", ...R}, "*");
  return [on, frac];
}"""


def _region_text(region: dict | None) -> str:
    if not region:
        return ""
    x, y, z = region["center"]
    return f"Region: centre ({x:.2f}, {y:.2f}, {z:.2f}), radius {region['radius']:.2f}"


def on_region_toggle(on: bool, _frac: float):
    """Outputs: region, region text. Off clears the marked region."""
    if not on:
        return None, ""
    return gr.skip(), "Click the scene in the viewer to mark the centre of the region."


def on_region_bridge(text: str | None):
    """Region marked in the viewer (JSON) -> region state and its description."""
    try:
        d = json.loads(text or "")
        region = {"center": [float(c) for c in d["center"]], "radius": float(d["radius"])}
    except (ValueError, TypeError, KeyError):
        return gr.skip(), gr.skip()
    return region, _region_text(region)


def on_edit(scene_id, number, prompt, seed, region_on, region):
    """Start an Edit scene job on the shown version. Outputs: status, edit_job."""
    version = _version(scene_id, number)
    if version is None:
        raise gr.Error("Pick a finished scene first.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Write an instruction first, e.g. 'make the façade green'.")
    try:
        area = edit_scene.parse_region(bool(region_on), region)
    except EditError as e:
        raise gr.Error(str(e)) from e
    job = EDITS.create_edit(
        prompt, version.path, scene_id, version.number, region=area, seed=int(seed or 0)
    )
    if job.status == "failed":
        return f"**Could not start.** {job.error['message']}", None
    where = " inside the marked region" if area else ""
    return (
        f"⚙️ Editing v{version.number}{where} (job {job.job_id}). This takes about "
        "10 to 20 minutes; you can keep working, the result becomes a new version.",
        job.job_id,
    )


def on_edit_tick(edit_job, scene_id, obj_job, placing):
    """Poll the edit job; when done, store its version and open it.

    Outputs: status, edit_job, result column, before, after, then `SCENE_OUTPUTS`.
    The new version is opened only if its scene is shown and no object is being placed.
    """
    keep = [gr.skip()] * 10
    if not edit_job:
        return [gr.skip()] * 5 + keep
    job = EDITS.get(edit_job)
    job = EDITS.refresh(job) if job else None
    if job is None:
        return ["The edit job was not found.", None, *[gr.skip()] * 3, *keep]
    if job.status == "failed":
        err = job.error or {}
        msg = f"❌ **Edit failed.** {err.get('message', '')} {err.get('tip', '')}".strip()
        return [msg, None, *[gr.skip()] * 3, *keep]
    if job.status != "done":
        label = "⏳ Waiting for a GPU" if job.status == "queued" else "⚙️ Editing"
        elapsed = format_elapsed(job.elapsed(EDITS.clock()))
        return [f"{label}: **{job.summary}** ({elapsed})", *[gr.skip()] * 4, *keep]
    try:
        new = edit_scene.finish_edit(STORE, EDITS, job)
    except edit_scene.FinishError as e:
        return [f"❌ **The edit could not be saved.** {e}", None, *[gr.skip()] * 3, *keep]
    before = STORE.download(f"{job.scene_path}/before.png")
    after = STORE.download(f"{job.scene_path}/after.png")
    images = [gr.update(visible=True), str(before) if before else None,
              str(after) if after else None]  # fmt: skip
    seconds = job.run.get("seconds")
    took = f" in {format_elapsed(seconds)}" if seconds else ""
    msg = f"✅ Edit **{job.summary}** saved as v{new.number}{took}."
    if scene_id == new.scene_id and not placing:
        gr.Info(f"Edit saved as v{new.number}; it is now shown.")
        return [msg, None, *images, *scene_view(scene_id, new.number, obj_job, placing)]
    return [f"{msg} Open it from the History tab.", None, *images, *keep]


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


def on_preview(camera_json, scene_id, number, prompt, strength, seed, placing):
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
        return f"**Could not start.** {job.error['message']}", None
    note = " The object being placed is not part of it." if placing else ""
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
    return [
        f"✅ Preview of **{job.summary}** ready{took}. The scene is unchanged.",
        None,
        gr.update(visible=True),
        str(styled),
        str(render) if render else None,
    ]


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
        return f"**Could not start the job.** {job.error['message']}", jobs_table()
    return f"**Job {job.job_id} started.** It appears in the list below.", jobs_table()


CSS = """
#placement-bridge, #camera-bridge, #region-bridge { display: none !important; }
#history-table td, #history-table th { font-size: 13px; }
#history-table tbody tr { cursor: pointer; }
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Splat Studio", css=CSS, head=BRIDGE_HEAD) as demo:
        gr.Markdown("# Splat Studio\nTurn a video or photo set into a 3D Gaussian splat scene.")
        if not DATASET:
            gr.Markdown("⚠️ `SPLAT_STUDIO_DATASET` is not set: jobs are stored locally only.")
        version, hidden, seen = gr.State(None), gr.State([]), gr.State(None)
        obj_job, placing, floor = gr.State(None), gr.State(False), gr.State(None)
        preview_job, edit_job, region = gr.State(None), gr.State(None), gr.State(None)

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
                    picker = gr.Dropdown(label="Finished scene", choices=[], value=None, scale=4)
                    open_newest = gr.Button("Open newest scene", scale=1)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        viewer = gr.HTML(scene_html(STORE, None))
                        shown = gr.Markdown()
                        objects_shown = gr.CheckboxGroup(
                            label="Objects shown (view only; downloads keep all objects)",
                            choices=[], value=[], visible=False,
                        )  # fmt: skip
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
                            with gr.Tab("Add object"):
                                obj_prompt = gr.Textbox(
                                    label="Object", placeholder="a mature oak tree"
                                )
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
                                    psize = gr.Slider(0.01, 1, value=0.25, step=0.001, label="Size")
                                    with gr.Row():
                                        place_ok = gr.Button(
                                            "Place in new version", variant="primary"
                                        )
                                        place_cancel = gr.Button("Cancel")
                            with gr.Tab("Edit scene"):
                                gr.Markdown(
                                    "Change the shown version from a text instruction. The "
                                    "result is a new version; objects added before become "
                                    "part of the scene."
                                )
                                edit_prompt = gr.Textbox(
                                    label="Instruction", placeholder="make the façade green"
                                )
                                region_on = gr.Checkbox(value=False, label="Limit to a region")
                                region_frac = gr.Slider(
                                    0.05, 1.0, value=0.2, step=0.01,
                                    label="Region size (share of the scene)",
                                )  # fmt: skip
                                region_text = gr.Markdown()
                                with gr.Row():
                                    edit_seed = gr.Number(label="Seed", value=0, precision=0)
                                    edit_go = gr.Button("Edit scene", variant="primary")
                                edit_status = gr.Markdown()
                                with gr.Column(visible=False) as edit_col:
                                    with gr.Row():
                                        edit_before = gr.Image(
                                            label="Before (one view)", interactive=False
                                        )
                                        edit_after = gr.Image(
                                            label="After (same view)", interactive=False
                                        )
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
                region_bridge = gr.Textbox(
                    elem_id="region-bridge", show_label=False, container=False
                )
        sliders = [px, py, pz, pturn, psize]
        end = [place_group, placing, obj_job]
        scene_outputs = [
            version, hidden, viewer, download, shown, objects_shown, history_table, *end
        ]  # fmt: skip
        log = (log_outputs, [picker, version], [history, log_table])

        # New scene
        files.change(on_upload, files, [message, create])
        create.click(on_create, files, [message, table])
        refresh.click(refresh_all, picker, [table, picker])

        # Start tab, finished-job notice, opening scenes and versions
        demo.load(on_load, [obj_job, placing], [tabs, table, picker, seen, *scene_outputs]).then(
            *log
        )
        gr.Timer(10).tick(on_tick, [picker, seen], [table, picker, seen])
        open_newest.click(on_open_newest, [obj_job, placing], [tabs, picker, *scene_outputs]).then(
            *log
        )
        picker.input(on_pick_scene, [picker, obj_job, placing], scene_outputs).then(*log)
        history_table.select(
            on_pick_row, [picker, history_table, obj_job, placing], scene_outputs
        ).then(*log)

        # Hide objects (view only)
        objects_shown.input(
            on_objects_shown,
            [camera_json, picker, version, objects_shown, obj_job, placing, floor, show_obj,
             *sliders],
            [hidden, viewer],
            js=GET_CAMERA_JS,
        )  # fmt: skip

        # Add object
        obj_go.click(
            on_generate,
            [picker, version, obj_prompt, obj_seed],
            [obj_status, obj_job, placing, place_group],
        )
        gr.Timer(5).tick(
            on_object_tick,
            [obj_job, placing, picker, version, hidden, show_obj],
            [obj_status, obj_job, placing, viewer, place_group, obj_ref, floor, *sliders],
        )
        for slider in sliders:
            slider.input(None, sliders, None, js=PLACEMENT_JS)
        bridge.change(on_bridge, bridge, sliders)
        show_obj.input(None, show_obj, None, js=SHOW_OBJECT_JS)
        place_ok.click(
            on_place,
            [picker, version, obj_job, *sliders],
            [obj_status, *scene_outputs],
            js=PLACE_JS,
        ).then(*log)
        place_cancel.click(
            on_cancel,
            [camera_json, picker, version, hidden],
            [obj_status, viewer, place_group, placing, obj_job],
            js=GET_CAMERA_JS,
        )

        # Edit scene
        region_on.change(
            on_region_toggle, [region_on, region_frac], [region, region_text], js=REGION_JS
        )
        region_frac.input(None, [region_on, region_frac], None, js=REGION_JS)
        region_bridge.change(on_region_bridge, region_bridge, [region, region_text])
        edit_go.click(
            on_edit,
            [picker, version, edit_prompt, edit_seed, region_on, region],
            [edit_status, edit_job],
        )
        gr.Timer(10).tick(
            on_edit_tick,
            [edit_job, picker, obj_job, placing],
            [edit_status, edit_job, edit_col, edit_before, edit_after, *scene_outputs],
        ).then(*log)

        # Preview
        pv_go.click(
            on_preview,
            [camera_json, picker, version, pv_prompt, pv_strength, pv_seed, placing],
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

        # Downloads (always the full files, whatever is hidden)
        get_ply.click(lambda j, n: on_download(j, n, "ply"), [picker, version], download)
        get_splat.click(lambda j, n: on_download(j, n, "splat"), [picker, version], download)
    return demo


if __name__ == "__main__":
    # Scene files are served from the store's download folder to the viewer.
    STORE.file_dir.mkdir(parents=True, exist_ok=True)
    build().launch(allowed_paths=[str(STORE.file_dir.resolve())])
