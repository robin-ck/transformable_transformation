"""Splat Studio Hugging Face Space: upload, validate, start reconstruction, track jobs.

Run locally:  python space/app.py
Secrets (Space settings): MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, HF_TOKEN, SPLAT_STUDIO_DATASET.
Without SPLAT_STUDIO_DATASET the app stores jobs in ./local_data (UI work only; the
Modal job reads from HF Hub, so reconstructions cannot finish in that mode).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

# On the Space, splat_studio is uploaded next to this file; locally use src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splat_studio.editing.objects import Placement  # noqa: E402
from splat_studio.space.add_object import PlaceError, place_object, start_placement  # noqa: E402
from splat_studio.space.jobs import (  # noqa: E402
    OBJECT_TIMEOUT_S,
    HubStore,
    Job,
    JobManager,
    LocalStore,
    format_elapsed,
)
from splat_studio.space.validation import HELP, UploadError, validate_upload  # noqa: E402
from splat_studio.space.versions import download_file, list_versions  # noqa: E402
from splat_studio.space.viewer import scene_html  # noqa: E402


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


def _versions(job_id: str | None):
    job = JOBS.get(job_id) if job_id else None
    if not job or job.status != "done":
        return []
    return list_versions(STORE, job_id, v1_created_at=job.finished_at)


def _version(job_id: str | None, number: int | None):
    return next((v for v in _versions(job_id) if v.number == number), None)


def _end_placement(obj_job: str | None, placing: bool):
    """Outputs that leave placement mode: (place_group, placing, obj_job).

    A finished object is dropped (nothing was saved); a job still generating is kept,
    and its object is shown on whichever version is picked when it is done.
    """
    return gr.update(visible=False), False, (None if placing else obj_job)


def on_pick_scene(job_id: str | None, obj_job: str | None, placing: bool):
    """Show the newest version of the picked scene and list all its versions."""
    versions = _versions(job_id)
    choices = [(v.label(), v.number) for v in versions]
    latest = versions[0] if versions else None
    return (
        gr.update(choices=choices, value=latest.number if latest else None),
        scene_html(STORE, latest.path if latest else None),
        gr.update(value=None, visible=False),
        *_end_placement(obj_job, placing),
    )


def on_pick_version(job_id: str | None, number: int | None, obj_job: str | None, placing: bool):
    version = _version(job_id, number)
    return (
        scene_html(STORE, version.path if version else None),
        gr.update(value=None, visible=False),
        *_end_placement(obj_job, placing),
    )


# --- add object (task 4.2) ----------------------------------------------------------

# Sends slider values to the viewer iframe, which moves the object without reloading.
PLACEMENT_JS = """(x, y, z, turn, size) => {
  const f = document.querySelector("iframe.splat-viewer");
  if (f && f.contentWindow) {
    f.contentWindow.postMessage({type: "placement", x, y, z, turn, size}, "*");
  }
  return [x, y, z, turn, size];
}"""


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


def on_object_tick(obj_job: str | None, placing: bool, scene_id: str | None, number):
    """Poll the object job; when it is done, show the object in the viewer for placement.

    Outputs: status, obj_job, placing, viewer, place_group, reference image, 5 sliders.
    """
    rest = [gr.skip()] * 9  # placing, viewer, group, image, sliders: unchanged
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
        STORE, version.path, object_path=job.scene_path, placement=placement.to_viewer()
    )
    return [
        f"✅ **{job.summary}** is ready. Move it with the sliders, then place it or cancel.",
        gr.skip(),
        True,
        viewer,
        gr.update(visible=True),
        str(reference) if reference else None,
        gr.update(minimum=ranges.x[0], maximum=ranges.x[1], value=placement.x),
        gr.update(minimum=ranges.y[0], maximum=ranges.y[1], value=placement.y),
        gr.update(minimum=ranges.z[0], maximum=ranges.z[1], value=placement.z),
        gr.update(value=0),
        gr.update(minimum=ranges.size[0], maximum=ranges.size[1], value=placement.size),
    ]


def on_place(scene_id, number, obj_job, x, y, z, turn, size, progress=gr.Progress()):  # noqa: B008 - Gradio's progress idiom
    version = _version(scene_id, number)
    job = OBJECTS.get(obj_job) if obj_job else None
    if version is None or job is None:
        raise gr.Error("Nothing to place: generate an object for a picked scene first.")
    progress(0.1, desc="Merging the object into a new version…")
    try:
        new = place_object(STORE, version, job, Placement(x, y, z, turn, size))
    except PlaceError as e:
        raise gr.Error(str(e)) from e
    versions = _versions(scene_id)
    return (
        f"✅ Added **{job.summary}** as v{new.number}.",
        gr.update(choices=[(v.label(), v.number) for v in versions], value=new.number),
        scene_html(STORE, new.path),
        gr.update(visible=False),
        False,
        None,
    )


def on_cancel(scene_id, number):
    version = _version(scene_id, number)
    return (
        "Placement cancelled; no new version was made.",
        scene_html(STORE, version.path if version else None),
        gr.update(visible=False),
        False,
        None,
    )


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


def build() -> gr.Blocks:
    with gr.Blocks(title="Splat Studio") as demo:
        gr.Markdown("# Splat Studio\nTurn a video or photo set into a 3D Gaussian splat scene.")
        if not DATASET:
            gr.Markdown("⚠️ `SPLAT_STUDIO_DATASET` is not set: jobs are stored locally only.")
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

        gr.Markdown("## Scene")
        picker = gr.Dropdown(label="Finished scene", choices=[], value=None)
        version = gr.Radio(label="Version (newest first)", choices=[], value=None)
        viewer = gr.HTML(scene_html(STORE, None))
        with gr.Row():
            get_ply = gr.Button("Download .ply")
            get_splat = gr.Button("Download .splat (compressed)")
        download = gr.File(label="Download", visible=False, interactive=False)

        gr.Markdown("## Add object")
        obj_job, placing = gr.State(None), gr.State(False)
        with gr.Row():
            obj_prompt = gr.Textbox(label="Object", placeholder="a mature oak tree", scale=4)
            obj_seed = gr.Number(label="Seed", value=0, precision=0, scale=1)
        obj_go = gr.Button("Generate object")
        obj_status = gr.Markdown()
        with gr.Group(visible=False) as place_group:
            with gr.Row():
                obj_ref = gr.Image(label="Reference image", interactive=False, height=220)
                with gr.Column(scale=2):
                    px = gr.Slider(-1, 1, value=0, step=0.001, label="Move x")
                    py = gr.Slider(-1, 1, value=0, step=0.001, label="Move y")
                    pz = gr.Slider(-1, 1, value=0, step=0.001, label="Height (base)")
                    pturn = gr.Slider(-180, 180, value=0, step=1, label="Turn (degrees)")
                    psize = gr.Slider(0.01, 1, value=0.25, step=0.001, label="Size")
            with gr.Row():
                place_ok = gr.Button("Place in new version", variant="primary")
                place_cancel = gr.Button("Cancel")
        sliders = [px, py, pz, pturn, psize]

        files.change(on_upload, files, [message, create])
        create.click(on_create, files, [message, table])
        refresh.click(refresh_all, picker, [table, picker])
        gr.Timer(10).tick(refresh_all, picker, [table, picker])
        demo.load(refresh_all, picker, [table, picker])
        end = [place_group, placing, obj_job]
        picker.input(on_pick_scene, [picker, obj_job, placing], [version, viewer, download, *end])
        version.input(
            on_pick_version, [picker, version, obj_job, placing], [viewer, download, *end]
        )
        obj_go.click(
            on_generate,
            [picker, version, obj_prompt, obj_seed],
            [obj_status, obj_job, placing, place_group],
        )
        gr.Timer(5).tick(
            on_object_tick,
            [obj_job, placing, picker, version],
            [obj_status, obj_job, placing, viewer, place_group, obj_ref, *sliders],
        )
        for slider in sliders:
            slider.input(None, sliders, None, js=PLACEMENT_JS)
        place_ok.click(
            on_place,
            [picker, version, obj_job, *sliders],
            [obj_status, version, viewer, place_group, placing, obj_job],
        )
        place_cancel.click(
            on_cancel, [picker, version], [obj_status, viewer, place_group, placing, obj_job]
        )
        get_ply.click(lambda j, n: on_download(j, n, "ply"), [picker, version], download)
        get_splat.click(lambda j, n: on_download(j, n, "splat"), [picker, version], download)
    return demo


if __name__ == "__main__":
    # Scene files are served from the store's download folder to the viewer.
    STORE.file_dir.mkdir(parents=True, exist_ok=True)
    build().launch(allowed_paths=[str(STORE.file_dir.resolve())])
