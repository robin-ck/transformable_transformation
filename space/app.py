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

from splat_studio.space.jobs import (  # noqa: E402
    HubStore,
    Job,
    JobManager,
    LocalStore,
    format_elapsed,
)
from splat_studio.space.validation import HELP, UploadError, validate_upload  # noqa: E402
from splat_studio.space.versions import download_file, list_versions  # noqa: E402
from splat_studio.space.viewer import scene_html  # noqa: E402

DATASET = os.environ.get("SPLAT_STUDIO_DATASET")
STORE = HubStore(DATASET, os.environ.get("HF_TOKEN")) if DATASET else LocalStore(Path("local_data"))
JOBS = JobManager(STORE)
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


def on_pick_scene(job_id: str | None):
    """Show the newest version of the picked scene and list all its versions."""
    versions = _versions(job_id)
    choices = [(v.label(), v.number) for v in versions]
    latest = versions[0] if versions else None
    return (
        gr.update(choices=choices, value=latest.number if latest else None),
        scene_html(STORE, latest.path if latest else None),
        gr.update(value=None, visible=False),
    )


def on_pick_version(job_id: str | None, number: int | None):
    version = _version(job_id, number)
    return scene_html(STORE, version.path if version else None), gr.update(
        value=None, visible=False
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

        files.change(on_upload, files, [message, create])
        create.click(on_create, files, [message, table])
        refresh.click(refresh_all, picker, [table, picker])
        gr.Timer(10).tick(refresh_all, picker, [table, picker])
        demo.load(refresh_all, picker, [table, picker])
        picker.input(on_pick_scene, picker, [version, viewer, download])
        version.input(on_pick_version, [picker, version], [viewer, download])
        get_ply.click(lambda j, n: on_download(j, n, "ply"), [picker, version], download)
        get_splat.click(lambda j, n: on_download(j, n, "splat"), [picker, version], download)
    return demo


if __name__ == "__main__":
    # Scene files are served from the store's download folder to the viewer.
    STORE.file_dir.mkdir(parents=True, exist_ok=True)
    build().launch(allowed_paths=[str(STORE.file_dir.resolve())])
