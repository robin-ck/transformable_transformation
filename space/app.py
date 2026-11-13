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

from splat_studio.space.jobs import HubStore, JobManager, LocalStore, format_elapsed  # noqa: E402
from splat_studio.space.validation import HELP, UploadError, validate_upload  # noqa: E402

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


def jobs_table() -> list[list[str]]:
    now = JOBS.clock()
    rows = []
    for j in JOBS.list():
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
        table = gr.Dataframe(headers=COLUMNS, value=jobs_table, interactive=False, wrap=True)
        refresh = gr.Button("Refresh")

        files.change(on_upload, files, [message, create])
        create.click(on_create, files, [message, table])
        refresh.click(jobs_table, None, table)
        gr.Timer(10).tick(jobs_table, None, table)
    return demo


if __name__ == "__main__":
    build().launch()
