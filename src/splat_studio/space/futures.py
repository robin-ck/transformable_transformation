"""Futures (add-viewpoint-futures): restyled images of saved viewpoints.

A future is a preview job (`previews/<job_id>/`, see `jobs.py`) whose `params` also hold
  viewpoint   the viewpoint id (`viewpoints.Viewpoint.id`)
  view_prefix the viewpoint's folder, whose render and depth the restyle started from
  starter     {"id", "label", "text"} of the starter direction used, or None
  own_prompt  the user's own text; `prompt` is what the editor got (starter first)
  parent      the job id of the future it was built on, or None
  editor      the `models.EDITORS` key
Old previews have no `viewpoint` and are not futures. Pure logic, no GPU.
"""

from __future__ import annotations

import time
from pathlib import Path

from splat_studio import models

from .jobs import FINAL, Job, JobManager
from .viewpoints import Starter, Viewpoint

MAX_PROMPT = 300  # SDXL reads at most 77 tokens; the starter comes first, so it is kept
THUMB_SIDE = 512
VIEWER_SIDE = 1280  # image laid over the 3D viewer (sent inline as a data URL)


class FutureError(ValueError):
    """A future cannot be started as asked; the message says why."""


def compose_prompt(starter: Starter | None, own: str | None) -> str:
    """The prompt sent to the editor: the starter's text, then the user's own text."""
    own = " ".join((own or "").split())
    parts = [starter.text.strip() if starter else "", own]
    prompt = ", ".join(p for p in parts if p)
    if not prompt:
        raise FutureError("Type what should change, or pick a starter direction.")
    if len(prompt) > MAX_PROMPT:
        raise FutureError(
            f"The prompt is {len(prompt)} characters long (starter included); the limit is "
            f"{MAX_PROMPT}. Shorten your text."
        )
    return prompt


def future_params(
    vp: Viewpoint,
    view_prefix: str,
    starter: Starter | None,
    own: str | None,
    parent: Job | None = None,
    editor: str | None = None,
) -> dict:
    """The future-specific fields of the preview job's `params` (see the module doc)."""
    spec = models.editor_spec(editor)
    if parent is not None and parent.params.get("viewpoint") != vp.id:
        raise FutureError("A future can only be built on a future of the same viewpoint.")
    return {
        "viewpoint": vp.id,
        "view_prefix": view_prefix,
        "starter": {"id": starter.id, "label": starter.label, "text": starter.text}
        if starter
        else None,
        "own_prompt": " ".join((own or "").split()),
        "parent": parent.job_id if parent else None,
        "editor": spec.key,
        "model": spec.label,
    }


def summary(params: dict) -> str:
    """Short name of a future: the user's own text, else the starter's label."""
    own = params.get("own_prompt") or ""
    starter = (params.get("starter") or {}).get("label")
    return (own or starter or params.get("prompt") or "")[:80]


def is_future(job: Job) -> bool:
    return job.kind == "preview" and bool(job.params.get("viewpoint"))


def viewpoint_futures(jobs: list[Job], vp_id: str | None) -> list[Job]:
    """Finished futures of one viewpoint, newest first (failed and running ones left out)."""
    mine = [j for j in jobs if is_future(j) and j.params["viewpoint"] == vp_id]
    return sorted((j for j in mine if j.status == "done"), key=lambda j: j.created_at, reverse=True)


class FutureIndex:
    """The preview jobs of the dataset, keeping finished records in memory.

    Finished jobs never change, so a refresh reads only folders it has not seen finished.
    """

    def __init__(self, manager: JobManager) -> None:
        self.manager = manager
        self.final: dict[str, Job] = {}

    def put(self, job: Job) -> None:
        if job.status in FINAL:
            self.final[job.job_id] = job

    def jobs(self) -> list[Job]:
        out = []
        for job_id in self.manager.store.list_dirs(self.manager.prefix):
            job = self.final.get(job_id) or self.manager.get(job_id)
            if job is None:
                continue
            self.put(job)
            out.append(job)
        return out


def caption(job: Job, chars: int = 48) -> str:
    """Gallery caption: the start of the future's own text (or starter label)."""
    text = summary(job.params)
    return text if len(text) <= chars else text[: chars - 1].rstrip() + "…"


def _date(t: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(t)) if t else "date unknown"


def details_markdown(job: Job, vp_name: str | None = None) -> str:
    """Full prompt, starter, date and provenance of one future (gallery, Compare)."""
    p = job.params
    starter = p.get("starter") or {}
    lines = [f"**{_date(job.created_at)}** · view **{vp_name or p.get('viewpoint', '?')}**"]
    lines.append(f"**Prompt:** {p.get('own_prompt') or '(starter only)'}")
    if starter:
        lines.append(f"**Starter:** {starter.get('label')} ({starter.get('text')})")
    if p.get("parent"):
        lines.append(f"**Built on:** future `{p['parent']}`")
    model = job.run.get("model") or p.get("model")
    lines.append(f"**Editor:** {model} · seed {p.get('seed', 0)} · job `{job.job_id}`")
    return "  \n".join(lines)


def thumbnail(src: Path, dest: Path, side: int = THUMB_SIDE) -> Path:
    """A JPEG copy of `src` whose longer side is at most `side` (made once, then reused)."""
    from PIL import Image

    dest = Path(dest)
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((side, side))
        img.save(dest, format="JPEG", quality=85)
    return dest


def data_url(path: Path) -> str:
    """A JPEG file as a `data:` URL, for the viewer overlay (no file route needed)."""
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()
