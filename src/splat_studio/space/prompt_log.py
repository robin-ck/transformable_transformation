"""Prompt log (task 4.5): every prompt, its mode, model and result, per scene.

Nothing new is stored for this: each edit already writes `version.json` (prompt,
mode, model, parent, date; task 3.4) and each preview job keeps its prompt and
source version in `previews/<job>/job.json` (task 4.3). This module reads both
and shows them as one log, plus the chain of edits that led to a version.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .jobs import Job
from .versions import Version

MODE_LABEL = {
    "reconstruction": "reconstruction",
    "add_object": "add object",
    "edit_scene": "edit scene",
    "preview": "preview (2D)",
}


@dataclass(frozen=True)
class LogEntry:
    created_at: float | None
    mode: str
    prompt: str | None
    model: str | None
    result: str  # "v3", or "preview of v2 (job ...)"
    status: str = "done"

    def row(self) -> list[str]:
        return [
            _date(self.created_at),
            MODE_LABEL.get(self.mode, self.mode),
            self.prompt or "",
            self.model or "",
            self.result if self.status == "done" else f"{self.result} [{self.status}]",
        ]


COLUMNS = ["Date", "Mode", "Prompt", "Model", "Result"]


def _date(t: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(t)) if t else "date unknown"


def _version_number(scene_path: str | None, scene_id: str) -> int | None:
    """`scenes/<scene_id>/v<n>` -> n; None for another scene or an odd path."""
    prefix = f"scenes/{scene_id}/v"
    if not scene_path or not scene_path.startswith(prefix):
        return None
    tail = scene_path[len(prefix) :]
    return int(tail) if tail.isdigit() else None


def scene_log(
    scene_id: str, versions: list[Version], previews: list[Job], edits: list[Job] = ()
) -> list[LogEntry]:
    """All prompts for one scene, newest first.

    Versions (every edit that produced one), previews made from its versions, and edit
    jobs that have not (yet) produced a version: running or failed ones. A finished edit
    is listed once, as its version.
    """
    entries = [LogEntry(v.created_at, v.mode, v.prompt, v.model, f"v{v.number}") for v in versions]
    for job in edits:
        if job.status == "done" or job.params.get("scene_id") != scene_id:
            continue
        entries.append(
            LogEntry(
                job.created_at,
                "edit_scene",
                job.params.get("prompt") or job.summary,
                job.params.get("model"),
                f"edit of v{job.params.get('version')} (job {job.job_id})",
                job.status,
            )
        )
    for job in previews:
        n = _version_number(job.params.get("scene_path"), scene_id)
        if n is None:
            continue
        entries.append(
            LogEntry(
                job.created_at,
                "preview",
                job.params.get("prompt") or job.summary,
                job.run.get("model") or job.params.get("model"),  # failed runs lack it
                f"preview of v{n} (job {job.job_id})",
                job.status,
            )
        )
    return sorted(entries, key=lambda e: e.created_at or 0, reverse=True)


def lineage(versions: list[Version], number: int) -> list[Version]:
    """The chain of versions that led to `number`, oldest first (follows `parent`)."""
    by_number = {v.number: v for v in versions}
    chain, seen = [], set()
    current = by_number.get(number)
    while current is not None and current.number not in seen:
        seen.add(current.number)
        chain.append(current)
        current = by_number.get(current.parent) if current.parent else None
    return chain[::-1]


def history_markdown(versions: list[Version], number: int | None) -> str:
    """'History of vN': prompt and model of the version and each step before it."""
    chain = lineage(versions, number) if number is not None else []
    if not chain:
        return ""
    lines = [f"**History of v{number}**", ""]
    for v in chain:
        what = MODE_LABEL.get(v.mode, v.mode)
        prompt = f' "{v.prompt}"' if v.prompt else ""
        model = f" · model `{v.model}`" if v.model else ""
        lines.append(f"- v{v.number} · {_date(v.created_at)} · {what}{prompt}{model}")
    return "\n".join(lines)
