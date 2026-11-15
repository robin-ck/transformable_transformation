"""Scene versions: list, add, and fetch files for download (task 3.4).

Every version is an immutable folder in the store; edits add a folder and never
change an old one, so switching back to an earlier version loses nothing:
  scenes/<scene_id>/v1/scene.ply, scene.splat          reconstruction (worker)
  scenes/<scene_id>/v<n>/scene.ply, scene.splat, version.json   later edits

`version.json` holds the prompt, mode, model, parent version and date. The
reconstruction worker does not write one for v1, so v1 falls back to the job's
finish time and mode "reconstruction".

Versions made with Add object also list the objects added up to them
(`objects: [{id, prompt, object_job, start, count, placement}]`, inherited from the parent;
entries written before `placement` was kept get it from the version that added them).
`start` and `count` are Gaussian indexes in `scene.splat` (32 bytes each), so an object
can be hidden in the viewer by dropping its byte range (`filtered_splat`). Versions
made before this existed have no list, so their objects cannot be hidden.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .jobs import Store

FILES = {"ply": "scene.ply", "splat": "scene.splat"}
SPLAT_RECORD_BYTES = 32  # see reconstruction/export.py
MODE_LABEL = {
    "reconstruction": "reconstruction",
    "add_object": "add object",
    "edit_scene": "edit scene",
    "move_object": "move object",
}
HISTORY_COLUMNS = ["Version", "Date", "Mode", "Prompt", "Model", "From"]
_VERSION_DIR = re.compile(r"^v([1-9]\d*)$")


@dataclass(frozen=True)
class Version:
    scene_id: str
    number: int
    created_at: float | None = None
    mode: str = "reconstruction"  # reconstruction | add_object | edit_scene | move_object
    prompt: str | None = None
    model: str | None = None
    parent: int | None = None
    # Objects added up to this version: ({id, prompt, object_job, start, count}, ...).
    objects: tuple[dict, ...] = field(default=(), compare=False)

    @property
    def path(self) -> str:
        return f"scenes/{self.scene_id}/v{self.number}"

    def label(self) -> str:
        date = _date(self.created_at)
        what = self.mode.replace("_", " ")
        if self.prompt:
            what += f': "{_shorten(self.prompt, 60)}"'
        return f"v{self.number} · {date} · {what}"


def _shorten(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def list_versions(store: Store, scene_id: str, v1_created_at: float | None = None) -> list[Version]:
    """All versions of a scene, newest first."""
    numbers = sorted(
        (
            int(m.group(1))
            for d in store.list_dirs(f"scenes/{scene_id}")
            if (m := _VERSION_DIR.match(d))
        ),
        reverse=True,
    )
    out = []
    placements: dict[tuple, dict] = {}  # (id, object job) -> placement, for old entries
    metas = {n: store.read_json(f"scenes/{scene_id}/v{n}/version.json") or {} for n in numbers}
    for meta in metas.values():
        objs = _objects(meta.get("objects"))
        if meta.get("mode") == "add_object" and objs and isinstance(meta.get("placement"), dict):
            placements.setdefault((objs[-1]["id"], objs[-1].get("object_job")), meta["placement"])
    for n in numbers:
        meta = metas[n]
        if n == 1 and meta.get("created_at") is None:
            meta["created_at"] = v1_created_at
        out.append(
            Version(
                scene_id,
                n,
                created_at=meta.get("created_at"),
                mode=meta.get("mode") or "reconstruction",
                prompt=meta.get("prompt"),
                model=meta.get("model"),
                parent=meta.get("parent"),
                objects=tuple(
                    o
                    if o.get("placement")
                    else {**o, "placement": placements.get((o["id"], o.get("object_job")))}
                    for o in _objects(meta.get("objects"))
                ),
            )
        )
    return out


def _objects(raw) -> list[dict]:
    """Valid entries of a stored `objects` list; anything malformed is skipped."""
    out = []
    for o in raw if isinstance(raw, list) else []:
        try:
            start, count = int(o["start"]), int(o["count"])
        except (KeyError, TypeError, ValueError):
            continue
        if start >= 0 and count > 0 and o.get("id"):
            out.append({**o, "start": start, "count": count})
    return out


def _date(t: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(t)) if t else "date unknown"


def history_rows(versions: list[Version], v1_input: str | None = None) -> list[list[str]]:
    """Rows for the History table (newest first, as `versions` is).

    v1 has no prompt; it shows `v1_input` (the reconstruction job's input summary).
    """
    rows = []
    for v in versions:
        prompt = v.prompt or (v1_input if v.number == 1 and v.mode == "reconstruction" else "")
        rows.append(
            [
                f"v{v.number}",
                _date(v.created_at),
                MODE_LABEL.get(v.mode, v.mode.replace("_", " ")),
                " ".join((prompt or "").split()),
                v.model or "",
                f"v{v.parent}" if v.parent else "",
            ]
        )
    return rows


def version_from_row(row: list | tuple | None) -> int | None:
    """Version number from a History row (its first cell, "v3")."""
    if not row:
        return None
    cell = str(row[0]).strip()
    return int(cell[1:]) if cell[:1] == "v" and cell[1:].isdigit() else None


def add_version(
    store: Store,
    scene_id: str,
    files: dict[str, Path],
    *,
    mode: str,
    prompt: str | None = None,
    model: str | None = None,
    parent: int | None = None,
    now: float | None = None,
    extra: dict | None = None,
    objects: list[dict] | None = None,
) -> Version:
    """Store a new version after the newest one; existing versions are never touched.

    `files` maps a kind in `FILES` ("ply", "splat") to a local file. `extra` is written
    to `version.json` as well (e.g. the object job and placement that made the version).
    `objects` is the list of added objects with their Gaussian ranges (see module doc).
    """
    unknown = set(files) - set(FILES)
    if unknown:
        raise ValueError(f"unknown file kinds: {sorted(unknown)}")
    existing = list_versions(store, scene_id)
    number = (existing[0].number + 1) if existing else 1
    version = Version(
        scene_id,
        number,
        created_at=time.time() if now is None else now,
        mode=mode,
        prompt=prompt,
        model=model,
        parent=parent,
        objects=tuple(objects or ()),
    )
    store.upload_files(
        {f"{version.path}/{FILES[k]}": Path(p) for k, p in files.items()},
        f"Scene {scene_id} v{number} ({mode})",
    )
    meta = {k: v for k, v in asdict(version).items() if k not in {"scene_id", "number"}}
    meta["objects"] = list(version.objects) if version.objects else None
    if meta["objects"] is None:
        del meta["objects"]
    meta.update({k: v for k, v in (extra or {}).items() if k not in meta})
    store.write_json(f"{version.path}/version.json", meta)
    return version


def download_file(store: Store, version: Version, kind: str) -> Path | None:
    """Local copy of a version's file, named `<scene>-v<n>.<ext>` so downloads stay distinct.

    Returns None if the version has no such file.
    """
    src = store.download(f"{version.path}/{FILES[kind]}")
    if src is None:
        return None
    target = Path(store.file_dir) / "downloads" / f"{version.scene_id}-v{version.number}.{kind}"
    if not (target.is_file() and target.stat().st_size == src.stat().st_size):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        try:
            os.link(src, target)  # no second copy of a several-hundred-MB file
        except OSError:
            shutil.copyfile(src, target)
    return target


def filtered_splat(store: Store, version: Version, hidden: list[str] | set[str]) -> Path | None:
    """Local `.splat` of `version` without the hidden objects' Gaussians (view only).

    Cached under `<file_dir>/hidden/<scene>-v<n>-<ids>.splat`; downloads keep using the
    full files. Returns the full scene file if nothing known is hidden, None if the
    version has no `.splat`.
    """
    src = store.download(f"{version.path}/{FILES['splat']}")
    if src is None:
        return None
    drop = sorted(
        ((o["start"], o["count"]) for o in version.objects if o["id"] in set(hidden)),
    )
    if not drop:
        return src
    ids = "-".join(sorted(o["id"] for o in version.objects if o["id"] in set(hidden)))
    target = Path(store.file_dir) / "hidden" / f"{version.scene_id}-v{version.number}-{ids}.splat"
    if target.is_file():
        return target
    total = src.stat().st_size // SPLAT_RECORD_BYTES
    keep, pos = [], 0
    for start, count in drop:
        if start > pos:
            keep.append((pos, start))
        pos = max(pos, start + count)
    if pos < total:
        keep.append((pos, total))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    with src.open("rb") as f, tmp.open("wb") as out:
        for a, b in keep:
            f.seek(a * SPLAT_RECORD_BYTES)
            remaining = (b - a) * SPLAT_RECORD_BYTES
            while remaining:
                chunk = f.read(min(remaining, 1 << 24))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)
    tmp.replace(target)
    return target


def view_splat(
    store: Store, version: Version, versions: list[Version], hidden: list[str] | set[str]
) -> Path | None:
    """Local `.splat` to view `version` with hidden objects ("o2") and edits ("e5").

    An edit cannot be cut out, so hiding edit vN shows vN's parent instead, plus the
    objects placed after the last edit (`version.objects`; edits keep no object list,
    so objects placed between two edits are part of the later edit). With several
    hidden edits the earliest one counts. View only; downloads keep the full files.
    """
    hidden = set(hidden or ())
    edits = {int(h[1:]) for h in hidden if h[:1] == "e" and h[1:].isdigit()}
    by_number = {v.number: v for v in versions}
    line, current = [], version
    while current is not None and current not in line:
        line.append(current)
        current = by_number.get(current.parent) if current.parent else None
    cut = [v for v in line if v.number in edits and v.mode == "edit_scene" and v.parent]
    parent = by_number.get(cut[-1].parent) if cut else None  # line is newest first
    if parent is None:
        return filtered_splat(store, version, hidden)
    base = store.download(f"{parent.path}/{FILES['splat']}")
    full = store.download(f"{version.path}/{FILES['splat']}")
    if base is None or full is None:
        return None
    keep = [o for o in version.objects if o["id"] not in hidden]
    ids = "-".join([f"e{cut[-1].number}", *sorted(o["id"] for o in keep)])
    target = Path(store.file_dir) / "hidden" / f"{version.scene_id}-v{version.number}-x{ids}.splat"
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    with tmp.open("wb") as out:
        out.write(base.read_bytes())
        with full.open("rb") as f:
            for o in keep:
                f.seek(o["start"] * SPLAT_RECORD_BYTES)
                out.write(f.read(o["count"] * SPLAT_RECORD_BYTES))
    tmp.replace(target)
    return target
