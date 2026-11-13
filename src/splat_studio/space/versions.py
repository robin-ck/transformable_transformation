"""Scene versions: list, add, and fetch files for download (task 3.4).

Every version is an immutable folder in the store; edits add a folder and never
change an old one, so switching back to an earlier version loses nothing:
  scenes/<scene_id>/v1/scene.ply, scene.splat          reconstruction (worker)
  scenes/<scene_id>/v<n>/scene.ply, scene.splat, version.json   later edits

`version.json` holds the prompt, mode, model, parent version and date. The
reconstruction worker does not write one for v1, so v1 falls back to the job's
finish time and mode "reconstruction".
"""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .jobs import Store

FILES = {"ply": "scene.ply", "splat": "scene.splat"}
_VERSION_DIR = re.compile(r"^v([1-9]\d*)$")


@dataclass(frozen=True)
class Version:
    scene_id: str
    number: int
    created_at: float | None = None
    mode: str = "reconstruction"  # reconstruction | add_object | edit_scene
    prompt: str | None = None
    model: str | None = None
    parent: int | None = None

    @property
    def path(self) -> str:
        return f"scenes/{self.scene_id}/v{self.number}"

    def label(self) -> str:
        date = (
            time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(self.created_at))
            if self.created_at
            else "date unknown"
        )
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
    for n in numbers:
        meta = store.read_json(f"scenes/{scene_id}/v{n}/version.json") or {}
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
            )
        )
    return out


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
) -> Version:
    """Store a new version after the newest one; existing versions are never touched.

    `files` maps a kind in `FILES` ("ply", "splat") to a local file. `extra` is written
    to `version.json` as well (e.g. the object job and placement that made the version).
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
    )
    store.upload_files(
        {f"{version.path}/{FILES[k]}": Path(p) for k, p in files.items()},
        f"Scene {scene_id} v{number} ({mode})",
    )
    meta = {k: v for k, v in asdict(version).items() if k not in {"scene_id", "number"}}
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
