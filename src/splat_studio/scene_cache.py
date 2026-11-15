"""Scene files cached on the Modal volume (speed-up-edit-jobs, task 2).

Versions never change once written, so a cached copy of `scenes/<id>/v<n>/<file>` stays
valid for good. Layout: `<root>/<repo_path>`, plus a `.last_used` stamp in each version
folder. When the cache grows over `cap_bytes`, the least recently used version folders go.

No Modal import: the caller commits and reloads the volume. Tested with a fake downloader.
"""

from __future__ import annotations

import os
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path

CACHE_CAP_GB = 10.0
STAMP = ".last_used"


class SceneCache:
    def __init__(
        self,
        root: Path,
        cap_bytes: int = int(CACHE_CAP_GB * 1e9),
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root, self.cap_bytes, self.clock = Path(root), cap_bytes, clock

    def _path(self, repo_path: str) -> Path:
        parts = Path(repo_path).parts
        if not parts or Path(repo_path).is_absolute() or ".." in parts:
            raise ValueError(f"not a dataset path: {repo_path!r}")
        return self.root / repo_path

    def _touch(self, folder: Path) -> None:
        (folder / STAMP).write_text(repr(self.clock()))

    def has(self, repo_path: str) -> bool:
        return self._path(repo_path).is_file()

    def fetch(self, repo_path: str, download: Callable[[str], str | Path]) -> tuple[Path, bool]:
        """The cached file for `repo_path`, downloading it on a miss. Returns (path, hit).

        `download(repo_path)` returns a local file; it is copied to a temporary name and
        renamed when complete, so another container never reads half a file.
        """
        dest = self._path(repo_path)
        if dest.is_file():
            self._touch(dest.parent)
            return dest, True
        self._store(dest, Path(download(repo_path)))
        return dest, False

    def put(self, repo_path: str, local_file: Path) -> Path:
        """Copy a file we already have (e.g. a freshly edited scene) into the cache."""
        dest = self._path(repo_path)
        if not dest.is_file():
            self._store(dest, Path(local_file))
        else:
            self._touch(dest.parent)
        return dest

    def _store(self, dest: Path, src: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.part")
        try:
            shutil.copyfile(src, tmp)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        self._touch(dest.parent)
        self.trim(keep=dest.parent)

    def _folders(self) -> list[tuple[float, int, Path]]:
        """(last used, bytes, folder) for each version folder that holds a stamp."""
        out = []
        for stamp in self.root.rglob(STAMP):
            folder = stamp.parent
            try:
                used = float(stamp.read_text() or 0)
            except (OSError, ValueError):
                used = 0.0
            size = sum(
                f.stat().st_size for f in folder.iterdir() if f.is_file() and f.name != STAMP
            )
            out.append((used, size, folder))
        return out

    def size_bytes(self) -> int:
        return sum(size for _, size, _ in self._folders())

    def trim(self, keep: Path | None = None) -> list[Path]:
        """Delete least recently used version folders until the cache is under the cap."""
        folders = sorted(self._folders(), key=lambda t: t[0])
        total = sum(size for _, size, _ in folders)
        removed = []
        for _, size, folder in folders:
            if total <= self.cap_bytes:
                break
            if keep is not None and folder == keep:
                continue
            shutil.rmtree(folder, ignore_errors=True)
            total -= size
            removed.append(folder)
        return removed
