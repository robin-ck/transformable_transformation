"""Lighter copies of `.splat` scenes for the browser viewer (view only).

The browser viewer keeps several copies of every Gaussian (parsed data, data textures,
GPU textures, the sort worker's centres), so a 1.6 M Gaussian scene takes several
hundred MB of the client's memory. Scenes above `VIEW_TARGET` Gaussians are shown as a
copy merged down with `editing.simplify` (neighbouring Gaussians merged by moment
matching, so fine detail softens rather than disappears). Downloads, edits and object
placement keep using the full files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from splat_studio.editing.simplify import grid_cells, simplify
from splat_studio.reconstruction.export import SH_C0, to_splat_bytes

VIEW_TARGET = 500_000
CACHE_VERSION = 1  # bump when the merge changes, so old copies are not reused
SPLAT_DTYPE = np.dtype(
    [("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)]
)
_FIELDS = [
    *(("x", "y", "z")),
    *(f"scale_{i}" for i in range(3)),
    *(f"f_dc_{i}" for i in range(3)),
    "opacity",
    *(f"rot_{i}" for i in range(4)),
]


def read_splat(path: Path) -> np.ndarray:
    """Gaussians of a `.splat` file in PLY field form (as `simplify` and the exporter use)."""
    s = np.fromfile(path, dtype=SPLAT_DTYPE)
    v = np.zeros(len(s), dtype=[(name, "<f4") for name in _FIELDS])
    for i, axis in enumerate("xyz"):
        v[axis] = s["pos"][:, i]
    for i in range(3):
        v[f"scale_{i}"] = np.log(np.maximum(s["scale"][:, i], 1e-12))
        v[f"f_dc_{i}"] = (s["rgba"][:, i] / 255.0 - 0.5) / SH_C0
    a = np.clip(s["rgba"][:, 3] / 255.0, 1e-4, 1 - 1e-4)
    v["opacity"] = np.log(a / (1 - a))
    for i in range(4):
        v[f"rot_{i}"] = (s["rot"][:, i].astype(np.float32) - 128) / 128
    return v


def light_splat(src: Path, cache_dir: Path, target: int = VIEW_TARGET) -> Path:
    """`src`, or a cached copy merged down to about `target` Gaussians if it has more.

    The cache name depends on the source path, size and modification time, so a changed
    file gets a new copy.
    """
    if src.stat().st_size // SPLAT_DTYPE.itemsize <= target:
        return src
    st = src.stat()
    key = f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}|{target}|{CACHE_VERSION}"
    name = hashlib.sha1(key.encode()).hexdigest()[:16]
    out = Path(cache_dir) / f"{name}.splat"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    v = read_splat(src)
    n = scene_grid(np.stack([v["x"], v["y"], v["z"]], axis=1), target)
    tmp.write_bytes(to_splat_bytes(simplify(v, target, (n,))))
    tmp.replace(out)
    return out


def scene_grid(p: np.ndarray, target: int, largest: int = 1 << 16) -> int:
    """Finest grid (cells a side) that merges `p` into at most `target` cells.

    Bisection, since the cell count grows with the grid size. `simplify`'s own grid steps
    are made for single objects; for a whole scene they can land far below `target`.
    """

    def cells(n: int) -> int:
        return int(grid_cells(p, n).max()) + 1

    lo, hi = 1, largest  # cells(lo) <= target is assumed (one cell, plus border cells)
    if cells(hi) <= target:
        return hi
    while hi - lo > max(1, lo // 50):  # within 2 %
        mid = (lo + hi) // 2
        if cells(mid) <= target:
            lo = mid
        else:
            hi = mid
    return lo
