"""Make generated objects smaller by merging neighbouring Gaussians.

TRELLIS gives about 1-1.5 M small Gaussians per object (32 per voxel of a 64^3 grid).
Merged into a scene, that roughly doubles the scene file. Dropping the weakest
Gaussians thins out fine detail such as leaves, so instead we put a grid over the
object and merge each cell's Gaussians into one by moment matching:

  weight w_i  = opacity_i * area_i           (how much the Gaussian shows)
  mean        = sum w_i mu_i / W
  covariance  = sum w_i (Sigma_i + (mu_i - mean)(mu_i - mean)^T) / W
  colour      = sum w_i c_i / W
  opacity     = min(1 - prod(1 - a_i), sum a_i area_i / area)   (never more than the union)

The grid is the finest one that brings the object under the target count. A cell
with one Gaussian keeps it unchanged.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TARGET = 250_000
GRID_SIZES = (512, 384, 256, 192, 160, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1)
_MAX_OPACITY = 0.995


def _rotation_matrices(q: np.ndarray) -> np.ndarray:
    """(N, 4) quaternions (w, x, y, z), any length -> (N, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )


def _quaternions(r: np.ndarray) -> np.ndarray:
    """(N, 3, 3) proper rotation matrices -> (N, 4) unit quaternions (w, x, y, z)."""
    m00, m11, m22 = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]
    q = np.stack(
        [
            np.sqrt(np.maximum(0, 1 + m00 + m11 + m22)) / 2,
            np.sqrt(np.maximum(0, 1 + m00 - m11 - m22)) / 2,
            np.sqrt(np.maximum(0, 1 - m00 + m11 - m22)) / 2,
            np.sqrt(np.maximum(0, 1 - m00 - m11 + m22)) / 2,
        ],
        axis=1,
    )
    # Signs from the off-diagonals, relative to the largest component (numerically safest).
    big = np.argmax(q, axis=1)
    pairs = {
        0: (r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]),
        1: (r[:, 2, 1] - r[:, 1, 2], r[:, 0, 1] + r[:, 1, 0], r[:, 0, 2] + r[:, 2, 0]),
        2: (r[:, 0, 2] - r[:, 2, 0], r[:, 0, 1] + r[:, 1, 0], r[:, 1, 2] + r[:, 2, 1]),
        3: (r[:, 1, 0] - r[:, 0, 1], r[:, 0, 2] + r[:, 2, 0], r[:, 1, 2] + r[:, 2, 1]),
    }
    others = {0: (1, 2, 3), 1: (0, 2, 3), 2: (0, 1, 3), 3: (0, 1, 2)}
    out = q.copy()
    for k in range(4):
        sel = big == k
        if not sel.any():
            continue
        denom = 4 * q[sel, k]
        for col, num in zip(others[k], pairs[k], strict=True):
            out[sel, col] = num[sel] / denom
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _cells(p: np.ndarray, target: int) -> np.ndarray:
    """Cell index per Gaussian for the finest grid that gives at most `target` cells."""
    lo, hi = np.percentile(p, 0.1, axis=0), np.percentile(p, 99.9, axis=0)
    span = float(np.max(hi - lo)) or 1.0
    for n in GRID_SIZES:
        ijk = np.clip(np.floor((p - lo) / (span / n)), -1, n).astype(np.int64) + 1
        key = (ijk[:, 0] * (n + 2) + ijk[:, 1]) * (n + 2) + ijk[:, 2]
        _, inverse = np.unique(key, return_inverse=True)
        if inverse.max() + 1 <= target:
            return inverse
    return inverse  # coarsest grid; still the best we can do


def simplify(v: np.ndarray, target: int = DEFAULT_TARGET) -> np.ndarray:
    """Copy of Gaussians `v` merged down to at most about `target` (same fields)."""
    if len(v) <= target:
        return v.copy()
    p = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    s = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64))
    a = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    a = np.clip(a, 1e-6, _MAX_OPACITY)
    area = np.cbrt(s.prod(axis=1)) ** 2
    w = a * area

    group = _cells(p, target)
    n = group.max() + 1

    def total(x: np.ndarray) -> np.ndarray:
        return np.bincount(group, weights=x, minlength=n)

    weight = total(w)
    mean = np.stack([total(w * p[:, k]) for k in range(3)], axis=1) / weight[:, None]

    rot = _rotation_matrices(np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64))
    cov = np.einsum("nij,nj,nkj->nik", rot, s**2, rot)
    d = p - mean[group]
    cov_sum = np.empty((n, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            c = total(w * (cov[:, i, j] + d[:, i] * d[:, j])) / weight
            cov_sum[:, i, j] = cov_sum[:, j, i] = c

    vals, vecs = np.linalg.eigh(cov_sum)
    vecs[np.linalg.det(vecs) < 0, :, 0] *= -1  # proper rotation (det +1)
    scales = np.sqrt(np.maximum(vals, 1e-12))
    new_area = np.cbrt(scales.prod(axis=1)) ** 2
    union = 1 - np.exp(total(np.log1p(-a)))
    coverage = total(a * area) / new_area
    alpha = np.clip(np.minimum(union, coverage), 1e-4, _MAX_OPACITY)

    out = np.zeros(n, dtype=v.dtype)
    out["x"], out["y"], out["z"] = mean.T
    for name in v.dtype.names:
        if name.startswith("f_dc_") or name.startswith("f_rest_"):
            out[name] = total(w * v[name].astype(np.float64)) / weight
    out["opacity"] = np.log(alpha / (1 - alpha))
    for i in range(3):
        out[f"scale_{i}"] = np.log(scales[:, i])
    q = _quaternions(vecs)
    for i in range(4):
        out[f"rot_{i}"] = q[:, i]
    return out
