"""Convert a 3DGS `.ply` into the compact `.splat` format used by web viewers.

`.splat` layout per Gaussian (32 bytes): position 3xfloat32, scale 3xfloat32,
RGBA 4xuint8, rotation quaternion (w, x, y, z) 4xuint8.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814
_PLY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
}  # fmt: skip


def read_ply(path: Path) -> np.ndarray:
    """Read the vertex element of a binary little-endian PLY as a structured array."""
    with path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        fields, count, in_vertex = [], 0, False
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            parts = line.split()
            if parts[0] == "format" and parts[1] != "binary_little_endian":
                raise ValueError(f"unsupported PLY format: {parts[1]}")
            if parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                fields.append((parts[2], "<" + _PLY_TYPES[parts[1]]))
        return np.fromfile(f, dtype=np.dtype(fields), count=count)


def to_splat_bytes(v: np.ndarray) -> bytes:
    """Encode Gaussians as `.splat`, sorted by size x opacity (largest first)."""
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1))
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    order = np.argsort(-(scales.prod(axis=1) * opacity), kind="stable")

    rgb = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    rgba = np.concatenate([rgb, opacity[:, None]], axis=1)
    rgba = np.clip(rgba * 255, 0, 255).astype(np.uint8)

    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float64)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    rot_u8 = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)

    out = np.zeros(len(v), dtype=[("pos", "<f4", 3), ("scale", "<f4", 3),
                                  ("rgba", "u1", 4), ("rot", "u1", 4)])  # fmt: skip
    out["pos"] = np.stack([v["x"], v["y"], v["z"]], axis=1)
    out["scale"] = scales
    out["rgba"] = rgba
    out["rot"] = rot_u8
    return out[order].tobytes()


def ply_to_splat(ply: Path, splat: Path) -> int:
    """Write `splat` from `ply`; returns the number of Gaussians."""
    v = read_ply(ply)
    splat.write_bytes(to_splat_bytes(v))
    return len(v)


_NP_TO_PLY = {"f4": "float", "f8": "double", "u1": "uchar", "i4": "int", "u4": "uint"}


def write_ply(path: Path, v: np.ndarray) -> None:
    """Write a structured array as the vertex element of a binary little-endian PLY."""
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {len(v)}"]
    for name in v.dtype.names:
        kind = v.dtype[name]
        lines.append(f"property {_NP_TO_PLY[kind.kind + str(kind.itemsize)]} {name}")
    lines.append("end_header")
    le = v.astype(v.dtype.newbyteorder("<"), copy=False)
    with path.open("wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        le.tofile(f)
