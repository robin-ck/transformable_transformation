import numpy as np
import pytest

from splat_studio.editing import simplify as simp

FIELDS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
FIELDS += ["scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def random_gaussians(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.zeros(n, dtype=[(f, "<f4") for f in FIELDS])
    p = rng.uniform(0, 1, (n, 3))
    v["x"], v["y"], v["z"] = p.T
    for i in range(3):
        v[f"f_dc_{i}"] = rng.normal(size=n)
        v[f"scale_{i}"] = np.log(rng.uniform(0.001, 0.01, n))
    v["opacity"] = rng.normal(size=n)
    q = rng.normal(size=(n, 4))
    for i in range(4):
        v[f"rot_{i}"] = q[:, i]
    return v


def covariances(v: np.ndarray) -> np.ndarray:
    r = simp._rotation_matrices(np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(float))
    s = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(float))
    return np.einsum("nij,nj,nkj->nik", r, s**2, r)


def test_quaternion_matrix_roundtrip() -> None:
    q = np.random.default_rng(1).normal(size=(500, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    back = simp._quaternions(simp._rotation_matrices(q))
    assert np.allclose(np.abs(np.sum(q * back, axis=1)), 1, atol=1e-9)  # same up to sign


def test_small_object_is_left_alone() -> None:
    v = random_gaussians(100)
    assert np.array_equal(simp.simplify(v, target=100), v)


def test_reaches_target_and_keeps_layout() -> None:
    v = random_gaussians(20_000)
    out = simp.simplify(v, target=2_000)
    assert 0 < len(out) <= 2_000 and out.dtype == v.dtype
    for k in "xyz":
        assert out[k].min() >= v[k].min() - 1e-6 and out[k].max() <= v[k].max() + 1e-6
    assert np.isfinite(np.stack([out[f] for f in FIELDS], 1)).all()


def test_lone_gaussian_keeps_its_shape() -> None:
    v = random_gaussians(3)
    v["x"] = [0.0, 0.5, 1.0]  # far apart: one per cell
    out = simp.simplify(v, target=2)  # forces merging to run, but cells stay single
    # one cell may hold two at a coarse grid; find the untouched one by position
    lone = [i for i in range(len(out)) if any(np.isclose(out["x"][i], v["x"]))]
    assert lone
    for i in lone:
        j = int(np.argmin(np.abs(v["x"] - out["x"][i])))
        assert np.allclose(covariances(out[i : i + 1]), covariances(v[j : j + 1]), rtol=1e-4)
        assert out["opacity"][i] == pytest.approx(v["opacity"][j], abs=1e-4)


def test_two_stacked_gaussians_merge_to_union_opacity() -> None:
    v = random_gaussians(2)
    for f in FIELDS:
        v[f][1] = v[f][0]
    v["opacity"] = 0.0  # alpha 0.5 each
    out = simp.simplify(v, target=1)
    assert len(out) == 1
    alpha = 1 / (1 + np.exp(-out["opacity"][0]))
    assert alpha == pytest.approx(0.75, abs=1e-4)  # 1 - 0.5 * 0.5
    assert np.allclose(covariances(out), covariances(v[:1]), rtol=1e-4)
