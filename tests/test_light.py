import html
from pathlib import Path

import numpy as np

from splat_studio.reconstruction.export import to_splat_bytes
from splat_studio.space import light, viewer
from splat_studio.space.jobs import LocalStore


def _splat_file(path: Path, n: int, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=light.SPLAT_DTYPE)
    s["pos"] = rng.normal(size=(n, 3)).astype(np.float32)
    s["scale"] = rng.uniform(0.005, 0.05, size=(n, 3)).astype(np.float32)
    s["rgba"] = rng.integers(0, 256, size=(n, 4), dtype=np.uint8)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    s["rot"] = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    s.tofile(path)
    return path


def test_read_splat_round_trips_through_the_exporter(tmp_path: Path) -> None:
    src = _splat_file(tmp_path / "a.splat", 500)
    a = np.fromfile(src, dtype=light.SPLAT_DTYPE)
    b = np.frombuffer(to_splat_bytes(light.read_splat(src)), dtype=light.SPLAT_DTYPE)
    ka, kb = np.lexsort(a["pos"].T), np.lexsort(b["pos"].T)  # the exporter reorders
    assert np.array_equal(a["pos"][ka], b["pos"][kb])
    assert np.allclose(a["scale"][ka], b["scale"][kb], rtol=1e-4)
    assert np.abs(a["rgba"][ka].astype(int) - b["rgba"][kb]).max() <= 1
    assert np.abs(a["rot"][ka].astype(int) - b["rot"][kb]).max() <= 1


def test_small_scenes_are_shown_as_they_are(tmp_path: Path) -> None:
    src = _splat_file(tmp_path / "a.splat", 300)
    assert light.light_splat(src, tmp_path / "light", target=300) == src
    assert not (tmp_path / "light").exists()


def test_big_scenes_get_a_cached_lighter_copy(tmp_path: Path) -> None:
    src = _splat_file(tmp_path / "a.splat", 20_000)
    out = light.light_splat(src, tmp_path / "light", target=2_000)
    count = out.stat().st_size // light.SPLAT_DTYPE.itemsize
    assert out.parent == tmp_path / "light"
    assert 1_000 < count <= 2_000  # close to the target, not far below it
    mtime = out.stat().st_mtime_ns
    assert light.light_splat(src, tmp_path / "light", target=2_000) == out
    assert out.stat().st_mtime_ns == mtime  # reused, not rebuilt
    assert not list((tmp_path / "light").glob("*.part"))


def test_a_changed_scene_gets_a_new_copy(tmp_path: Path) -> None:
    src = _splat_file(tmp_path / "a.splat", 20_000)
    first = light.light_splat(src, tmp_path / "light", target=2_000)
    _splat_file(src, 21_000, seed=1)
    assert light.light_splat(src, tmp_path / "light", target=2_000) != first


def test_scene_grid_is_the_finest_that_fits() -> None:
    p = np.random.default_rng(0).normal(size=(20_000, 3))
    n = light.scene_grid(p, 2_000)
    assert light.grid_cells(p, n).max() + 1 <= 2_000
    assert light.grid_cells(p, int(n * 1.05) + 1).max() + 1 > 2_000


def test_scene_html_shows_the_lighter_copy(tmp_path: Path, monkeypatch) -> None:
    store = LocalStore(tmp_path)
    _splat_file(tmp_path / "scenes/j1/v1/scene.splat", 20_000)
    real = light.light_splat
    monkeypatch.setattr(viewer, "light_splat", lambda src, cache: real(src, cache, target=2_000))
    out = html.unescape(viewer.scene_html(store, "scenes/j1/v1"))
    assert "/light/" in out
    assert "scenes/j1/v1/scene.splat" not in out


def test_scene_html_falls_back_to_the_full_scene(tmp_path: Path, monkeypatch) -> None:
    store = LocalStore(tmp_path)
    _splat_file(tmp_path / "scenes/j1/v1/scene.splat", 300)

    def broken(src, cache):
        raise MemoryError

    monkeypatch.setattr(viewer, "light_splat", broken)
    out = html.unescape(viewer.scene_html(store, "scenes/j1/v1"))
    assert "scenes/j1/v1/scene.splat" in out
