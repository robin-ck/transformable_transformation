from pathlib import Path

import pytest

from splat_studio.scene_cache import STAMP, SceneCache


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        self.t += 1
        return self.t


def downloader(src: Path, size: int = 10):
    calls = []

    def download(repo_path: str) -> Path:
        calls.append(repo_path)
        f = src / repo_path.replace("/", "_")
        f.write_bytes(b"x" * size)
        return f

    return download, calls


def test_fetch_downloads_once_then_hits(tmp_path: Path) -> None:
    cache = SceneCache(tmp_path / "c", clock=Clock())
    download, calls = downloader(tmp_path)
    path, hit = cache.fetch("scenes/s/v1/scene.ply", download)
    assert not hit and path == tmp_path / "c/scenes/s/v1/scene.ply" and path.read_bytes()
    again, hit = cache.fetch("scenes/s/v1/scene.ply", download)
    assert hit and again == path and calls == ["scenes/s/v1/scene.ply"]
    assert (path.parent / STAMP).is_file()
    assert not [f for f in path.parent.iterdir() if f.name.endswith(".part")]


def test_put_then_fetch_skips_download(tmp_path: Path) -> None:
    cache = SceneCache(tmp_path / "c")
    local = tmp_path / "edited.ply"
    local.write_bytes(b"new scene")
    cache.put("scenes/s/v4/scene.ply", local)
    download, calls = downloader(tmp_path)
    path, hit = cache.fetch("scenes/s/v4/scene.ply", download)
    assert hit and calls == [] and path.read_bytes() == b"new scene"


def test_failed_download_leaves_nothing(tmp_path: Path) -> None:
    cache = SceneCache(tmp_path / "c")

    def broken(_):
        raise ConnectionError("lost")

    with pytest.raises(ConnectionError):
        cache.fetch("scenes/s/v1/scene.ply", broken)
    assert not cache.has("scenes/s/v1/scene.ply")


def test_trim_drops_least_recently_used_versions(tmp_path: Path) -> None:
    cache = SceneCache(tmp_path / "c", cap_bytes=25, clock=Clock())
    download, _ = downloader(tmp_path)
    cache.fetch("scenes/a/v1/scene.ply", download)
    cache.fetch("scenes/b/v1/scene.ply", download)
    cache.fetch("scenes/a/v1/scene.ply", download)  # a is now newer than b
    cache.fetch("scenes/c/v1/scene.ply", download)  # 30 bytes > 25: b goes
    assert cache.has("scenes/a/v1/scene.ply") and cache.has("scenes/c/v1/scene.ply")
    assert not cache.has("scenes/b/v1/scene.ply")
    assert cache.size_bytes() == 20


def test_trim_keeps_the_file_just_written(tmp_path: Path) -> None:
    cache = SceneCache(tmp_path / "c", cap_bytes=5, clock=Clock())
    download, _ = downloader(tmp_path)
    path, _ = cache.fetch("scenes/a/v1/scene.ply", download)
    assert path.is_file()


@pytest.mark.parametrize("bad", ["", "/etc/passwd", "scenes/../../x"])
def test_rejects_paths_outside_the_cache(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        SceneCache(tmp_path / "c").fetch(bad, lambda _: tmp_path)
