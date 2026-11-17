"""PreviewRenderer.render and render_view (add-viewpoint-futures) with the GPU, the
editor and the Hub stubbed out. Needs `modal` to load modal_app/app.py; skipped without it."""

import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("modal")
huggingface_hub = pytest.importorskip("huggingface_hub")
from huggingface_hub.errors import EntryNotFoundError  # noqa: E402
from PIL import Image  # noqa: E402

from splat_studio.reconstruction.cost import LoadTimes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class FakeEditor:
    key = "sdxl-depth"

    def restyle(self, image, depth, prompt, seed, **kw):
        assert image.size == depth.size
        return Image.new("RGB", image.size, (seed % 255, 0, 0))


def test_futures_render_once_per_viewpoint_and_commit_once(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "modal_app"))
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(get_device_name=lambda i: "NVIDIA L4")
    monkeypatch.setitem(sys.modules, "torch", torch)
    import app

    REPO: dict[str, bytes] = {}
    COMMITS: list[list[str]] = []
    RENDERS: list[str] = []

    class FakeApi:
        def create_commit(self, repo_id, operations, commit_message, repo_type):
            for op in operations:
                src = op.path_or_fileobj
                REPO[op.path_in_repo] = src if isinstance(src, bytes) else Path(src).read_bytes()
            COMMITS.append(sorted(op.path_in_repo for op in operations))

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
            REPO[path_in_repo] = path_or_fileobj
            COMMITS.append([path_in_repo])

    def fake_download(repo, path, repo_type, cache_dir):
        if path not in REPO:
            raise EntryNotFoundError(path)
        out = Path(cache_dir) / path.replace("/", "_")
        out.write_bytes(REPO[path])
        return str(out)

    def fake_render(self, repo, scene_path, camera, stage):
        RENDERS.append(scene_path)
        stage("render", 0.1)
        return (Image.new("RGB", (64, 64), "green"), Image.new("RGB", (64, 64), "gray"),
                {"camera": camera, "size": [64, 64], "source_scene": scene_path,
                 "num_gaussians": 5, "scene_cached": True})  # fmt: skip

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setenv("SPLAT_STUDIO_DATASET", "me/data")
    Cls = app.PreviewRenderer._get_user_cls()
    monkeypatch.setattr(Cls, "_render", fake_render)
    w = Cls.__new__(Cls)
    w.loads, w.editor, w.views, w.images = LoadTimes(), FakeEditor(), {}, {}
    render_view = Cls.render_view._get_raw_f()
    render = Cls.render._get_raw_f()

    cam = {
        "c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "fov_deg": 50,
        "aspect": 1,
    }
    render_view = Cls.render_view._get_raw_f()
    render = Cls.render._get_raw_f()

    # 1. Save a viewpoint.
    res = render_view(w, "scenes/s/v1", cam, "viewpoints/s/a")
    assert res["status"] == "done" and res["image_jpeg"][:2] == b"\xff\xd8", res
    assert COMMITS[-1] == [
        "viewpoints/s/a/depth.png",
        "viewpoints/s/a/render.png",
        "viewpoints/s/a/view.json",
    ], COMMITS
    # 2. Future of it: no render, only preview.png + run.json committed.
    n = len(RENDERS)
    res = render(w, "j1", "scenes/s/v1", cam, "reeds", seed=3, view_prefix="viewpoints/s/a")
    assert res["status"] == "done", res
    assert len(RENDERS) == n and res["view_rendered"] is False
    assert COMMITS[-1] == ["previews/j1/preview.png", "previews/j1/run.json"], COMMITS[-1]
    assert json.loads(REPO["previews/j1/run.json"])["status"] == "done"
    # 3. Cold container: the view comes from the dataset, not a render.
    w.views.clear()
    res = render(w, "j2", "scenes/s/v1", cam, "x", view_prefix="viewpoints/s/a")
    assert res["status"] == "done" and len(RENDERS) == n
    # 4. New viewpoint from the current view: render + view files + preview in one commit.
    res = render(w, "j3", "scenes/s/v1", cam, "weir", view_prefix="viewpoints/s/b")
    assert len(RENDERS) == n + 1 and res["view_rendered"] is True
    assert COMMITS[-1] == [
        "previews/j3/preview.png",
        "previews/j3/run.json",
        "viewpoints/s/b/depth.png",
        "viewpoints/s/b/render.png",
        "viewpoints/s/b/view.json",
    ], COMMITS[-1]
    # 5. Build on a future: input from memory; from the dataset after a restart.
    res = render(
        w,
        "j4",
        "scenes/s/v1",
        cam,
        "bridge",
        view_prefix="viewpoints/s/a",
        input_path="previews/j1/preview.png",
    )
    assert res["status"] == "done" and res["input_path"] == "previews/j1/preview.png"
    w.images.clear()
    res = render(
        w,
        "j5",
        "scenes/s/v1",
        cam,
        "bridge",
        view_prefix="viewpoints/s/a",
        input_path="previews/j1/preview.png",
    )
    assert res["status"] == "done", res
    # 6. Old-style preview still uploads render and depth in its own folder.
    res = render(w, "j6", "scenes/s/v1", cam, "autumn")
    assert COMMITS[-1] == [
        "previews/j6/depth.png",
        "previews/j6/preview.png",
        "previews/j6/render.png",
        "previews/j6/run.json",
    ], COMMITS[-1]
    # 7. Missing parent image: failed, and the failed run.json is the last write.
    res = render(
        w,
        "j7",
        "scenes/s/v1",
        cam,
        "x",
        view_prefix="viewpoints/s/a",
        input_path="previews/nope/preview.png",
    )
    assert res["status"] == "failed" and COMMITS[-1] == ["previews/j7/run.json"]
    assert json.loads(REPO["previews/j7/run.json"])["status"] == "failed"


def test_every_editor_has_a_modal_class_running_it(monkeypatch) -> None:
    from splat_studio import models

    monkeypatch.syspath_prepend(str(ROOT / "modal_app"))
    import app

    for key, spec in models.EDITORS.items():
        cls = getattr(app, spec.modal_class)._get_user_cls()
        assert cls.EDITOR == key
