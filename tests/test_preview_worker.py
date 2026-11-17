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


def _worker(monkeypatch, cls_name: str = "PreviewRenderer", editor=None, masker=None):
    """A Modal class instance with GPU, editor and Hub stubbed: (worker, REPO, COMMITS,
    RENDERS); REPO is the fake dataset (path -> bytes)."""
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
    Cls = getattr(app, cls_name)._get_user_cls()
    monkeypatch.setattr(Cls, "_render", fake_render)
    w = Cls.__new__(Cls)
    w.loads, w.editor, w.views, w.images = LoadTimes(), editor or FakeEditor(), {}, {}
    w.masker = masker
    return w, REPO, COMMITS, RENDERS


CAM = {"c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "fov_deg": 50,
       "aspect": 1}  # fmt: skip


def test_futures_render_once_per_viewpoint_and_commit_once(monkeypatch) -> None:
    w, REPO, COMMITS, RENDERS = _worker(monkeypatch)
    Cls = type(w)
    cam = CAM
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


class FakeMasker:
    """Finds "chairs" as the box x 10..30, y 20..40 of a 64 x 64 image; nothing else."""

    labels = ["det@1", "sam@2"]

    def __init__(self):
        self.calls = []

    def detect(self, image, query):
        from splat_studio.editing.masking import Detection

        self.calls.append(("detect", query))
        return [Detection((10, 20, 30, 40), 0.8, "chairs")] if "chair" in query else []

    def segment(self, image, detections):
        import numpy as np

        self.calls.append(("segment", len(detections)))
        hard = np.zeros(image.size[::-1], bool)
        hard[20:40, 10:30] = True
        return hard


class RedEditor:
    key = "flux-kontext"

    def __init__(self):
        self.calls = 0

    def restyle(self, image, depth, prompt, seed, **kw):
        self.calls += 1
        return Image.new("RGB", image.size, (255, 0, 0))  # changes everything


def test_object_mask_keeps_the_rest_of_the_view(monkeypatch) -> None:
    import io

    import numpy as np

    masker, editor = FakeMasker(), RedEditor()
    w, REPO, COMMITS, _ = _worker(monkeypatch, "KontextRenderer", editor, masker)
    render_view = type(w).render_view._get_raw_f()
    render = type(w).render._get_raw_f()
    render_view(w, "scenes/s/v1", CAM, "viewpoints/s/a")

    # 1. Masked future: red inside the chairs' mask, the green render everywhere else.
    res = render(w, "m1", "scenes/s/v1", CAM, "make the chairs red", view_prefix="viewpoints/s/a",
                 object_query="  chairs ")  # fmt: skip
    assert res["status"] == "done", res
    assert COMMITS[-1] == ["previews/m1/mask.png", "previews/m1/preview.png",
                           "previews/m1/run.json"]  # fmt: skip
    out = np.asarray(Image.open(io.BytesIO(REPO["previews/m1/preview.png"])))
    mask = np.asarray(Image.open(io.BytesIO(REPO["previews/m1/mask.png"])))
    assert res["mask_png"] == REPO["previews/m1/mask.png"]
    assert (out[mask == 0] == [0, 128, 0]).all()  # PIL "green" is (0, 128, 0)
    assert (out[30, 20] == [255, 0, 0]).all() and (out[0, 0] == [0, 128, 0]).all()
    run = json.loads(REPO["previews/m1/run.json"])
    assert {"detect", "segment", "restyle", "composite"} <= set(run["stages"])
    assert run["object_query"] == "chairs"
    assert run["mask"]["detections"] == 1 and run["mask"]["boxes"] == [[10, 20, 30, 40]]
    assert run["mask"]["models"] == ["det@1", "sam@2"] and run["mask"]["removal"] is False

    # 2. Not found: fails before the edit, with a message naming the object.
    n = editor.calls
    res = render(w, "m2", "scenes/s/v1", CAM, "paint the boat", view_prefix="viewpoints/s/a",
                 object_query="boat")  # fmt: skip
    assert res["status"] == "failed" and res["error"]["code"] == "object_not_found"
    assert '"boat"' in res["error"]["message"] and editor.calls == n
    assert COMMITS[-1] == ["previews/m2/run.json"]
    assert "detect" in json.loads(REPO["previews/m2/run.json"])["stages"]

    # 3. No query: no mask, Kontext's image as it is.
    res = render(w, "m3", "scenes/s/v1", CAM, "make it red", view_prefix="viewpoints/s/a")
    assert res["status"] == "done" and "mask_png" not in res and res["mask"] is None
    assert COMMITS[-1] == ["previews/m3/preview.png", "previews/m3/run.json"]

    # 4. An editor without a masker refuses the query.
    w.masker = None
    res = render(w, "m4", "scenes/s/v1", CAM, "x", view_prefix="viewpoints/s/a",
                 object_query="chairs")  # fmt: skip
    assert res["status"] == "failed" and "cannot limit" in res["error"]["tip"]
