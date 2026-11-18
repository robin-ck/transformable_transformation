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
        masks = np.zeros((len(detections), *image.size[::-1]), bool)
        masks[0, 20:40, 10:30] = True
        return masks


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
    assert COMMITS[-1] == ["previews/m1/detections.jpg", "previews/m1/mask.png",
                           "previews/m1/preview.png", "previews/m1/run.json"]  # fmt: skip
    assert res["detections_jpeg"][:2] == b"\xff\xd8"
    labelled = Image.open(io.BytesIO(REPO["previews/m1/detections.jpg"]))
    assert labelled.format == "JPEG" and labelled.size == (64, 64)
    out = np.asarray(Image.open(io.BytesIO(REPO["previews/m1/preview.png"])))
    mask = np.asarray(Image.open(io.BytesIO(REPO["previews/m1/mask.png"])))
    assert res["mask_png"] == REPO["previews/m1/mask.png"]
    assert (out[mask == 0] == [0, 128, 0]).all()  # PIL "green" is (0, 128, 0)
    assert (out[30, 20] == [255, 0, 0]).all() and (out[0, 0] == [0, 128, 0]).all()
    run = json.loads(REPO["previews/m1/run.json"])
    assert {"detect", "segment", "annotate", "restyle", "composite"} <= set(run["stages"])
    assert run["object_query"] == "chairs" and run["prompt"] == "make the chairs red"
    assert run["editor_prompt"].startswith("make the chairs red. Keep the shape")
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
    assert "detections_jpeg" not in res
    assert COMMITS[-1] == ["previews/m3/preview.png", "previews/m3/run.json"]

    # 4. An editor without a masker refuses the query.
    w.masker = None
    res = render(w, "m4", "scenes/s/v1", CAM, "x", view_prefix="viewpoints/s/a",
                 object_query="chairs")  # fmt: skip
    assert res["status"] == "failed" and "cannot limit" in res["error"]["tip"]


class CropEditor(RedEditor):
    """Paints its whole input red and records the size and prompt it got."""

    def restyle(self, image, depth, prompt, seed, **kw):
        self.got = (image.size, depth.size, prompt)
        return super().restyle(image, depth, prompt, seed, **kw)


def test_drawn_area_edits_a_crop_and_keeps_the_rest(monkeypatch) -> None:
    import io

    import numpy as np

    from splat_studio.editing import area

    editor = CropEditor()
    w, REPO, COMMITS, _ = _worker(monkeypatch, "KontextRenderer", editor, FakeMasker())
    render_view = type(w).render_view._get_raw_f()
    render = type(w).render._get_raw_f()
    render_view(w, "scenes/s/v1", CAM, "viewpoints/s/a")

    # Drawn at half the render's size: the worker scales it to the start image.
    hard = np.zeros((32, 32), bool)
    hard[4:8, 4:8] = True  # pixels 8..16 of the 64 x 64 render
    res = render(w, "a1", "scenes/s/v1", CAM, "a willow", view_prefix="viewpoints/s/a",
                 area_png=area.to_png(hard))  # fmt: skip
    assert res["status"] == "done", res
    assert COMMITS[-1] == ["previews/a1/area.png", "previews/a1/mask.png",
                           "previews/a1/preview.png", "previews/a1/run.json"]  # fmt: skip
    run = json.loads(REPO["previews/a1/run.json"])
    x0, y0, x1, y1 = run["area"]["crop"]
    assert editor.got[0] == editor.got[1] == (x1 - x0, y1 - y0) and x1 - x0 < 64
    assert editor.got[2].startswith("a willow. Make the change in the middle")
    assert run["area"]["bounds"] == [8, 8, 16, 16] and run["mask"] is None
    assert {"area", "restyle", "composite"} <= set(run["stages"])
    out = np.asarray(Image.open(io.BytesIO(REPO["previews/a1/preview.png"])))
    mask = np.asarray(Image.open(io.BytesIO(REPO["previews/a1/mask.png"])))
    assert res["mask_png"] == REPO["previews/a1/mask.png"]
    assert (out[mask == 0] == [0, 128, 0]).all() and (out[12, 12] == [255, 0, 0]).all()
    stored = area.from_png(REPO["previews/a1/area.png"])
    assert stored.shape == (64, 64) and stored[12, 12] and not stored[30, 30]

    # Empty area, area plus object, and an editor without a masker are refused.
    n = editor.calls
    res = render(w, "a2", "scenes/s/v1", CAM, "x", view_prefix="viewpoints/s/a",
                 area_png=area.to_png(np.zeros((32, 32), bool)))  # fmt: skip
    assert res["status"] == "failed" and res["error"]["code"] == "area_empty"
    res = render(w, "a3", "scenes/s/v1", CAM, "x", view_prefix="viewpoints/s/a",
                 object_query="chairs", area_png=area.to_png(hard))  # fmt: skip
    assert res["status"] == "failed" and "both" in res["error"]["tip"]
    w.masker = None
    res = render(w, "a4", "scenes/s/v1", CAM, "x", view_prefix="viewpoints/s/a",
                 area_png=area.to_png(hard))  # fmt: skip
    assert res["status"] == "failed" and "cannot limit" in res["error"]["tip"]
    assert editor.calls == n


def test_detect_only_labels_the_plain_render_and_stores_nothing(monkeypatch) -> None:
    masker = FakeMasker()
    w, REPO, COMMITS, _ = _worker(monkeypatch, "KontextRenderer", RedEditor(), masker)
    detect = type(w).detect._get_raw_f()
    type(w).render_view._get_raw_f()(w, "scenes/s/v1", CAM, "viewpoints/s/a")
    n = len(COMMITS)

    res = detect(w, "viewpoints/s/a", " chairs ")
    assert res["status"] == "done", res
    assert res["detections_jpeg"][:2] == b"\xff\xd8"
    assert res["mask"]["labels"] == ["chairs"] and res["mask"]["query"] == "chairs"
    assert {"detect", "segment", "annotate"} <= set(res["stages"])
    assert len(COMMITS) == n  # nothing written to the dataset

    res = detect(w, "viewpoints/s/a", "boat")
    assert res["status"] == "failed" and res["error"]["code"] == "object_not_found"
    res = detect(w, "viewpoints/s/missing", "chairs")
    assert res["status"] == "failed" and "no stored render" in res["error"]["tip"]
    w.masker = None
    res = detect(w, "viewpoints/s/a", "chairs")
    assert res["status"] == "failed" and "cannot limit" in res["error"]["tip"]


def test_scene_instructions_reach_the_editor(monkeypatch) -> None:
    """edit-editor-instructions: the scene's own texts replace the defaults."""
    import numpy as np

    from splat_studio.editing import area

    class Recorder(CropEditor):
        def restyle(self, image, depth, prompt, seed, **kw):
            self.kw = kw
            return super().restyle(image, depth, prompt, seed, **kw)

    editor = Recorder()
    w, REPO, _, _ = _worker(monkeypatch, "KontextRenderer", editor, FakeMasker())
    type(w).render_view._get_raw_f()(w, "scenes/s/v1", CAM, "viewpoints/s/a")
    render = type(w).render._get_raw_f()
    hard = np.zeros((64, 64), bool)
    hard[8:24, 8:24] = True
    texts = {"area_change": "Put it in the middle.", "negative": "cartoon"}
    res = render(w, "i1", "scenes/s/v1", CAM, "a willow", view_prefix="viewpoints/s/a",
                 area_png=area.to_png(hard), instructions=texts)  # fmt: skip
    assert res["status"] == "done", res
    assert editor.got[2] == "a willow. Put it in the middle."
    assert editor.kw["negative_prompt"] == "cartoon"
    run = json.loads(REPO["previews/i1/run.json"])
    assert run["instructions"]["area_change"] == "Put it in the middle."
    assert run["instructions"]["object_change"].startswith("Keep the shape")


def test_preview_tool_uses_its_own_negative_prompt(monkeypatch) -> None:
    class Recorder(FakeEditor):
        def restyle(self, image, depth, prompt, seed, **kw):
            self.kw = kw
            return super().restyle(image, depth, prompt, seed, **kw)

    editor = Recorder()
    w, REPO, _, _ = _worker(monkeypatch, editor=editor)
    render = type(w).render._get_raw_f()
    texts = {"preview_negative": "cartoon", "negative": "other"}
    res = render(w, "p1", "scenes/s/v1", CAM, "autumn", instructions=texts)
    assert res["status"] == "done", res
    assert editor.kw["negative_prompt"] == "cartoon"
    assert json.loads(REPO["previews/p1/run.json"])["negative_prompt"] == "cartoon"


class RewritingEditor:
    """Like Step1X-Edit: returns an image and keeps its rewritten prompt in last_info."""

    key = "step1x-edit"

    def __init__(self):
        self.last_info = {}

    def restyle(self, image, depth, prompt, seed, **kw):
        self.last_info = {"editor_prompt_rewritten": f"Rewritten: {prompt}"}
        return Image.new("RGB", image.size, (0, 0, 255))


@pytest.mark.parametrize("cls_name", ["QwenEditRenderer", "Flux2KleinRenderer",
                                      "Step1XRenderer"])  # fmt: skip
def test_new_instruction_editors_mask_and_record_extra_info(monkeypatch, cls_name) -> None:
    # add-futures-editors: same masking as Kontext; what the editor returns besides the
    # image (Step1X's rewritten prompt) goes into run.json.
    import numpy as np

    masker, editor = FakeMasker(), RewritingEditor()
    w, REPO, COMMITS, _ = _worker(monkeypatch, cls_name, editor, masker)
    render = type(w).render._get_raw_f()
    res = render(w, "q1", "scenes/s/v1", CAM, "make the chairs blue", object_query="chairs")
    assert res["status"] == "done", res
    run = json.loads(REPO["previews/q1/run.json"])
    assert run["editor"] == type(w).EDITOR and run["negative_prompt"] is None
    assert run["editor_prompt_rewritten"].startswith("Rewritten: make the chairs blue")
    out = np.asarray(Image.open(__import__("io").BytesIO(REPO["previews/q1/preview.png"])))
    assert (out[0, 0] == [0, 128, 0]).all() and (out[30, 20] == [0, 0, 255]).all()


def test_union_editor_restyles_the_whole_view_without_a_mask(monkeypatch) -> None:
    w, REPO, _, _ = _worker(monkeypatch, "SdxlUnionRenderer")
    render = type(w).render._get_raw_f()
    res = render(w, "u1", "scenes/s/v1", CAM, "in 2050", seed=7)
    assert res["status"] == "done", res
    run = json.loads(REPO["previews/u1/run.json"])
    assert run["editor"] == "sdxl-union" and run["mask"] is None
    bad = render(w, "u2", "scenes/s/v1", CAM, "red chairs", object_query="chairs")
    assert bad["status"] == "failed"
