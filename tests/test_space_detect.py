"""The Space's "Detect objects" handler (add-detection-preview) with Modal faked.
Needs gradio to import space/app.py; skipped without it."""

from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")
ROOT = Path(__file__).resolve().parent.parent


_LOADED: list = []


def _space_app():
    """space/app.py under its own module name: modal_app/app.py is also imported as `app`."""
    if not _LOADED:
        import importlib.util

        spec = importlib.util.spec_from_file_location("space_app", ROOT / "space" / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOADED.append(module)
    return _LOADED[0]


@pytest.fixture
def app(monkeypatch, tmp_path):
    space_app = _space_app()
    from splat_studio.space import viewpoints

    setup = viewpoints.default_setup()
    vp = viewpoints.add_viewpoint(setup, "bridge", 1, {"c2w": [[1, 0, 0, 0]] * 4})
    setup.editor = "flux-kontext"
    monkeypatch.setattr(space_app, "_setup", lambda scene_id: setup)
    monkeypatch.setattr(space_app, "_local", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(space_app.time, "sleep", lambda s: None)
    return space_app, setup, vp


class FakeClient:
    def __init__(self, result, polls=2):
        self.result, self.polls, self.sent = result, polls, []

    def submit_detect(self, view_prefix, query, class_name):
        self.sent.append((view_prefix, query, class_name))
        return "fc-1"

    def poll(self, call_id):
        self.polls -= 1
        return ("finished", self.result) if self.polls <= 0 else ("running", None)


def test_detect_objects_shows_the_labelled_render(app, monkeypatch):
    space_app, setup, vp = app
    client = FakeClient({"status": "done", "detections_jpeg": b"\xff\xd8jpg",
                         "mask": {"labels": ["chair", "chair", "stool"]}})  # fmt: skip
    monkeypatch.setattr(space_app.PREVIEWS, "client", client)
    steps = list(space_app.on_detect_objects("s1", vp.id, " chairs, stool "))
    assert "Looking for **chairs, stool**" in steps[0][0]
    status, image = steps[-1]
    assert client.sent == [(f"viewpoints/s1/{vp.id}", "chairs, stool", "KontextRenderer")]
    assert "Detected: 2 × chair, 1 × stool" in status
    assert image["visible"] and image["label"] == "Detected: 2 × chair, 1 × stool"
    assert Path(image["value"]).read_bytes() == b"\xff\xd8jpg"


def test_detect_objects_reports_failures_and_refuses_bad_input(app, monkeypatch):
    space_app, setup, vp = app
    err = {"code": "object_not_found", "message": 'Nothing matching "boat" was found.',
           "tip": "Tip: try another word."}  # fmt: skip
    monkeypatch.setattr(space_app.PREVIEWS, "client", FakeClient({"status": "failed",
                                                                  "error": err}))  # fmt: skip
    status, image = list(space_app.on_detect_objects("s1", vp.id, "boat"))[-1]
    assert status.startswith("❌ Nothing matching") and image["visible"] is False
    with pytest.raises(gr.Error):
        list(space_app.on_detect_objects("s1", "nope", "chairs"))  # no saved view
    with pytest.raises(gr.Error):
        list(space_app.on_detect_objects("s1", vp.id, ""))  # nothing named
    setup.editor = "sdxl-depth"
    with pytest.raises(gr.Error):
        list(space_app.on_detect_objects("s1", vp.id, "chairs"))
