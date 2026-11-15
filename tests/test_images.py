from pathlib import Path

import pytest

from splat_studio.editing import images
from splat_studio.space import jobs
from splat_studio.space.validation import UploadError, validate_source_image


@pytest.mark.parametrize("aspect", [1.0, 1.5, 16 / 9, 2 / 3, 9 / 16, 10.0, 0.01])
def test_output_size_is_flux_friendly(aspect: float) -> None:
    w, h = images.output_size(aspect)
    assert w % 16 == 0 and h % 16 == 0
    assert 0.8 < w * h / images.TARGET_PIXELS < 1.2
    assert w / h == pytest.approx(min(max(aspect, 0.4), 2.5), rel=0.05)


def test_img2img_keeps_about_four_real_steps() -> None:
    assert images.img2img_steps(1.0) == 4
    assert images.img2img_steps(0.5) == 8
    assert images.img2img_steps(0.2) == images.MAX_STEPS
    assert images.aspect_of("nonsense") == images.ASPECTS[images.DEFAULT_ASPECT]


def test_validate_source_image(tmp_path: Path) -> None:
    ok = tmp_path / "a.webp"
    ok.write_bytes(b"x")
    assert validate_source_image(str(ok)) == ok
    with pytest.raises(UploadError, match="No image"):
        validate_source_image(None)
    bad = tmp_path / "a.gif"
    bad.write_bytes(b"x")
    with pytest.raises(UploadError, match="Unsupported"):
        validate_source_image(bad)


class FakeClient:
    def __init__(self) -> None:
        self.submitted: list[tuple] = []
        self.results: dict[str, object] = {}

    def submit_image(self, job_id, prompt, seed=0, source_path=None, strength=0.7, aspect=1.0):
        self.submitted.append((job_id, prompt, seed, source_path, strength, aspect))
        return f"fc-{job_id}"

    def poll(self, call_id: str):
        r = self.results.get(call_id)
        if isinstance(r, Exception):
            raise r
        return ("finished", r) if r is not None else ("running", None)


def test_image_jobs_text_and_image_modes(tmp_path: Path) -> None:
    store, client = jobs.LocalStore(tmp_path / "s"), FakeClient()
    mgr = jobs.JobManager(store, client, lambda: 1.0, prefix="images", timeout_s=600)

    text = mgr.create_image("a  green roof", seed=2, aspect=1.5)
    assert client.submitted[-1] == (text.job_id, "a  green roof", 2, None, 0.7, 1.5)
    assert text.kind == "image" and text.summary == "a green roof"
    assert text.params["mode"] == "text" and text.params["aspect"] == 1.5

    src = tmp_path / "photo.JPG"
    src.write_bytes(b"jpg")
    edit = mgr.create_image("add wildflowers", source=src, strength=0.4)
    path = f"images/{edit.job_id}/input/source.jpg"
    assert client.submitted[-1][3:5] == (path, 0.4)
    assert (store.root / path).read_bytes() == b"jpg"
    assert edit.params["mode"] == "image" and "aspect" not in edit.params

    client.results[text.call_id] = {"status": "done", "scene_path": f"images/{text.job_id}"}
    assert mgr.refresh(text).status == "done"
    client.results[edit.call_id] = RuntimeError("worker lost")
    assert mgr.refresh(edit).error["code"] == "image_failed"
    assert not (store.root / "scenes").exists()
