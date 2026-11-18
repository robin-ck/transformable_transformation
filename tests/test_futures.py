from pathlib import Path

import pytest
from PIL import Image

from splat_studio import models
from splat_studio.space import futures, viewpoints
from splat_studio.space.jobs import Job, JobManager, LocalStore

CAM = {"c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]],
       "fov_deg": 50.0, "aspect": 1.5}  # fmt: skip


class FakeClient:
    def __init__(self) -> None:
        self.submitted: list[tuple] = []
        self.results: dict[str, object] = {}

    def submit_preview(self, job_id, scene_path, camera, prompt, **kw) -> str:
        self.submitted.append((job_id, scene_path, prompt, kw))
        return f"fc-{job_id}"

    def poll(self, call_id):
        r = self.results.get(call_id)
        return ("finished", r) if r is not None else ("running", None)


def _setup():
    setup = viewpoints.default_setup()
    vp = viewpoints.add_viewpoint(setup, "bridge", 2, CAM)
    return setup, vp


def test_prompt_puts_starter_first_and_has_a_limit() -> None:
    setup, _ = _setup()
    nature = setup.starter("nature-first")
    assert futures.compose_prompt(nature, "  willows  where the parking is ") == (
        f"{nature.text}, willows where the parking is"
    )
    assert futures.compose_prompt(nature, "") == nature.text  # starter only
    assert futures.compose_prompt(None, "a fish ladder") == "a fish ladder"
    with pytest.raises(futures.FutureError, match="Type what"):
        futures.compose_prompt(None, "  ")
    with pytest.raises(futures.FutureError, match="limit is 300"):
        futures.compose_prompt(nature, "x" * (300 - len(nature.text)))
    futures.compose_prompt(nature, "x" * (298 - len(nature.text)))  # exactly 300: fine


def test_future_job_records_viewpoint_starter_parent_editor(tmp_path: Path) -> None:
    store, client = LocalStore(tmp_path), FakeClient()
    mgr = JobManager(store, client, lambda: 10.0, prefix="previews")
    setup, vp = _setup()
    starter = setup.starter("flood-safe")
    prefix = viewpoints.view_prefix("s1", vp.id)
    params = futures.future_params(vp, prefix, starter, "a green dike", editor=None)
    prompt = futures.compose_prompt(starter, "a green dike")
    job = mgr.create_preview(prompt, "scenes/s1/v2", CAM, seed=4, future=params)
    assert job.summary == "a green dike"
    sent = {"seed": 4, "strength": 0.7, "view_prefix": prefix, "steps": 20}  # Fast
    assert client.submitted[-1] == (job.job_id, "scenes/s1/v2", prompt, sent)
    saved = mgr.get(job.job_id).params
    assert saved["viewpoint"] == vp.id and saved["parent"] is None
    assert saved["starter"] == {"id": "flood-safe", "label": "flood safe", "text": starter.text}
    assert saved["editor"] == "sdxl-depth" and saved["model"] == models.editor_spec(None).label
    assert saved["prompt"] == prompt and saved["own_prompt"] == "a green dike"

    # The worker returns the image: it is kept in memory, not written into job.json.
    client.results[job.call_id] = {"status": "done", "scene_path": f"previews/{job.job_id}",
                                   "image_jpeg": b"\xff\xd8jpeg"}  # fmt: skip
    done = mgr.refresh(job, save=False)
    assert done.status == "done" and mgr.images[job.job_id] == b"\xff\xd8jpeg"
    assert mgr.get(job.job_id).status == "queued"  # not saved yet
    mgr.save(done)
    assert mgr.get(job.job_id).status == "done" and "image_jpeg" not in mgr.get(job.job_id).run

    # Build on it: same viewpoint, its image as input, parent recorded.
    child_params = futures.future_params(vp, prefix, None, "add a footbridge", parent=done)
    child = mgr.create_preview("add a footbridge", "scenes/s1/v2", CAM, future=child_params,
                               input_path=f"previews/{job.job_id}/preview.png")  # fmt: skip
    assert client.submitted[-1][3]["input_path"] == f"previews/{job.job_id}/preview.png"
    assert mgr.get(child.job_id).params["parent"] == job.job_id
    other = viewpoints.add_viewpoint(setup, "east", 2, CAM)
    with pytest.raises(futures.FutureError, match="same viewpoint"):
        futures.future_params(other, prefix, None, "x", parent=done)


def test_old_preview_is_not_a_future_and_keeps_its_call(tmp_path: Path) -> None:
    client = FakeClient()
    mgr = JobManager(LocalStore(tmp_path), client, lambda: 1.0, prefix="previews")
    job = mgr.create_preview("autumn", "scenes/s1/v1", CAM)
    assert client.submitted[-1][3] == {"seed": 0, "strength": 0.7}
    assert not futures.is_future(job) and job.summary == "autumn"


def _future(job_id: str, vp: str, t: float, status: str = "done", **params) -> Job:
    job = Job(job_id, "preview", job_id, t, status=status)
    job.params = {"viewpoint": vp, "own_prompt": params.pop("own", ""), **params}
    return job


def test_gallery_lists_finished_futures_of_one_viewpoint_newest_first() -> None:
    old_preview = Job("p0", "preview", "x", 9.0, status="done")
    jobs = [_future("a", "vp1", 1.0), _future("b", "vp1", 3.0), _future("c", "vp2", 4.0),
            _future("d", "vp1", 5.0, status="failed"), _future("e", "vp1", 6.0, "running"),
            old_preview]  # fmt: skip
    assert [j.job_id for j in futures.viewpoint_futures(jobs, "vp1")] == ["b", "a"]
    assert futures.viewpoint_futures(jobs, None) == []


def test_index_reads_finished_jobs_once(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    mgr = JobManager(store, FakeClient(), prefix="previews")
    from dataclasses import asdict

    for job in (_future("a", "v", 1.0), _future("b", "v", 2.0, status="running")):
        store.write_json(f"previews/{job.job_id}/job.json", asdict(job))
    index = futures.FutureIndex(mgr)
    reads = []
    original = mgr.get
    mgr.get = lambda job_id: reads.append(job_id) or original(job_id)
    assert sorted(j.job_id for j in index.jobs()) == ["a", "b"]
    assert sorted(j.job_id for j in index.jobs()) == ["a", "b"]
    assert reads == ["a", "b", "b"]  # "a" is finished: read once


def test_caption_details_and_thumbnail(tmp_path: Path) -> None:
    starter = {"id": "housing", "label": "housing", "text": "new houses"}
    job = _future("j1", "vp1", 60.0, own="", starter=starter, parent="j0", seed=2,
                  model="sdxl@1 + cn@2")  # fmt: skip
    assert futures.caption(job) == "housing"
    long = _future("j2", "vp1", 0.0, own="x" * 60)
    assert futures.caption(long).endswith("…") and len(futures.caption(long)) == 48
    md = futures.details_markdown(job, "bridge")
    assert "view **bridge**" in md and "(starter only)" in md and "new houses" in md
    assert "`j0`" in md and "sdxl@1 + cn@2" in md and "1970-01-01" in md

    src = tmp_path / "big.png"
    Image.new("RGB", (1024, 768), "blue").save(src)
    thumb = futures.thumbnail(src, tmp_path / "t" / "a.jpg")
    with Image.open(thumb) as img:
        assert img.size == (512, 384) and img.format == "JPEG"
    assert futures.thumbnail(tmp_path / "missing.png", thumb) == thumb  # cached


def test_data_url_for_the_viewer_overlay(tmp_path: Path) -> None:
    import base64

    src = tmp_path / "big.png"
    Image.new("RGB", (2000, 1000), "green").save(src)
    jpg = futures.thumbnail(src, tmp_path / "v.jpg", futures.VIEWER_SIDE)
    url = futures.data_url(jpg)
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == jpg.read_bytes()
    with Image.open(jpg) as img:
        assert img.size == (1280, 640)


def test_speed_picks_the_editors_steps():
    assert futures.steps_for("flux-kontext", "fast") == 20
    assert futures.steps_for("flux-kontext", "quality") == 28
    assert futures.steps_for(None, None) == models.editor_spec(None).fast_steps
    with pytest.raises(futures.FutureError):
        futures.steps_for(None, "turbo")


def test_object_to_change_goes_to_kontext_and_shows_in_details(tmp_path: Path) -> None:
    store, client = LocalStore(tmp_path), FakeClient()
    mgr = JobManager(store, client, lambda: 10.0, prefix="previews")
    _, vp = _setup()
    prefix = viewpoints.view_prefix("s1", vp.id)
    params = futures.future_params(vp, prefix, None, "make the chairs red", None,
                                   "flux-kontext", object_query="  chairs ")  # fmt: skip
    assert params["object_query"] == "chairs"
    job = mgr.create_preview("make the chairs red", "scenes/s1/v2", CAM, future=params)
    kw = client.submitted[-1][3]
    assert kw["object_query"] == "chairs" and kw["class_name"] == "KontextRenderer"
    # The returned mask is kept in memory like the image, never in job.json.
    client.results[job.call_id] = {"status": "done", "scene_path": "previews/x",
                                   "image_jpeg": b"jpg", "mask_png": b"png",
                                   "mask": {"detections": 3}}  # fmt: skip
    job = mgr.refresh(job)
    assert mgr.masks[job.job_id] == b"png" and "mask_png" not in job.run
    assert "**Object to change:** chairs (3 found" in futures.details_markdown(job)

    # No object: no field, nothing sent.
    plain = futures.future_params(vp, prefix, None, "x", None, "flux-kontext", object_query=" ")
    assert "object_query" not in plain
    # SDXL cannot mask; over-long queries are refused.
    with pytest.raises(futures.FutureError, match="Only instruction editors"):
        futures.future_params(vp, prefix, None, "x", None, "sdxl-depth", object_query="chairs")
    with pytest.raises(futures.FutureError, match="limit"):
        futures.future_params(vp, prefix, None, "x", None, "flux-kontext", object_query="x" * 61)


def test_drawn_area_is_stored_and_sent(tmp_path: Path) -> None:
    import numpy as np

    from splat_studio.editing import area

    store, client = LocalStore(tmp_path), FakeClient()
    mgr = JobManager(store, client, lambda: 10.0, prefix="previews")
    _, vp = _setup()
    prefix = viewpoints.view_prefix("s1", vp.id)
    hard = np.zeros((100, 200), bool)
    hard[10:30, 20:60] = True
    params = futures.future_params(vp, prefix, None, "a willow", None, "flux-kontext",
                                   area_mask=hard)  # fmt: skip
    assert params["area"] == {"bounds": [20, 10, 60, 30], "cover": 0.04}
    assert futures.is_limited(params) and "object_query" not in params
    png = area.to_png(hard)
    job = mgr.create_preview("a willow", "scenes/s1/v2", CAM, future=params, area_png=png)
    kw = client.submitted[-1][3]
    assert kw["area_png"] == png and kw["class_name"] == "KontextRenderer"
    assert "area_png" not in job.params  # bytes never go into job.json
    assert "**Area to change:** drawn, 4% of the view" in futures.details_markdown(job)

    # An empty drawing is no area; a tiny one, SDXL, or an object as well are refused.
    plain = futures.future_params(vp, prefix, None, "x", None, "flux-kontext",
                                  area_mask=np.zeros((10, 10), bool))  # fmt: skip
    assert "area" not in plain and not futures.is_limited(plain)
    tiny = np.zeros((100, 100), bool)
    tiny[0, 0] = True
    with pytest.raises(futures.FutureError, match="too small"):
        futures.future_params(vp, prefix, None, "x", None, "flux-kontext", area_mask=tiny)
    with pytest.raises(futures.FutureError, match="drawn area"):
        futures.future_params(vp, prefix, None, "x", None, "sdxl-depth", area_mask=hard)
    with pytest.raises(futures.FutureError, match="not both"):
        futures.future_params(vp, prefix, None, "x", None, "flux-kontext",
                              object_query="chairs", area_mask=hard)  # fmt: skip


def test_delete_future_removes_its_folder(tmp_path: Path) -> None:
    from dataclasses import asdict

    store = LocalStore(tmp_path)
    mgr = JobManager(store, FakeClient(), prefix="previews")
    for job in (_future("a", "v", 1.0), _future("b", "v", 2.0, status="running")):
        store.write_json(f"previews/{job.job_id}/job.json", asdict(job))
    plain = Job("p", "preview", "p", 3.0, status="done")
    store.write_json("previews/p/job.json", asdict(plain))
    index = futures.FutureIndex(mgr)
    index.jobs()
    assert futures.delete_future(index, "a").job_id == "a"
    assert store.list_dirs("previews") == ["b", "p"] and "a" not in index.final
    for job_id, why in (("b", "still being made"), ("p", "not found"), ("a", "not found")):
        with pytest.raises(futures.FutureError, match=why):
            futures.delete_future(index, job_id)


def test_detections_are_kept_listed_and_captioned(tmp_path: Path) -> None:
    store, client = LocalStore(tmp_path), FakeClient()
    mgr = JobManager(store, client, lambda: 10.0, prefix="previews")
    _, vp = _setup()
    prefix = viewpoints.view_prefix("s1", vp.id)
    params = futures.future_params(vp, prefix, None, "paint the chairs", None, "flux-kontext",
                                   object_query="chairs, stool")  # fmt: skip
    job = mgr.create_preview("paint the chairs", "scenes/s1/v2", CAM, future=params)
    mask = {"detections": 3, "labels": ["chair", "stool", "chair"], "scores": [0.62, 0.5, 0.41]}
    client.results[job.call_id] = {"status": "done", "scene_path": "previews/x",
                                   "image_jpeg": b"jpg", "detections_jpeg": b"det",
                                   "mask": mask}  # fmt: skip
    job = mgr.refresh(job)
    assert mgr.detections[job.job_id] == b"det" and "detections_jpeg" not in job.run
    assert futures.detections_caption(job) == "Detected: 2 × chair, 1 × stool"
    details = futures.details_markdown(job)
    assert "**Detected:** 1 chair 0.62 · 2 stool 0.50 · 3 chair 0.41" in details
    plain = futures.future_params(vp, prefix, None, "x", None, "flux-kontext")
    other = mgr.create_preview("x", "scenes/s1/v2", CAM, future=plain)
    assert futures.detections_caption(other) is None
    assert "Detected" not in futures.details_markdown(other)


def test_detect_query_needs_an_object_and_a_masking_editor() -> None:
    assert futures.detect_query("flux-kontext", "  chairs,  table ") == "chairs, table"
    with pytest.raises(futures.FutureError, match="Type the object"):
        futures.detect_query("flux-kontext", " ")
    with pytest.raises(futures.FutureError, match="Only instruction editors"):
        futures.detect_query("sdxl-depth", "chairs")


def test_try_editors_skips_restyle_editors_when_limited() -> None:
    # add-futures-editors
    run, skipped = futures.try_editors(["step1x-edit", "sdxl-depth", "qwen-edit"], False)
    assert run == ["sdxl-depth", "qwen-edit", "step1x-edit"] and skipped == []
    run, skipped = futures.try_editors(
        ["sdxl-union", "flux2-klein", "sdxl-depth", "flux2-klein"], True
    )
    assert run == ["flux2-klein"] and skipped == ["SDXL depth", "SDXL union"]
    with pytest.raises(futures.FutureError, match="None of the ticked editors"):
        futures.try_editors(["sdxl-depth"], True)
    with pytest.raises(futures.FutureError, match="Tick at least one"):
        futures.try_editors([], False)
    with pytest.raises(futures.FutureError, match="Unknown editor"):
        futures.try_editors(["hidream"], False)


def test_editor_tag_and_rewritten_prompt_in_details() -> None:
    from splat_studio.space.jobs import Job

    job = Job("j1", "preview", "x", 0.0)
    job.params = {"viewpoint": "v", "editor": "qwen-edit", "own_prompt": "x"}
    assert futures.editor_tag(job) == "Qwen"
    job.params["editor"] = "gone-editor"
    assert futures.editor_tag(job) == "gone-editor"
    del job.params["editor"]
    assert futures.editor_tag(job) == "SDXL depth"  # futures from before editors existed
    job.run = {"editor_prompt_rewritten": "Replace the bench with grass."}
    assert "**Rewritten by the editor:** Replace the bench" in futures.details_markdown(job)
    assert "Kontext, Qwen, klein, Step1X" == futures.instruct_names()
    assert "not used" in futures.sent_preview(None, "in 2050", "sdxl-union")
