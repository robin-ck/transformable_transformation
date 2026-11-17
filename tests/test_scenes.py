"""Scene details and deletion (manage-scenes)."""

from dataclasses import asdict
from pathlib import Path

import pytest

from splat_studio.space import jobs, scenes
from splat_studio.space.jobs import Job, JobManager, LocalStore

SCENE, OTHER = "20261115-184011-7e524c", "20261116-090000-aaaaaa"


def _job(job_id: str, kind: str = "video", status: str = "done", **params) -> Job:
    return Job(job_id, kind, f"{kind} {job_id}", 1_000.0, status=status, params=params)


def _put(store: LocalStore, prefix: str, job: Job, files: tuple[str, ...] = ()) -> None:
    store.write_json(f"{prefix}/{job.job_id}/job.json", asdict(job))
    for name in files:
        target = store.root / prefix / job.job_id / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")


@pytest.fixture
def store(tmp_path: Path) -> LocalStore:
    """Two scenes with edits, previews and objects, plus an image job."""
    store = LocalStore(tmp_path / "store")
    for sid in (SCENE, OTHER):
        _put(store, "jobs", _job(sid), ("input/000_a.jpg",))
        for n in (1, 2):
            (store.root / "scenes" / sid / f"v{n}").mkdir(parents=True)
            (store.root / "scenes" / sid / f"v{n}" / "scene.splat").write_bytes(b"x")
    _put(store, "edits", _job("e1", "edit", scene_id=SCENE, version=1), ("before.png",))
    _put(store, "edits", _job("e2", "edit", scene_id=OTHER, version=1))
    _put(store, "previews", _job("p1", "preview", scene_path=f"scenes/{SCENE}/v2"))
    _put(store, "previews", _job("p2", "preview", scene_path=f"scenes/{OTHER}/v1"))
    _put(store, "objects", _job("o1", "object", scene_id=SCENE), ("object.splat",))
    _put(store, "images", _job("i1", "image"), ("result.png",))
    return store


def _plan(store: LocalStore, scene_id: str = SCENE) -> scenes.DeletePlan:
    def listed(prefix: str) -> list[Job]:
        return JobManager(store, prefix=prefix).list(refresh=False)

    return scenes.delete_plan(scene_id, 2, listed("objects"), listed("previews"), listed("edits"))


# --- name and notes ----------------------------------------------------------------


def test_clean_name_and_notes() -> None:
    assert scenes.clean_name("  Main   street\n before ") == "Main street before"
    assert scenes.clean_name("   ") is None and scenes.clean_name(None) is None
    with pytest.raises(scenes.SceneError, match="limit is 80"):
        scenes.clean_name("x" * 81)
    assert scenes.clean_notes("  two\nlines  ") == "two\nlines"
    with pytest.raises(scenes.SceneError, match="limit is 2000"):
        scenes.clean_notes("x" * 2001)


def test_labels_use_the_name_and_keep_the_id() -> None:
    job = _job(SCENE)
    assert scenes.choice_label(job) == f"{SCENE} · video {SCENE}"
    assert scenes.scene_title(job) == f"video {SCENE}"
    job.name = "Main street"
    assert scenes.choice_label(job) == f"Main street · {SCENE}"
    assert scenes.scene_title(job) == "Main street"


def test_set_details_saves_in_job_json(store: LocalStore) -> None:
    mgr = JobManager(store)
    job = scenes.set_details(mgr, SCENE, " Main  street ", "filmed at noon\n")
    assert (job.name, job.notes) == ("Main street", "filmed at noon")
    again = mgr.get(SCENE)
    assert again == job and again.summary == f"video {SCENE}"  # other fields kept
    assert scenes.set_details(mgr, SCENE, "", "").name is None  # cleared


def test_set_details_needs_a_finished_scene(store: LocalStore) -> None:
    mgr = JobManager(store)
    _put(store, "jobs", _job("20261117-000000-bbbbbb", status="running"))
    for sid in ("20261117-000000-bbbbbb", "missing", None):
        with pytest.raises(scenes.SceneError, match="finished scene"):
            scenes.set_details(mgr, sid, "x", "")


def test_old_job_json_has_no_name() -> None:
    old = {"job_id": SCENE, "kind": "video", "summary": "v.mp4", "created_at": 1.0}
    job = Job.from_dict(old)
    assert job.name is None and job.notes == ""


# --- delete ------------------------------------------------------------------------


def test_plan_lists_only_this_scenes_folders(store: LocalStore) -> None:
    plan = _plan(store)
    assert sorted(plan.folders) == sorted(
        [f"jobs/{SCENE}", f"scenes/{SCENE}", "edits/e1", "previews/p1", "objects/o1"]
    )
    assert plan.counts == {"versions": 2, "edits": 1, "previews": 1, "objects": 1}
    assert plan.summary() == "2 versions, 1 edit, 1 preview and 1 object"
    assert plan.running == []


def test_plan_summary_without_jobs() -> None:
    plan = scenes.delete_plan(SCENE, 1, [], [], [])
    assert plan.summary() == "1 version"
    assert plan.folders == [f"jobs/{SCENE}", f"scenes/{SCENE}"]


def test_delete_removes_the_scene_and_keeps_the_rest(store: LocalStore) -> None:
    file_dir = Path(store.file_dir)
    for name in (f"{SCENE}-v1.ply", f"{SCENE}-v2.splat", f"{OTHER}-v1.ply"):
        (file_dir / "downloads").mkdir(exist_ok=True)
        (file_dir / "downloads" / name).write_bytes(b"x")
    (file_dir / "hidden").mkdir()
    (file_dir / "hidden" / f"{SCENE}-v2-o1.splat").write_bytes(b"x")

    scenes.delete_scene(store, _plan(store))

    root = store.root
    for gone in (f"jobs/{SCENE}", f"scenes/{SCENE}", "edits/e1", "previews/p1", "objects/o1"):
        assert not (root / gone).exists(), gone
    for kept in (f"jobs/{OTHER}", f"scenes/{OTHER}/v2", "edits/e2", "previews/p2", "images/i1"):
        assert (root / kept).exists(), kept
    assert sorted(p.name for p in (file_dir / "downloads").iterdir()) == [f"{OTHER}-v1.ply"]
    assert list((file_dir / "hidden").iterdir()) == []
    assert [j.job_id for j in JobManager(store).list(refresh=False)] == [OTHER]


def test_delete_refused_while_a_job_of_the_scene_runs(store: LocalStore) -> None:
    _put(store, "edits", _job("e3", "edit", status="running", scene_id=SCENE))
    plan = _plan(store)
    assert [j.job_id for j in plan.running] == ["e3"]
    with pytest.raises(scenes.SceneError, match='running jobs finish .edit "edit e3"'):
        scenes.delete_scene(store, plan)
    assert (store.root / "scenes" / SCENE / "v1").is_dir()  # nothing deleted


def test_a_running_job_of_another_scene_does_not_block(store: LocalStore) -> None:
    _put(store, "previews", _job("p3", "preview", "queued", scene_path=f"scenes/{OTHER}/v2"))
    assert _plan(store).running == []


@pytest.mark.parametrize(
    "folder", ["scenes", "scenes/", "/scenes", "scenes/..", "../x", "a/b/c", "scenes/a b", ""]
)
def test_only_job_folders_can_be_deleted(store: LocalStore, folder: str) -> None:
    with pytest.raises(ValueError, match="refusing"):
        store.delete_folders([f"jobs/{SCENE}", folder])
    assert (store.root / "jobs" / SCENE).is_dir()  # checked before anything is removed


def test_hub_store_deletes_folders_in_one_commit(tmp_path: Path) -> None:
    pytest.importorskip("huggingface_hub")

    class FakeApi:
        def __init__(self) -> None:
            self.commits = []

        def create_commit(self, repo_id, ops, commit_message, repo_type):
            self.commits.append((repo_id, ops, commit_message, repo_type))

    hub = jobs.HubStore.__new__(jobs.HubStore)
    hub.repo_id, hub.api, hub.file_dir = "me/data", FakeApi(), tmp_path / "cache"
    (hub.file_dir / "scenes" / SCENE / "v1").mkdir(parents=True)
    (hub.file_dir / "scenes" / OTHER).mkdir(parents=True)

    hub.delete_folders([f"jobs/{SCENE}", f"scenes/{SCENE}"], "Delete scene")

    [(repo, ops, message, kind)] = hub.api.commits
    assert (repo, message, kind) == ("me/data", "Delete scene", "dataset")
    assert [(op.path_in_repo, op.is_folder) for op in ops] == [
        (f"jobs/{SCENE}/", True),
        (f"scenes/{SCENE}/", True),
    ]
    assert not (hub.file_dir / "scenes" / SCENE).exists()
    assert (hub.file_dir / "scenes" / OTHER).is_dir()


def test_details_markdown() -> None:
    assert "Pick a finished scene" in scenes.details_markdown(None, 0)
    job = _job(SCENE)
    job.finished_at = 3_600.0
    text = scenes.details_markdown(job, 3, "Standard: 7,000 steps")
    assert f"`{SCENE}`" in text and "1970-01-01 01:00 UTC" in text
    assert "**Versions:** 3" in text and "Standard: 7,000 steps" in text


def test_delete_includes_viewpoints_and_futures(store: LocalStore) -> None:
    from splat_studio.space import viewpoints

    setup = viewpoints.load_setup(store, SCENE)
    vp = viewpoints.add_viewpoint(setup, "bridge", 2, {"c2w": [[1, 0, 0, 0]] * 4})
    viewpoints.save_setup(store, SCENE, setup)
    (store.root / viewpoints.view_prefix(SCENE, vp.id)).mkdir(parents=True)
    (store.root / viewpoints.view_prefix(SCENE, vp.id) / "render.png").write_bytes(b"x")
    future = _job("f1", "preview", scene_path=f"scenes/{SCENE}/v2", viewpoint=vp.id)
    _put(store, "previews", future, ("preview.png",))

    def listed(prefix: str) -> list[Job]:
        return JobManager(store, prefix=prefix).list(refresh=False)

    plan = scenes.delete_plan(
        SCENE, 2, listed("objects"), listed("previews"), listed("edits"), viewpoints=True
    )
    assert f"viewpoints/{SCENE}" in plan.folders and "previews/f1" in plan.folders
    assert plan.counts["previews"] == 2
    scenes.delete_scene(store, plan)
    assert not (store.root / "viewpoints" / SCENE).exists()
    assert not (store.root / "previews/f1").exists()
    assert (store.root / "previews/p2").exists()
    # Without a viewpoints folder nothing is added (the Hub refuses missing folders).
    assert not any(f.startswith("viewpoints/") for f in _plan(store, OTHER).folders)
