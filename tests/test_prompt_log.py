from splat_studio.space import prompt_log
from splat_studio.space.jobs import Job
from splat_studio.space.versions import Version

DAY = 86_400.0


def _versions() -> list[Version]:
    return [
        Version("s1", 3, 3 * DAY, "edit_scene", "make the facade green", "ip2p@abc", parent=2),
        Version("s1", 2, 2 * DAY, "add_object", "a mature oak tree", "flux+trellis", parent=1),
        Version("s1", 1, 1 * DAY),
    ]


def _preview(job_id: str, scene_path: str, t: float, status: str = "done") -> Job:
    job = Job(job_id, "preview", "autumn", t, status=status)
    job.params = {"prompt": "same street in autumn", "scene_path": scene_path}
    job.run = {"model": "sdxl+depth"}
    return job


def test_scene_log_lists_versions_and_previews_newest_first() -> None:
    previews = [
        _preview("p1", "scenes/s1/v2", 2.5 * DAY),
        _preview("p2", "scenes/other/v1", 4 * DAY),  # another scene: left out
        _preview("p3", "scenes/s1/v3", 5 * DAY, status="failed"),
    ]
    rows = [e.row() for e in prompt_log.scene_log("s1", _versions(), previews)]
    assert [r[4] for r in rows] == [
        "preview of v3 (job p3) [failed]",
        "v3",
        "preview of v2 (job p1)",
        "v2",
        "v1",
    ]
    assert rows[1][1:4] == ["edit scene", "make the facade green", "ip2p@abc"]
    assert rows[2][1:4] == ["preview (2D)", "same street in autumn", "sdxl+depth"]
    assert rows[4][1] == "reconstruction" and rows[4][2] == ""
    assert len(rows[0]) == len(prompt_log.COLUMNS)


def test_history_of_version_3_shows_prompt_and_model_of_each_step() -> None:
    md = prompt_log.history_markdown(_versions(), 3)
    assert md.startswith("**History of v3**")
    lines = md.splitlines()[2:]
    assert [ln.split(" · ")[0] for ln in lines] == ["- v1", "- v2", "- v3"]
    assert '"make the facade green"' in lines[2] and "`ip2p@abc`" in lines[2]
    assert '"a mature oak tree"' in lines[1]


def test_lineage_skips_unrelated_branches_and_survives_cycles() -> None:
    vs = _versions() + [Version("s1", 4, 4 * DAY, "add_object", "lamp", parent=1)]
    assert [v.number for v in prompt_log.lineage(vs, 4)] == [1, 4]
    looped = [Version("s1", 1, parent=2), Version("s1", 2, parent=1)]
    assert [v.number for v in prompt_log.lineage(looped, 2)] == [1, 2]
    assert prompt_log.history_markdown(vs, None) == ""
    assert prompt_log.history_markdown(vs, 9) == ""
