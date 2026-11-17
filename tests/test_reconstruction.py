import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from splat_studio.reconstruction import colmap, cost, errors, export, frames, pipeline
from splat_studio.reconstruction import settings as scene_settings

# --- frames -------------------------------------------------------------------


def test_sample_fps_targets_frame_count() -> None:
    assert frames.sample_fps(90, 30) == pytest.approx(2.0)  # 180 frames from 90 s


def test_sample_fps_never_exceeds_source() -> None:
    assert frames.sample_fps(3, 24) == 24


def test_sample_fps_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        frames.sample_fps(0, 30)


def test_candidates_command_has_fps_scale_and_score_stream(tmp_path: Path) -> None:
    cmd = frames.candidates_command(Path("in.mp4"), tmp_path, 2.0)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "fps=2.0000" in graph and "min(1600,iw)" in graph
    assert f"scale={frames.SCORE_SIZE}:{frames.SCORE_SIZE}" in graph and "format=gray" in graph
    assert str(tmp_path / "%05d.jpg") in cmd and cmd[-1] == "pipe:1"
    small = frames.candidates_command(Path("in.mp4"), tmp_path, 2.0, max_long_side=1200)
    assert "min(1200,iw)" in small[small.index("-filter_complex") + 1]


def test_sharpness_is_lower_for_blurred_image() -> None:
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, (64, 64)).astype(np.uint8)
    blurred = (sharp[:-1, :-1].astype(float) + sharp[1:, 1:] + sharp[1:, :-1] + sharp[:-1, 1:]) / 4
    assert frames.sharpness(blurred) < frames.sharpness(sharp) / 2
    assert frames.sharpness(np.full((8, 8), 7, np.uint8)) == 0


def test_pick_sharpest_takes_best_in_each_window() -> None:
    scores = [1, 5, 2, 9, 3, 4, 0, 0, 8]
    assert frames.pick_sharpest(scores, 3) == [1, 3, 8]
    assert frames.pick_sharpest(scores, 9) == list(range(9))
    assert frames.pick_sharpest(scores, 20) == list(range(9))
    assert frames.pick_sharpest([], 5) == []


def test_copy_photos_scales_large_ones_and_keeps_exif(tmp_path: Path) -> None:
    from PIL import Image

    src = tmp_path / "src"
    src.mkdir()
    exif = Image.Exif()
    exif[0x010F] = "TestCam"  # Make
    Image.new("RGB", (4000, 3000), "red").save(src / "big.JPG", exif=exif)
    Image.new("RGB", (1000, 2000), "blue").save(src / "tall.png")
    Image.new("RGB", (800, 600), "green").save(src / "small.jpg")
    photos = [src / "big.JPG", src / "tall.png", src / "small.jpg"]
    out = tmp_path / "images"
    assert frames.copy_photos(photos, out, max_long_side=1600) == 3
    with Image.open(out / "00000.jpg") as im:
        assert im.size == (1600, 1200) and im.getexif()[0x010F] == "TestCam"
    with Image.open(out / "00001.png") as im:
        assert im.size == (800, 1600)
    assert (out / "00002.jpg").read_bytes() == (src / "small.jpg").read_bytes()  # untouched


# --- scene settings (scene-settings) ----------------------------------------------


def test_scene_settings_default_and_overrides() -> None:
    assert scene_settings.check_settings(None) == {"steps": 7000, "frames": 180, "max_side": 1600}
    assert scene_settings.check_settings({"steps": 3000.0})["steps"] == 3000
    for bad, match in [
        ({"steps": 999}, "between"),
        ({"frames": 301}, "between"),
        ({"max_side": 2000}, "between"),
        ({"steps": 1500.5}, "whole number"),
        ({"steps": True}, "whole number"),
        ({"sh_degree": 3}, "unknown"),
    ]:
        with pytest.raises(ValueError, match=match):
            scene_settings.check_settings(bad)


def test_scene_presets_are_valid_and_named() -> None:
    for name, preset in scene_settings.PRESETS.items():
        assert scene_settings.check_settings(preset) == preset
        assert scene_settings.preset_of(preset) == name
        scene_settings.check_time(preset)  # every preset fits the time limit
    assert scene_settings.preset_of({"steps": 7500}) is None


def test_scene_estimate_bounds_the_measured_runs() -> None:
    # three Standard runs took 589, 629 and 866 s (the slowest about $0.31)
    standard = scene_settings.estimate_seconds(None)
    assert 866 <= standard < 950
    assert 0.28 < scene_settings.estimate_usd(standard) < 0.34
    est = {n: scene_settings.estimate_seconds(p) for n, p in scene_settings.PRESETS.items()}
    assert est["Fast"] < est["Standard"] < est["High"]


def test_scene_estimate_uses_photo_count() -> None:
    video = scene_settings.estimate_breakdown({"frames": 60})
    photos = scene_settings.estimate_breakdown({"frames": 60}, photos=300)
    assert photos["frames"] == 0 and photos["poses"] > video["poses"]


def test_scene_time_limit() -> None:
    longest = {"steps": 30000, "frames": 300, "max_side": 1600}
    with pytest.raises(ValueError, match="lower the frames"):
        scene_settings.check_time(longest)
    with pytest.raises(ValueError, match="upload fewer photos"):
        scene_settings.check_time(longest, photos=300)
    assert scene_settings.check_time({"steps": 30000, "max_side": 1200}) < scene_settings.MAX_S


def test_pipeline_refuses_bad_settings_before_work(tmp_path: Path) -> None:
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "v.mp4").touch()
    with pytest.raises(errors.ReconstructionError) as e:
        pipeline.run(inp, tmp_path / "work", tmp_path, settings={"steps": 30000, "frames": 300})
    assert e.value.code == "bad_settings" and "limit" in e.value.message
    assert not (tmp_path / "work").exists()


# --- colmap -------------------------------------------------------------------


def _model(dir_: Path, n: int) -> None:
    dir_.mkdir(parents=True)
    (dir_ / "images.bin").write_bytes(struct.pack("<Q", n))


def test_matcher_depends_on_input_kind(tmp_path: Path) -> None:
    assert colmap.commands(tmp_path, True)[1][1] == "sequential_matcher"
    assert colmap.commands(tmp_path, False)[1][1] == "exhaustive_matcher"


def test_best_model_picks_largest(tmp_path: Path) -> None:
    _model(tmp_path / "0", 10)
    _model(tmp_path / "1", 80)
    model, n = colmap.best_model(tmp_path)
    assert model == tmp_path / "1" and n == 80


def test_check_model_no_poses(tmp_path: Path) -> None:
    with pytest.raises(errors.ReconstructionError) as e:
        colmap.check_model(tmp_path / "sparse", 100)
    assert e.value.code == "pose_failed" and "Tip" in e.value.tip


def test_check_model_low_registration(tmp_path: Path) -> None:
    _model(tmp_path / "0", 30)
    with pytest.raises(errors.ReconstructionError) as e:
        colmap.check_model(tmp_path, 100)
    assert e.value.code == "pose_partial"
    assert "30 of 100" in e.value.message


def test_normalise_layout_moves_best_to_zero(tmp_path: Path) -> None:
    _model(tmp_path / "0", 10)
    _model(tmp_path / "1", 80)
    colmap.normalise_layout(tmp_path, tmp_path / "1")
    assert colmap.registered_images(tmp_path / "0") == 80


# --- export -------------------------------------------------------------------

PROPS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]  # fmt: skip


def _write_ply(path: Path, rows: np.ndarray) -> None:
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {len(rows)}\n"
    header += "".join(f"property float {p}\n" for p in PROPS) + "end_header\n"
    path.write_bytes(header.encode() + rows.astype("<f4").tobytes())


def test_ply_to_splat_roundtrip(tmp_path: Path) -> None:
    small = [0, 0, 0, 0, 0, 0, 0, -3, -3, -3, 1, 0, 0, 0]
    big = [1, 2, 3, 1, 0, -1, 5, 0, 0, 0, 2, 0, 0, 0]  # unnormalised quaternion
    _write_ply(tmp_path / "s.ply", np.array([small, big]))
    n = export.ply_to_splat(tmp_path / "s.ply", tmp_path / "s.splat")
    data = (tmp_path / "s.splat").read_bytes()
    assert n == 2 and len(data) == 64
    first = np.frombuffer(data[:24], dtype="<f4")
    assert list(first[:3]) == [1, 2, 3]  # biggest Gaussian sorted first
    assert first[3] == pytest.approx(1.0)  # exp(0)
    rgba = list(data[24:28])
    assert rgba[1] == 127 and rgba[3] > 250  # grey green channel, near-opaque
    assert list(data[28:32]) == [255, 128, 128, 128]  # normalised (1,0,0,0)


def test_read_ply_rejects_ascii(tmp_path: Path) -> None:
    (tmp_path / "a.ply").write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    with pytest.raises(ValueError):
        export.read_ply(tmp_path / "a.ply")


# --- cost ---------------------------------------------------------------------


def test_estimate_cost() -> None:
    assert cost.estimate_cost("T4", 1000) == pytest.approx(0.164)
    with pytest.raises(KeyError):
        cost.estimate_cost("TPU", 1)


def test_gpu_from_device_name() -> None:
    assert cost.gpu_from_device_name("NVIDIA L40S", "L4") == "L40S"
    assert cost.gpu_from_device_name("NVIDIA L4", "A10G") == "L4"
    assert cost.gpu_from_device_name("NVIDIA A10G", "L4") == "A10G"
    assert cost.gpu_from_device_name("NVIDIA A100-SXM4-40GB", "L4") == "A100"
    assert cost.gpu_from_device_name("Some new card", "L4") == "L4"


def test_runlog_finish_records_cost_and_error() -> None:
    log = cost.RunLog(job_id="j", gpu="A10G", started_at=0)
    log.stage("poses", 12.34)
    out = log.finish("failed", errors.no_poses().to_dict()).to_dict()
    assert out["status"] == "failed" and out["stages"] == {"poses": 12.3}
    assert out["est_cost_usd"] > 0 and out["error"]["code"] == "pose_failed"


# --- pipeline -----------------------------------------------------------------


def test_classify_inputs(tmp_path: Path) -> None:
    (tmp_path / "a.MP4").touch()
    assert pipeline.classify_inputs(tmp_path)[0] == "video"
    (tmp_path / "b.jpg").touch()
    with pytest.raises(errors.ReconstructionError):
        pipeline.classify_inputs(tmp_path)


def test_too_few_photos_fails_before_gpu_work(tmp_path: Path) -> None:
    inp = tmp_path / "in"
    inp.mkdir()
    for i in range(5):
        (inp / f"{i}.jpg").touch()
    with pytest.raises(errors.ReconstructionError) as e:
        pipeline.run(inp, tmp_path / "work", tmp_path)
    assert e.value.code == "too_few_frames"


def test_train_command_saves_ply(tmp_path: Path) -> None:
    cmd = pipeline.train_command(tmp_path, tmp_path / "d", tmp_path / "r", 500)
    assert "--save_ply" in cmd and cmd[cmd.index("--max_steps") + 1] == "500"


# --- modal client (mocked) ----------------------------------------------------


def test_modal_client_spawn_and_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    from splat_studio import modal_client

    class Call:
        object_id = "fc-123"

        def __init__(self, done: bool) -> None:
            self.done = done

        def get(self, timeout: float) -> dict:
            if not self.done:
                raise TimeoutError
            return {"status": "done"}

    fake = types.SimpleNamespace(
        Function=types.SimpleNamespace(
            from_name=lambda app, name: types.SimpleNamespace(spawn=lambda *a, **k: Call(False))
        ),
        FunctionCall=types.SimpleNamespace(from_id=lambda cid: Call(cid == "done")),
        exception=types.SimpleNamespace(TimeoutError=TimeoutError),
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    assert modal_client.submit_reconstruction("job1") == "fc-123"
    assert modal_client.poll("x") == ("running", None)
    assert modal_client.poll("done") == ("finished", {"status": "done"})


def test_modal_client_class_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    from splat_studio import modal_client

    calls = []

    class Method:
        def __init__(self, cls: str, name: str) -> None:
            self.cls, self.name = cls, name

        def spawn(self, *args, **kwargs):
            calls.append((self.cls, self.name, args, kwargs))
            return types.SimpleNamespace(object_id="fc-9")

    def from_name(app: str, cls: str):
        assert app == modal_client.APP_NAME
        return lambda: types.SimpleNamespace(
            generate=Method(cls, "generate"), render=Method(cls, "render"), edit=Method(cls, "edit")
        )

    monkeypatch.setitem(
        sys.modules, "modal", types.SimpleNamespace(Cls=types.SimpleNamespace(from_name=from_name))
    )
    assert modal_client.submit_object("j1", "oak", seed=2) == "fc-9"
    assert modal_client.submit_object("j3", "", reference_path="objects/j3/input/r.png") == "fc-9"
    assert modal_client.submit_preview("j2", "scenes/s/v1", {"fov": 50}, "warm light") == "fc-9"
    assert modal_client.submit_edit("e1", "s", 2, {"fov": 50}, "green", budget={"steps": 900})
    assert calls[-1] == (
        "SceneEditor",
        "edit",
        ("e1", "s", 2, {"fov": 50}, "green"),
        {"region": None, "seed": 0, "budget": {"steps": 900}},
    )
    calls.pop()
    assert modal_client.submit_image("j4", "roof", source_path="images/j4/input/s.png") == "fc-9"
    assert calls == [
        ("ObjectGenerator", "generate", ("j1", "oak"), {"seed": 2}),
        (
            "ObjectGenerator",
            "generate",
            ("j3", ""),
            {"seed": 0, "reference_path": "objects/j3/input/r.png"},
        ),
        (
            "PreviewRenderer",
            "render",
            ("j2", "scenes/s/v1", {"fov": 50}, "warm light"),
            {"seed": 0, "strength": 0.7},
        ),
        (
            "ImageStudio",
            "generate",
            ("j4", "roof"),
            {"seed": 0, "source_path": "images/j4/input/s.png", "strength": 0.7, "aspect": 1.0},
        ),
    ]


def test_load_times_reported_once() -> None:
    from splat_studio.reconstruction.cost import LoadTimes, RunLog

    loads = LoadTimes()
    loads.record("load_image_model", 20.44)
    first = RunLog(job_id="a", gpu="L40S")
    first.stage("x", 1.0)
    loads.apply(first)
    assert first.warm is False
    assert list(first.stages) == ["load_image_model", "x"]
    assert first.stages["load_image_model"] == 20.4

    second = RunLog(job_id="b", gpu="L40S")
    loads.apply(second)
    assert second.warm is True and second.stages == {}
    assert RunLog(job_id="c", gpu="L4").to_dict()["warm"] is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_extract_frames_real_video(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=2000x1000:rate=30",
         "-t", "4", str(video)],
        check=True,
    )  # fmt: skip
    n = frames.extract_frames(video, tmp_path / "img", target_frames=20)
    assert 18 <= n <= 22
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(next((tmp_path / "img").glob("*.jpg")))],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    assert out == "1600,800"
    assert not [p for p in tmp_path.iterdir() if p.name not in {"v.mp4", "img"}]  # no leftovers


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_extract_frames_skips_blurred_frames(tmp_path: Path) -> None:
    from PIL import Image

    # 30 fps, only frames 0 and 1 of every 6 sharp; 20 frames from 4 s means 15 fps
    # candidates in groups of 3 (6 source frames), each group holding one sharp frame.
    video = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30",
         "-vf", "boxblur=8:enable='gte(mod(n,6),2)'", "-t", "4", str(video)],
        check=True,
    )  # fmt: skip
    n = frames.extract_frames(video, tmp_path / "img", target_frames=20)
    assert 18 <= n <= 22
    ref = tmp_path / "sharp.jpg"  # frame 0 is sharp
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(ref)],
        check=True,
    )  # fmt: skip

    def score(f: Path) -> float:
        with Image.open(f) as im:
            return frames.sharpness(np.asarray(im.convert("L")))

    sharp = score(ref)
    assert all(score(f) > sharp / 2 for f in (tmp_path / "img").glob("*.jpg"))


# --- colmap runner (fake subprocess) -------------------------------------------


def _fake_colmap(monkeypatch: pytest.MonkeyPatch, mapper_rc: int, mapper_err: str, n_reg: int):
    def fake_run(cmd, capture_output, text, env):
        assert env["QT_QPA_PLATFORM"] == "offscreen"
        if cmd[1] == "mapper":
            out = Path(cmd[cmd.index("--output_path") + 1])
            assert out.is_dir(), "mapper needs an existing output dir"
            if n_reg:
                _model(out / "0", n_reg)
            return subprocess.CompletedProcess(cmd, mapper_rc, "", mapper_err)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(colmap.subprocess, "run", fake_run)


def _images(tmp_path: Path, n: int) -> Path:
    (tmp_path / "images").mkdir()
    for i in range(n):
        (tmp_path / "images" / f"{i:05d}.jpg").write_bytes(b"x")
    return tmp_path


def test_colmap_run_creates_sparse_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_colmap(monkeypatch, 0, "", 25)
    assert colmap.run(_images(tmp_path, 30), sequential=True) == tmp_path / "sparse" / "0"


def test_colmap_crash_is_not_blamed_on_footage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_colmap(monkeypatch, 1, "ERROR: `output_path` is not a directory", 0)
    with pytest.raises(errors.ReconstructionError) as e:
        colmap.run(_images(tmp_path, 30), sequential=True)
    assert e.value.code == "pose_tool_failed"
    assert "output_path" in e.value.to_dict()["detail"]


def test_colmap_bad_footage_keeps_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_colmap(monkeypatch, 1, "No good initial image pair found.", 0)
    with pytest.raises(errors.ReconstructionError) as e:
        colmap.run(_images(tmp_path, 30), sequential=True)
    assert e.value.code == "pose_failed" and "initial image pair" in e.value.detail
