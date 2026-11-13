import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from splat_studio.reconstruction import colmap, cost, errors, export, frames, pipeline

# --- frames -------------------------------------------------------------------


def test_sample_fps_targets_frame_count() -> None:
    assert frames.sample_fps(90, 30) == pytest.approx(2.0)  # 180 frames from 90 s


def test_sample_fps_never_exceeds_source() -> None:
    assert frames.sample_fps(3, 24) == 24


def test_sample_fps_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        frames.sample_fps(0, 30)


def test_ffmpeg_command_has_fps_and_pattern(tmp_path: Path) -> None:
    cmd = frames.ffmpeg_command(Path("in.mp4"), tmp_path, 2.0)
    assert "fps=2.0000" in cmd[cmd.index("-vf") + 1]
    assert cmd[-1].endswith("%05d.jpg")


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
