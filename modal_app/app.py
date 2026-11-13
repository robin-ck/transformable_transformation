"""Modal GPU functions for Splat Studio.

Deploy:  modal deploy modal_app/app.py
Spike:   modal run modal_app/app.py::gpu_probe
Needs a Modal secret named `huggingface-secret` with HF_TOKEN, and the env var
SPLAT_STUDIO_DATASET (e.g. "your-name/splat-studio-data") set on that secret.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = "splat-studio"
GSPLAT_VERSION = "1.5.3"  # pin; check https://github.com/nerfstudio-project/gsplat/releases
GSPLAT_DIR = "/opt/gsplat"
TRAIN_GPU = "A10G"
TRAIN_TIMEOUT_S = 60 * 60  # hard cap per job to protect the monthly credit
CPU_CORES = 4.0

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "colmap", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", "numpy<2", "huggingface_hub>=0.25")
    .pip_install(
        f"gsplat=={GSPLAT_VERSION}",
        extra_index_url="https://docs.gsplat.studio/whl/pt24cu124",
    )
    # fused-ssim's setup.py imports torch, so it must build without isolation
    # (isolated build envs can't see the torch installed above). The build
    # container has no GPU, so the CUDA arch is set explicitly: 8.6 = A10G.
    # Modal's add_python interpreter was built with clang, so extensions would
    # look for clang++; point them at the image's gcc instead.
    .apt_install("build-essential")
    .pip_install("setuptools", "wheel")
    .env({"TORCH_CUDA_ARCH_LIST": "8.6", "CC": "gcc", "CXX": "g++"})
    .run_commands(
        f"git clone --depth 1 --branch v{GSPLAT_VERSION} "
        f"https://github.com/nerfstudio-project/gsplat {GSPLAT_DIR}",
        f"pip install --no-build-isolation -r {GSPLAT_DIR}/examples/requirements.txt",
    )
    .add_local_python_source("splat_studio")
)

app = modal.App(APP_NAME, image=image)
hf_secret = modal.Secret.from_name("huggingface-secret")


def _upload_json(api, repo: str, path: str, data: dict) -> None:
    api.upload_file(
        path_or_fileobj=json.dumps(data, indent=2).encode(),
        path_in_repo=path,
        repo_id=repo,
        repo_type="dataset",
    )


@app.function(gpu="T4", timeout=300, secrets=[hf_secret])
def gpu_probe(job_id: str = "spike") -> dict:
    """Task 2.1 spike: prove the Space can start a GPU job and log runtime + cost."""
    from splat_studio.reconstruction.cost import RunLog

    log = RunLog(job_id=job_id, gpu="T4")
    t = time.time()
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    log.stage("nvidia_smi", time.time() - t)
    out = {**log.finish("done").to_dict(), "nvidia_smi": smi}
    print(json.dumps(out))  # `modal run` discards return values; print to see it
    return out


@app.function(gpu=TRAIN_GPU, cpu=CPU_CORES, timeout=TRAIN_TIMEOUT_S, secrets=[hf_secret], retries=0)
def reconstruct(job_id: str, steps: int = 7000) -> dict:
    """Download inputs from the HF dataset, reconstruct, upload scene + run log."""
    from huggingface_hub import HfApi, snapshot_download

    from splat_studio.reconstruction import pipeline
    from splat_studio.reconstruction.cost import RunLog
    from splat_studio.reconstruction.errors import ReconstructionError, training_failed

    repo = os.environ["SPLAT_STUDIO_DATASET"]
    api = HfApi()
    log = RunLog(job_id=job_id, gpu=TRAIN_GPU, cpu_cores=CPU_CORES)
    _upload_json(api, repo, f"jobs/{job_id}/run.json", log.to_dict())

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                allow_patterns=[f"jobs/{job_id}/input/*"],
                local_dir=tmp_path / "dl",
            )
            result = pipeline.run(
                tmp_path / "dl" / "jobs" / job_id / "input",
                tmp_path / "work",
                Path(GSPLAT_DIR),
                steps=steps,
                on_stage=log.stage,
            )
            for local, name in [(result.ply, "scene.ply"), (result.splat, "scene.splat")]:
                api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=f"scenes/{job_id}/v1/{name}",
                    repo_id=repo,
                    repo_type="dataset",
                )
            log.finish("done")
            out = {
                **log.to_dict(),
                "scene_path": f"scenes/{job_id}/v1",
                "num_gaussians": result.num_gaussians,
                "num_frames": result.num_frames,
            }
        except ReconstructionError as e:
            out = log.finish("failed", e.to_dict()).to_dict()
        except Exception as e:  # unexpected: still report something readable
            out = log.finish("failed", training_failed(repr(e)).to_dict()).to_dict()

    _upload_json(api, repo, f"jobs/{job_id}/run.json", out)
    print(json.dumps(out))
    return out
