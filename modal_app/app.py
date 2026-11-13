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

# Add object (task 4.2): FLUX.1-schnell makes a reference image, TRELLIS turns it into
# Gaussians. FLUX's transformer alone is 24 GB in bf16, so this runs on an L40S (48 GB).
# Only TRELLIS's Gaussian path is used; its research-only renderers are not installed.
OBJECT_GPU = "L40S"
OBJECT_TIMEOUT_S = 15 * 60  # keep in sync with splat_studio/space/jobs.py
TRELLIS_COMMIT = "193e0502b5cb3eb5997ec457d5340e95d09b158e"
TRELLIS_DIR = "/opt/trellis"
MODELS_DIR = "/models"  # Modal Volume: weights survive between runs

object_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "xformers==0.0.27.post2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy<2",
        "huggingface_hub>=0.25,<1",
        "diffusers==0.31.0",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "sentencepiece",
        "protobuf",
        "spconv-cu120==2.3.6",
        "pillow",
        "imageio",
        "easydict",
        "opencv-python-headless",
        "scipy",
        "plyfile",
        "trimesh",
        "open3d==0.18.0",  # imported by trellis.pipelines at package load
        "rembg==2.0.60",
        "onnxruntime==1.20.1",
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
    )
    # kaolin (Apache-2.0) is imported by TRELLIS's FlexiCubes mesh code at package load.
    .pip_install(
        "kaolin==0.17.0",
        find_links="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html",
    )
    .run_commands(
        f"git clone https://github.com/microsoft/TRELLIS {TRELLIS_DIR}",
        f"cd {TRELLIS_DIR} && git checkout {TRELLIS_COMMIT} && git submodule update --init",
    )
    .env(
        {
            "PYTHONPATH": TRELLIS_DIR,
            "ATTN_BACKEND": "xformers",
            "SPARSE_BACKEND": "spconv",
            "SPCONV_ALGO": "native",
            "HF_HOME": f"{MODELS_DIR}/hf",
            "TORCH_HOME": f"{MODELS_DIR}/torch",
            "U2NET_HOME": f"{MODELS_DIR}/u2net",
        }
    )
    .add_local_python_source("splat_studio")
)

app = modal.App(APP_NAME, image=image)
hf_secret = modal.Secret.from_name("huggingface-secret")
models_volume = modal.Volume.from_name("splat-studio-models", create_if_missing=True)


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


@app.function(image=object_image, timeout=300, secrets=[hf_secret])
def check_model_access() -> dict:
    """Cheap CPU check that the HF token can read every model we use (gated ones too)."""
    from huggingface_hub import HfApi

    from splat_studio import models

    api, out = HfApi(), {}
    try:  # the image must import TRELLIS and FLUX; cheaper to learn that here than on a GPU
        import diffusers  # noqa: F401
        import trellis.pipelines  # noqa: F401

        out["imports"] = "ok"
    except Exception as e:
        out["imports"] = f"{type(e).__name__}: {str(e)[:300]}"
    for spec in models.ALL:
        try:
            api.auth_check(spec.repo_id)  # raises GatedRepoError without accepted terms
            api.model_info(spec.repo_id, revision=spec.revision)  # pinned commit exists
            out[spec.repo_id] = "ok"
        except Exception as e:
            out[spec.repo_id] = f"{type(e).__name__}: {str(e)[:200]}"
    print(json.dumps(out, indent=2))
    return out


OBJECT_PROMPT = (
    "{prompt}. A single object, centred, the whole object in view, "
    "plain white background, soft even studio lighting, high detail."
)


@app.function(
    image=object_image,
    gpu=OBJECT_GPU,
    memory=48 * 1024,
    timeout=OBJECT_TIMEOUT_S,
    secrets=[hf_secret],
    volumes={MODELS_DIR: models_volume},
    retries=0,
)
def generate_object(
    job_id: str, prompt: str, seed: int = 0, reference_image: str | None = None
) -> dict:
    """Text -> reference image -> 3D Gaussians; uploads the object to objects/<job_id>/.

    `reference_image` (a path inside the container) skips the text-to-image step; used
    to test the 3D step alone, e.g. with an image from TRELLIS's own examples:
      modal run modal_app/app.py::generate_object --job-id test --prompt mailbox
        --reference-image /opt/trellis/assets/example_image/typical_misc_mailbox.png
    """
    import torch
    from huggingface_hub import HfApi, snapshot_download

    from splat_studio import models
    from splat_studio.editing.objects import normalise_object
    from splat_studio.reconstruction import errors
    from splat_studio.reconstruction.cost import RunLog
    from splat_studio.reconstruction.export import read_ply, to_splat_bytes, write_ply

    repo = os.environ["SPLAT_STUDIO_DATASET"]
    api = HfApi()
    prefix = f"objects/{job_id}"
    log = RunLog(job_id=job_id, gpu=OBJECT_GPU)
    _upload_json(api, repo, f"{prefix}/run.json", log.to_dict())
    model_label = f"{models.TEXT_TO_IMAGE.label} + {models.IMAGE_TO_3D.label}"
    if reference_image:
        model_label = models.IMAGE_TO_3D.label

    try:
        from trellis.pipelines import TrellisImageTo3DPipeline  # fail fast on a broken image

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            if reference_image:
                from PIL import Image

                image = Image.open(reference_image)
                image.save(tmp_path / "reference.png")
            else:
                t = time.time()
                from diffusers import FluxPipeline

                try:
                    flux = FluxPipeline.from_pretrained(
                        models.TEXT_TO_IMAGE.repo_id,
                        revision=models.TEXT_TO_IMAGE.revision,
                        torch_dtype=torch.bfloat16,
                    ).to("cuda")
                except Exception as e:
                    if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
                        raise errors.model_access(models.TEXT_TO_IMAGE.repo_id, repr(e)) from e
                    raise
                log.stage("load_image_model", time.time() - t)

                t = time.time()
                image = flux(
                    OBJECT_PROMPT.format(prompt=prompt.strip().rstrip(".")),
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=256,
                    height=1024,
                    width=1024,
                    generator=torch.Generator("cpu").manual_seed(seed),
                ).images[0]
                del flux
                torch.cuda.empty_cache()
                image.save(tmp_path / "reference.png")
                log.stage("image", time.time() - t)

            t = time.time()
            trellis_dir = snapshot_download(
                models.IMAGE_TO_3D.repo_id, revision=models.IMAGE_TO_3D.revision
            )
            trellis = TrellisImageTo3DPipeline.from_pretrained(trellis_dir)
            trellis.cuda()
            models_volume.commit()  # keep downloaded weights for the next run
            log.stage("load_3d_model", time.time() - t)

            t = time.time()
            out = trellis.run(image, seed=seed, formats=["gaussian"])
            raw = tmp_path / "raw.ply"
            out["gaussian"][0].save_ply(str(raw))  # default transform: TRELLIS y-up -> z-up
            obj = normalise_object(read_ply(raw))
            write_ply(tmp_path / "object.ply", obj)
            (tmp_path / "object.splat").write_bytes(to_splat_bytes(obj))
            log.stage("gaussians", time.time() - t)

            for name in ["object.ply", "object.splat", "reference.png"]:
                api.upload_file(
                    path_or_fileobj=str(tmp_path / name),
                    path_in_repo=f"{prefix}/{name}",
                    repo_id=repo,
                    repo_type="dataset",
                )
        log.finish("done")
        result = {
            **log.to_dict(),
            "scene_path": prefix,
            "num_gaussians": len(obj),
            "model": model_label,
            "prompt": prompt,
            "seed": seed,
        }
    except errors.ReconstructionError as e:
        result = log.finish("failed", e.to_dict()).to_dict()
    except Exception as e:
        import traceback

        err = errors.object_failed(repr(e))
        err.detail = traceback.format_exc()[-errors.DETAIL_CHARS :]
        result = log.finish("failed", err.to_dict()).to_dict()

    _upload_json(api, repo, f"{prefix}/run.json", result)
    print(json.dumps(result))
    return result
