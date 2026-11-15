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

gsplat_image = (
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
)
image = gsplat_image.add_local_python_source("splat_studio")

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

# Preview (task 4.3): gsplat renders the viewer's view (colour + depth), SDXL img2img with
# a depth ControlNet restyles it. Reuses the gsplat build; SDXL + ControlNet in fp16 need
# about 10 GB, so an L4 (24 GB) is enough.
PREVIEW_GPU = "L4"
PREVIEW_TIMEOUT_S = 10 * 60  # keep in sync with splat_studio/space/jobs.py
# Edit scene (task 4.4) uses the same image: gsplat for training, diffusers for IP2P.
EDIT_GPU = "A10G"
EDIT_TIMEOUT_S = 30 * 60  # keep in sync with splat_studio/space/jobs.py

preview_image = (
    gsplat_image.pip_install(
        "diffusers==0.31.0",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "safetensors",
        "pillow",
    )
    .env({"HF_HOME": f"{MODELS_DIR}/hf"})
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
    from splat_studio.editing.simplify import simplify
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
        from concurrent.futures import ThreadPoolExecutor

        t = time.time()
        from trellis.pipelines import TrellisImageTo3DPipeline  # fail fast on a broken image

        log.stage("imports", time.time() - t)

        def prefetch_trellis() -> str:
            """Read TRELLIS's weight files into the page cache while FLUX loads and runs.

            Only file reads: building TRELLIS's models in this thread is not safe, because
            diffusers builds FLUX under accelerate's init_empty_weights, which patches
            torch.nn for the whole process and would leave TRELLIS's weights empty.
            """
            path = snapshot_download(
                models.IMAGE_TO_3D.repo_id, revision=models.IMAGE_TO_3D.revision
            )
            roots = [Path(path), Path(os.environ["TORCH_HOME"]), Path(os.environ["U2NET_HOME"])]
            for root in roots:
                for f in root.rglob("*") if root.exists() else []:
                    if f.is_file() and f.stat().st_size > 1_000_000:
                        with f.open("rb") as fh:
                            while fh.read(16 << 20):
                                pass
            return path

        loader = ThreadPoolExecutor(max_workers=1)
        trellis_future = loader.submit(prefetch_trellis)

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
            trellis_dir = trellis_future.result()
            loader.shutdown()
            log.stage("wait_prefetch", time.time() - t)
            t = time.time()
            trellis = TrellisImageTo3DPipeline.from_pretrained(trellis_dir)
            trellis.cuda()
            models_volume.commit()  # keep downloaded weights for the next run
            log.stage("load_3d_model", time.time() - t)

            t = time.time()
            out = trellis.run(image, seed=seed, formats=["gaussian"])
            raw = tmp_path / "raw.ply"
            # TRELLIS's own frame is z-up like our scenes; its default transform would turn
            # the object to y-up for other viewers, so keep the frame as it is.
            out["gaussian"][0].save_ply(str(raw), transform=None)
            log.stage("gaussians", time.time() - t)

            t = time.time()
            raw_gaussians = read_ply(raw)
            obj = simplify(normalise_object(raw_gaussians))
            write_ply(tmp_path / "object.ply", obj)
            (tmp_path / "object.splat").write_bytes(to_splat_bytes(obj))
            log.stage("simplify", time.time() - t)

            t = time.time()
            for name in ["object.ply", "object.splat", "reference.png"]:
                api.upload_file(
                    path_or_fileobj=str(tmp_path / name),
                    path_in_repo=f"{prefix}/{name}",
                    repo_id=repo,
                    repo_type="dataset",
                )
            log.stage("upload", time.time() - t)
        log.finish("done")
        result = {
            **log.to_dict(),
            "scene_path": prefix,
            "num_gaussians": len(obj),
            "raw_gaussians": len(raw_gaussians),
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


@app.function(
    image=preview_image,
    gpu=EDIT_GPU,
    timeout=EDIT_TIMEOUT_S,
    secrets=[hf_secret],
    volumes={MODELS_DIR: models_volume},
    retries=0,
)
def edit_scene(
    job_id: str,
    scene_id: str,
    number: int,
    camera: dict,
    prompt: str,
    region: list[float] | None = None,
    seed: int = 0,
    steps: int = 1500,
) -> dict:
    """Edit version `number` of a scene from a text instruction; saves the next version.

    Writes the result as a new version itself (scenes/<id>/v<n>/ with version.json),
    so it is kept even if nobody has the Space open when the job ends. Before/after
    renders of the user's view go to edits/<job_id>/.
    """
    t = time.time()
    from huggingface_hub import HfApi, hf_hub_download

    from splat_studio import models
    from splat_studio.editing import edit_worker, scene_edit
    from splat_studio.editing.preview import ViewCamera
    from splat_studio.reconstruction import errors
    from splat_studio.reconstruction.cost import RunLog
    from splat_studio.space.jobs import HubStore
    from splat_studio.space.versions import add_version

    repo = os.environ["SPLAT_STUDIO_DATASET"]
    api = HfApi()
    prefix = f"edits/{job_id}"
    log = RunLog(job_id=job_id, gpu=EDIT_GPU)
    log.stage("imports", time.time() - t)
    _upload_json(api, repo, f"{prefix}/run.json", log.to_dict())
    spec = models.IMAGE_EDIT

    def on_stage(name: str, seconds: float) -> None:
        """Log the stage and publish the run log, so the Space can show progress."""
        log.stage(name, seconds)
        try:
            _upload_json(api, repo, f"{prefix}/run.json", log.to_dict())
        except Exception as e:  # progress is nice to have; never fail the edit for it
            print(f"progress upload failed: {e!r}")

    try:
        cam = ViewCamera.from_dict(camera)
        area = scene_edit.Region(*region) if region else None
        t = time.time()
        ply = hf_hub_download(repo, f"scenes/{scene_id}/v{number}/scene.ply", repo_type="dataset")
        log.stage("download_scene", time.time() - t)
        with tempfile.TemporaryDirectory() as tmp:
            res = edit_worker.run_edit(
                Path(ply),
                cam,
                prompt,
                Path(tmp),
                model=(spec.repo_id, spec.revision),
                region=area,
                settings=edit_worker.EditSettings(steps=steps, seed=seed),
                on_stage=on_stage,
            )
            models_volume.commit()  # keep downloaded weights for the next run
            t = time.time()
            # The edited scene goes to edits/<job>/ first, so a failure while saving the
            # version below does not lose the training result.
            for name, path in [("before.png", res.before), ("after.png", res.after),
                               ("edit_example.png", res.example), ("scene.ply", res.ply),
                               ("scene.splat", res.splat)]:  # fmt: skip
                api.upload_file(path_or_fileobj=str(path), path_in_repo=f"{prefix}/{name}",
                                repo_id=repo, repo_type="dataset")  # fmt: skip
            store = HubStore(repo, os.environ.get("HF_TOKEN"), cache_dir=Path(tmp) / "cache")
            version = add_version(
                store,
                scene_id,
                {"ply": res.ply, "splat": res.splat},
                mode="edit_scene",
                prompt=prompt,
                model=spec.label,
                parent=number,
                extra={
                    "edit_job": job_id,
                    "region": area.to_list() if area else None,
                    "camera": camera,
                    "seed": seed,
                    "steps": steps,
                },
            )
            log.stage("upload", time.time() - t)
        log.finish("done")
        result = {
            **log.to_dict(),
            "scene_path": version.path,
            "version": version.number,
            "parent": number,
            "model": spec.label,
            "prompt": prompt,
            "region": area.to_list() if area else None,
            "num_gaussians": res.num_gaussians,
            "num_selected": res.num_selected,
            "size": list(res.size),
            **res.stats,
        }
    except errors.ReconstructionError as e:
        result = log.finish("failed", e.to_dict()).to_dict()
    except Exception as e:
        import traceback

        err = errors.edit_failed(repr(e))
        err.detail = traceback.format_exc()[-errors.DETAIL_CHARS :]
        result = log.finish("failed", err.to_dict()).to_dict()

    _upload_json(api, repo, f"{prefix}/run.json", result)
    print(json.dumps(result))
    return result


@app.function(image=preview_image, timeout=120, secrets=[hf_secret])
def check_edit_image() -> dict:
    """Cheap CPU check that the preview/edit image can import what the jobs need."""
    import diffusers
    import huggingface_hub

    out = {"huggingface_hub": huggingface_hub.__version__, "diffusers": diffusers.__version__}
    try:
        from splat_studio.editing import edit_worker, scene_edit  # noqa: F401
        from splat_studio.space.jobs import HubStore
        from splat_studio.space.versions import list_versions

        repo = os.environ.get("SPLAT_STUDIO_DATASET")
        out["imports"] = "ok"
        if repo:  # exercise the folder listing that failed in the first edit runs
            store = HubStore(repo, os.environ.get("HF_TOKEN"), cache_dir=Path("/tmp/c"))
            out["versions_of_first_scene"] = [
                v.number for s in store.list_dirs("scenes")[:1] for v in list_versions(store, s)
            ]
    except Exception as e:
        out["imports"] = f"{type(e).__name__}: {str(e)[:300]}"
    print(json.dumps(out))
    return out


PREVIEW_NEGATIVE = "blurry, low quality, distorted, deformed, oversaturated, text, watermark"
VIEWER_BACKGROUND = (31, 36, 48)  # the viewer's #1f2430, behind empty parts of the view


@app.function(
    image=preview_image,
    gpu=PREVIEW_GPU,
    timeout=PREVIEW_TIMEOUT_S,
    secrets=[hf_secret],
    volumes={MODELS_DIR: models_volume},
    retries=0,
)
def render_preview(
    job_id: str,
    scene_path: str,
    camera: dict,
    prompt: str,
    seed: int = 0,
    strength: float = 0.7,
    control_scale: float = 0.6,
    steps: int = 30,
) -> dict:
    """Render the scene from `camera` and restyle it; uploads to previews/<job_id>/.

    `camera` is `editing.preview.ViewCamera.to_dict()` (the viewer's camera).
    Writes render.png (plain view), depth.png (control image) and preview.png.
    """
    t = time.time()
    import numpy as np
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image

    from splat_studio import models
    from splat_studio.editing import preview
    from splat_studio.reconstruction import errors
    from splat_studio.reconstruction.cost import RunLog

    repo = os.environ["SPLAT_STUDIO_DATASET"]
    api = HfApi()
    prefix = f"previews/{job_id}"
    log = RunLog(job_id=job_id, gpu=PREVIEW_GPU)
    log.stage("imports", time.time() - t)
    _upload_json(api, repo, f"{prefix}/run.json", log.to_dict())
    model_label = f"{models.PREVIEW_BASE.label} + {models.PREVIEW_DEPTH.label}"

    try:
        cam = preview.ViewCamera.from_dict(camera)
        width, height = preview.output_size(cam.aspect)

        t = time.time()
        splat_file = hf_hub_download(repo, f"{scene_path}/scene.splat", repo_type="dataset")
        g = preview.read_splat(Path(splat_file).read_bytes())
        log.stage("download_scene", time.time() - t)

        t = time.time()
        from gsplat import rasterization

        def tensor(a):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device="cuda")

        # RGB+ED renders colour plus expected depth. The background is blended in below:
        # gsplat pads the channel count internally, so its `backgrounds` argument does
        # not fit this mode.
        with torch.no_grad():
            colors, alphas, _ = rasterization(
                means=tensor(g.means),
                quats=tensor(g.quats),
                scales=tensor(g.scales),
                opacities=tensor(g.opacities),
                colors=tensor(g.colors),
                viewmats=tensor(preview.viewmat(cam))[None],
                Ks=tensor(preview.intrinsics(cam, width, height))[None],
                width=width,
                height=height,
                render_mode="RGB+ED",
            )
        depth = colors[0, ..., 3].cpu().numpy()
        alpha = alphas[0, ..., 0].cpu().numpy()
        bg = np.array(VIEWER_BACKGROUND, dtype=np.float32) / 255
        rgb = colors[0, ..., :3].cpu().numpy() + (1 - alpha[..., None]) * bg
        rgb = np.clip(rgb, 0, 1)
        del colors, alphas
        torch.cuda.empty_cache()
        render = Image.fromarray((rgb * 255).round().astype(np.uint8))
        control = Image.fromarray(preview.depth_to_control(depth, alpha)).convert("RGB")
        log.stage("render", time.time() - t)
        if (alpha >= preview.MIN_ALPHA).mean() < 0.05:
            raise errors.preview_empty()

        t = time.time()
        from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

        controlnet = ControlNetModel.from_pretrained(
            models.PREVIEW_DEPTH.repo_id,
            revision=models.PREVIEW_DEPTH.revision,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            models.PREVIEW_BASE.repo_id,
            revision=models.PREVIEW_BASE.revision,
            controlnet=controlnet,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        models_volume.commit()  # keep downloaded weights for the next run
        log.stage("load_models", time.time() - t)

        t = time.time()
        styled = pipe(
            prompt=prompt.strip(),
            negative_prompt=PREVIEW_NEGATIVE,
            image=render,
            control_image=control,
            strength=float(strength),
            controlnet_conditioning_scale=float(control_scale),
            num_inference_steps=int(steps),
            guidance_scale=7.0,
            width=width,
            height=height,
            generator=torch.Generator("cpu").manual_seed(int(seed)),
        ).images[0]
        log.stage("restyle", time.time() - t)

        t = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            images = {"render.png": render, "depth.png": control, "preview.png": styled}
            for name, img in images.items():
                img.save(Path(tmp) / name)
                api.upload_file(
                    path_or_fileobj=str(Path(tmp) / name),
                    path_in_repo=f"{prefix}/{name}",
                    repo_id=repo,
                    repo_type="dataset",
                )
        log.stage("upload", time.time() - t)
        log.finish("done")
        result = {
            **log.to_dict(),
            "scene_path": prefix,
            "source_scene": scene_path,
            "model": model_label,
            "prompt": prompt,
            "seed": seed,
            "strength": strength,
            "control_scale": control_scale,
            "steps": steps,
            "size": [width, height],
            "num_gaussians": len(g),
        }
    except errors.ReconstructionError as e:
        result = log.finish("failed", e.to_dict()).to_dict()
    except Exception as e:
        import traceback

        err = errors.preview_failed(repr(e))
        err.detail = traceback.format_exc()[-errors.DETAIL_CHARS :]
        result = log.finish("failed", err.to_dict()).to_dict()

    _upload_json(api, repo, f"{prefix}/run.json", result)
    print(json.dumps(result))
    return result
