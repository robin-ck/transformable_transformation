"""Modal GPU functions for Splat Studio.

Deploy:  modal deploy modal_app/app.py
Spike:   modal run modal_app/app.py::gpu_probe
Object, preview, edit and image jobs are classes (ObjectGenerator, PreviewRenderer,
KontextRenderer, SceneEditor, ImageStudio) that load their models once per container and
stay warm for WARM_S after the last call.
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
TRAIN_GPU = "A10G"  # with CPU_CORES: keep in sync with splat_studio/reconstruction/settings.py
TRAIN_TIMEOUT_S = 60 * 60  # hard cap per job to protect the monthly credit
CPU_CORES = 4.0

# gsplat is built from source with its CUDA kernels. The prebuilt 1.5.3 wheels exist only
# for Python 3.10; on 3.11 pip took PyPI's pure-Python wheel, which JIT-compiles the
# kernels on the first render of every new container (70 to 90 s; speed-up-edit-jobs 1).
# The build container has no GPU, so the CUDA archs are set explicitly:
# 8.0 = A100, 8.6 = A10G, 8.9 = L4 and L40S, 9.0 = H100. A GPU whose arch is missing
# fails at the first render ("Failed to set maximum shared memory size"). Extensions
# without isolation see the torch installed above. Modal's add_python interpreter was
# built with clang, so extensions would look for clang++; point them at the image's gcc.
gsplat_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "colmap", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", "numpy<2", "huggingface_hub>=0.25")
    .pip_install("setuptools", "wheel", "ninja", "jaxtyping", "rich>=12")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0", "CC": "gcc", "CXX": "g++", "MAX_JOBS": "4"})
    .run_commands(
        f"git clone --depth 1 --recursive --shallow-submodules --branch v{GSPLAT_VERSION} "
        f"https://github.com/nerfstudio-project/gsplat {GSPLAT_DIR}",
        # The examples' requirements (fused-ssim etc.) do not list gsplat itself; installing
        # them first means nothing can replace the source build below.
        f"pip install --no-build-isolation -r {GSPLAT_DIR}/examples/requirements.txt",
        f"pip install --no-build-isolation --no-deps {GSPLAT_DIR}",
        # Fail the build, not a job, if the compiled kernels are missing.
        'python -c "import torch, gsplat; from gsplat import csrc; '
        f"assert gsplat.__version__ == '{GSPLAT_VERSION}', gsplat.__version__\"",
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
# Object, preview and edit containers stay up this long after their last call, with the
# models loaded. Longer cuts cold starts but bills idle GPU time.
WARM_S = 5 * 60

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
# about 10 GB, so an L4 (24 GB) is enough. Preview takes the first of these GPUs that is
# free instead of waiting for an L4: on 2026-11-15 and 16 previews waited 4 to 41 min for
# one (reopen-previews). The image's gsplat kernels cover all three (arch 8.6 and 8.9).
PREVIEW_GPUS = ["L4", "A10G", "L40S"]
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

# Futures with FLUX.1 Kontext dev (models.EDITORS["flux-kontext"]): the same gsplat render
# as the preview image, plus a diffusers release with FluxKontextPipeline (0.34+). The
# 12 B transformer and T5 in bf16 need about 34 GB, so an L40S (48 GB); A100-80GB or
# H100 if no L40S is free (gsplat's kernels are built for all three).
KONTEXT_GPUS = ["L40S", "A100-80GB", "H100"]

kontext_image = (
    gsplat_image.pip_install(
        "diffusers==0.35.1",
        "transformers==4.53.3",
        "accelerate==1.10.1",
        "sentencepiece",
        "protobuf",
        "safetensors",
        "pillow",
    )
    .env({"HF_HOME": f"{MODELS_DIR}/hf"})
    .add_local_python_source("splat_studio")
)

# Scene files cached on the models volume (splat_studio.scene_cache); versions never change.
SCENE_CACHE_DIR = f"{MODELS_DIR}/scene-cache"

# Image editing tab: FLUX.1-schnell for text-to-image and image-to-image. Same GPU as
# Add object (the 24 GB transformer needs 48 GB), but without TRELLIS's heavy build.
IMAGE_GPU = OBJECT_GPU
IMAGE_TIMEOUT_S = 10 * 60  # keep in sync with splat_studio/space/jobs.py

flux_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python="3.11")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
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


def _reload_volume() -> None:
    """See files other containers committed (cached weights and scenes) since this one began."""
    try:
        models_volume.reload()
    except Exception as e:  # e.g. files still open; the job then just misses the cache
        print(f"volume reload skipped: {e!r}")


def _cached_scene_file(repo: str, repo_path: str) -> tuple[Path, bool]:
    """A dataset scene file from the volume's scene cache, downloading it on a miss."""
    from huggingface_hub import hf_hub_download

    from splat_studio.scene_cache import SceneCache

    # A temporary HF cache, so scenes do not also pile up in the volume's HF_HOME.
    with tempfile.TemporaryDirectory(prefix="scene-dl-") as tmp:
        file, hit = SceneCache(Path(SCENE_CACHE_DIR)).fetch(
            repo_path, lambda path: hf_hub_download(repo, path, repo_type="dataset", cache_dir=tmp)
        )
    if not hit:
        models_volume.commit()
    return file, hit


def _warm_gsplat(mode: str = "RGB") -> None:
    """Load gsplat's compiled kernels and the CUDA context once, on a tiny scene."""
    import torch
    from gsplat import rasterization

    n = 16
    with torch.no_grad():
        rasterization(
            means=torch.randn(n, 3, device="cuda") * 0.1,
            quats=torch.nn.functional.normalize(torch.randn(n, 4, device="cuda"), dim=-1),
            scales=torch.full((n, 3), 0.05, device="cuda"),
            opacities=torch.full((n,), 0.5, device="cuda"),
            colors=torch.rand(n, 3, device="cuda"),
            viewmats=torch.eye(4, device="cuda")[None],
            Ks=torch.tensor([[32.0, 0, 16], [0, 32.0, 16], [0, 0, 1]], device="cuda")[None],
            width=32,
            height=32,
            render_mode=mode,
        )
    torch.cuda.synchronize()


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
def reconstruct(job_id: str, settings: dict | None = None) -> dict:
    """Download inputs from the HF dataset, reconstruct, upload scene + run log.

    `settings` (`reconstruction.settings.LIMITS`: steps, frames, max_side) override the
    default preset; the pipeline checks them before any work.
    """
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
                settings=settings,
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
                "settings": result.settings,
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


@app.cls(
    image=object_image,
    gpu=OBJECT_GPU,
    memory=48 * 1024,
    timeout=OBJECT_TIMEOUT_S,
    secrets=[hf_secret],
    volumes={MODELS_DIR: models_volume},
    retries=0,
    scaledown_window=WARM_S,
    max_containers=1,
)
class ObjectGenerator:
    """Text -> reference image -> 3D Gaussians, with FLUX and TRELLIS kept on the GPU.

    Test the 3D step alone with an image from TRELLIS's own examples:
      modal run modal_app/app.py::ObjectGenerator.generate --job-id test --prompt mailbox
        --reference-image /opt/trellis/assets/example_image/typical_misc_mailbox.png
    """

    @modal.enter()
    def load(self) -> None:
        from splat_studio.reconstruction.cost import LoadTimes

        self.loads = LoadTimes()
        self.flux = self.trellis = None
        try:
            self._load_models()
        except Exception as e:  # reported by the next job, which loads again
            print(f"model load failed in enter: {e!r}")

    def _load_models(self) -> None:
        import torch

        from splat_studio import models
        from splat_studio.reconstruction import errors

        t = time.time()
        from diffusers import FluxPipeline
        from trellis.pipelines import TrellisImageTo3DPipeline

        self.loads.record("imports", time.time() - t)

        if self.flux is None:
            t = time.time()
            try:
                self.flux = FluxPipeline.from_pretrained(
                    models.TEXT_TO_IMAGE.repo_id,
                    revision=models.TEXT_TO_IMAGE.revision,
                    torch_dtype=torch.bfloat16,
                ).to("cuda")
            except Exception as e:
                if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
                    raise errors.model_access(models.TEXT_TO_IMAGE.repo_id, repr(e)) from e
                raise
            self.loads.record("load_image_model", time.time() - t)

        if self.trellis is None:
            # Built after FLUX, in the same thread: FLUX's loader patches torch.nn while it
            # runs, so building TRELLIS alongside it would leave TRELLIS's weights empty.
            t = time.time()
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                models.IMAGE_TO_3D.repo_id, revision=models.IMAGE_TO_3D.revision
            )
            trellis = TrellisImageTo3DPipeline.from_pretrained(path)
            trellis.cuda()
            self.trellis = trellis
            self.loads.record("load_3d_model", time.time() - t)
        models_volume.commit()  # keep downloaded weights for the next container

    @modal.method()
    def generate(
        self,
        job_id: str,
        prompt: str,
        seed: int = 0,
        reference_image: str | None = None,
        reference_path: str | None = None,
    ) -> dict:
        """Uploads the object to objects/<job_id>/. A reference image skips the
        text-to-image step: `reference_path` is a file in the dataset (a user's photo),
        `reference_image` a path inside the container (for testing)."""
        import torch
        from huggingface_hub import HfApi

        from splat_studio import models
        from splat_studio.editing.objects import normalise_object
        from splat_studio.editing.simplify import simplify
        from splat_studio.hub_outputs import finish_with_outputs
        from splat_studio.reconstruction import errors
        from splat_studio.reconstruction.cost import RunLog
        from splat_studio.reconstruction.export import read_ply, to_splat_bytes, write_ply

        repo = os.environ["SPLAT_STUDIO_DATASET"]
        api = HfApi()
        prefix = f"objects/{job_id}"
        log = RunLog(job_id=job_id, gpu=OBJECT_GPU)
        _upload_json(api, repo, f"{prefix}/run.json", log.to_dict())
        model_label = f"{models.TEXT_TO_IMAGE.label} + {models.IMAGE_TO_3D.label}"
        if reference_image or reference_path:
            model_label = models.IMAGE_TO_3D.label

        try:
            if reference_path:
                from huggingface_hub import hf_hub_download

                reference_image = hf_hub_download(repo, reference_path, repo_type="dataset")
            if self.flux is None or self.trellis is None:
                self._load_models()  # enter failed: load here so the error is reported
            self.loads.apply(log)
            torch.cuda.reset_peak_memory_stats()

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                if reference_image:
                    from PIL import Image, ImageOps

                    # Phone photos store their rotation in EXIF; TRELLIS needs it applied.
                    image = ImageOps.exif_transpose(Image.open(reference_image))
                    if image.mode not in ("RGB", "RGBA"):
                        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
                    image.save(tmp_path / "reference.png")
                else:
                    t = time.time()
                    image = self.flux(
                        OBJECT_PROMPT.format(prompt=prompt.strip().rstrip(".")),
                        guidance_scale=0.0,
                        num_inference_steps=4,
                        max_sequence_length=256,
                        height=1024,
                        width=1024,
                        generator=torch.Generator("cpu").manual_seed(seed),
                    ).images[0]
                    image.save(tmp_path / "reference.png")
                    log.stage("image", time.time() - t)

                t = time.time()
                out = self.trellis.run(image, seed=seed, formats=["gaussian"])
                raw = tmp_path / "raw.ply"
                # TRELLIS's own frame is z-up like our scenes; its default transform would
                # turn the object to y-up for other viewers, so keep the frame as it is.
                out["gaussian"][0].save_ply(str(raw), transform=None)
                del out
                torch.cuda.empty_cache()
                log.stage("gaussians", time.time() - t)

                t = time.time()
                raw_gaussians = read_ply(raw)
                obj = simplify(normalise_object(raw_gaussians))
                write_ply(tmp_path / "object.ply", obj)
                (tmp_path / "object.splat").write_bytes(to_splat_bytes(obj))
                log.stage("simplify", time.time() - t)

                info = {
                    "scene_path": prefix,
                    "num_gaussians": len(obj),
                    "raw_gaussians": len(raw_gaussians),
                    "model": model_label,
                    "prompt": prompt,
                    "seed": seed,
                    "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                    "gpu_mem_total_gb": round(
                        torch.cuda.get_device_properties(0).total_memory / 2**30, 1
                    ),
                }
                files = {
                    f"{prefix}/{name}": tmp_path / name
                    for name in ["object.ply", "object.splat", "reference.png"]
                }
                result = finish_with_outputs(
                    api, repo, prefix, log, files, info, f"Object {job_id}"
                )
        except errors.ReconstructionError as e:
            result = log.finish("failed", e.to_dict()).to_dict()
        except Exception as e:
            import traceback

            err = errors.object_failed(repr(e))
            err.detail = traceback.format_exc()[-errors.DETAIL_CHARS :]
            result = log.finish("failed", err.to_dict()).to_dict()

        if result["status"] != "done":  # a done run.json went in with the object
            _upload_json(api, repo, f"{prefix}/run.json", result)
        print(json.dumps(result))
        return result


@app.cls(
    image=preview_image,
    gpu=EDIT_GPU,
    timeout=EDIT_TIMEOUT_S,
    secrets=[hf_secret],
    volumes={MODELS_DIR: models_volume},
    retries=0,
    scaledown_window=WARM_S,
    max_containers=1,
)
class SceneEditor:
    """Edit a scene version from a text instruction (InstructPix2Pix kept on the GPU).

    Run directly:
      modal run modal_app/app.py::SceneEditor.edit --job-id test --scene-id <id> --number 1
        --camera '<ViewCamera json>' --prompt 'make the table green'
    """

    @modal.enter()
    def load(self) -> None:
        from splat_studio.reconstruction.cost import LoadTimes

        self.loads = LoadTimes()
        self.editor = None
        try:
            self._load_models()
        except Exception as e:  # reported by the next job, which loads again
            print(f"model load failed in enter: {e!r}")

    def _load_models(self) -> None:
        from splat_studio import models

        t = time.time()
        from splat_studio.editing import edit_worker

        self.loads.record("imports", time.time() - t)
        t = time.time()
        _warm_gsplat()
        self.loads.record("warm_gsplat", time.time() - t)
        t = time.time()
        spec = models.IMAGE_EDIT
        self.editor = edit_worker.Editor(spec.repo_id, spec.revision)
        models_volume.commit()  # keep downloaded weights for the next container
        self.loads.record("load_editor", time.time() - t)

    @modal.method()
    def edit(
        self,
        job_id: str,
        scene_id: str,
        number: int,
        camera: dict,
        prompt: str,
        region: list[float] | None = None,
        seed: int = 0,
        budget: dict | None = None,
    ) -> dict:
        """Edit version `number` of a scene from a text instruction; saves the next version.

        Writes the result as a new version itself (scenes/<id>/v<n>/ with version.json),
        so it is kept even if nobody has the Space open when the job ends. Before/after
        renders of the user's view go to edits/<job_id>/. `budget` overrides the work
        settings (`scene_edit.BUDGET_LIMITS`); missing keys take the default preset.
        """
        from huggingface_hub import HfApi

        from splat_studio import models
        from splat_studio.editing import edit_worker, scene_edit
        from splat_studio.editing.preview import ViewCamera
        from splat_studio.hub_outputs import commit_files
        from splat_studio.reconstruction import errors
        from splat_studio.reconstruction.cost import RunLog
        from splat_studio.scene_cache import SceneCache
        from splat_studio.space.jobs import HubStore
        from splat_studio.space.versions import add_version

        repo = os.environ["SPLAT_STUDIO_DATASET"]
        api = HfApi()
        prefix = f"edits/{job_id}"
        log = RunLog(job_id=job_id, gpu=EDIT_GPU)
        spec = models.IMAGE_EDIT
        _upload_json(api, repo, f"{prefix}/run.json", log.to_dict())

        def on_stage(name: str, seconds: float) -> None:
            """Log the stage and publish the run log, so the Space can show progress."""
            log.stage(name, seconds)
            try:
                _upload_json(api, repo, f"{prefix}/run.json", log.to_dict())
            except Exception as e:  # progress is nice to have; never fail the edit for it
                print(f"progress upload failed: {e!r}")

        try:
            if self.editor is None:
                self._load_models()  # enter failed: load here so the error is reported
            self.loads.apply(log)
            try:
                settings = edit_worker.settings_for(budget, seed=seed)
            except ValueError as e:
                raise errors.edit_failed(str(e)) from e
            cam = ViewCamera.from_dict(camera)
            area = scene_edit.Region(*region) if region else None
            t = time.time()
            _reload_volume()
            ply, cached = _cached_scene_file(repo, f"scenes/{scene_id}/v{number}/scene.ply")
            on_stage("download_scene", time.time() - t)
            with tempfile.TemporaryDirectory() as tmp:
                res = edit_worker.run_edit(
                    ply,
                    cam,
                    prompt,
                    Path(tmp),
                    model=(spec.repo_id, spec.revision),
                    region=area,
                    settings=settings,
                    on_stage=on_stage,
                    editor=self.editor,
                )
                t = time.time()
                # The edited scene goes to edits/<job>/ first, so a failure while saving
                # the version below does not lose the training result.
                outputs = {"before.png": res.before, "after.png": res.after,
                           "edit_example.png": res.example, "scene.ply": res.ply,
                           "scene.splat": res.splat}  # fmt: skip
                commit_files(
                    api, repo, {f"{prefix}/{n}": p for n, p in outputs.items()}, f"Edit {job_id}"
                )
                store = HubStore(repo, os.environ.get("HF_TOKEN"), cache_dir=Path(tmp) / "cache")
                used = {k: getattr(settings, k) for k in scene_edit.BUDGET_LIMITS}
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
                        **used,
                        "preset": scene_edit.preset_of(used),
                    },
                )
                log.stage("upload", time.time() - t)
                # An edit on this new version can then skip the download.
                try:
                    cache = SceneCache(Path(SCENE_CACHE_DIR))
                    cache.put(f"{version.path}/scene.ply", res.ply)
                    cache.put(f"{version.path}/scene.splat", res.splat)
                    models_volume.commit()
                except Exception as e:  # the cache is an optimisation only
                    print(f"scene cache write failed: {e!r}")
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
                "scene_cached": cached,
                "preset": scene_edit.preset_of(used),
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


VIEWER_BACKGROUND = (31, 36, 48)  # the viewer's #1f2430, behind empty parts of the view


VIEW_FILES = ("render.png", "depth.png", "view.json")  # a viewpoint's stored files
KEEP_IMAGES = 16  # finished futures kept in memory per container, as parents for the next


def _jpeg(img, quality: int = 90) -> bytes:
    """A PIL image as JPEG bytes (returned to the Space, which shows it at once)."""
    import io

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _png(img) -> bytes:
    """A PIL image as PNG bytes (the mask returned to the Space)."""
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _printable(result: dict) -> str:
    return json.dumps({k: v for k, v in result.items() if not isinstance(v, bytes)})


class _ViewRestyler:
    """Render the scene from the viewer's camera and restyle it with one editor.

    Each subclass below is a Modal class for one `models.EDITORS` entry (`EDITOR`), with
    its own image and GPU. The models and gsplat's kernels are set up once per container,
    in `load`. Viewpoint images (add-viewpoint-futures) and recent restyled images stay in
    memory, so later futures of the same viewpoint, or built on a recent future, download
    nothing.
    """

    EDITOR = "sdxl-depth"  # models.EDITORS key; set by each subclass
    GPUS = PREVIEW_GPUS

    @modal.enter()
    def load(self) -> None:
        from splat_studio.reconstruction.cost import LoadTimes

        self.loads = LoadTimes()
        self.editor = None
        self.masker = None  # editing.masking.ObjectMasker, for editors that can mask
        self.views: dict[str, tuple] = {}  # view prefix -> (render, control, view info)
        self.images: dict[str, object] = {}  # dataset path -> restyled PIL image
        try:
            self._load_models()
        except Exception as e:  # reported by the next job, which loads again
            print(f"model load failed in enter: {e!r}")

    def _load_models(self) -> None:
        t = time.time()
        from splat_studio.editing import restyle

        editor = restyle.make_editor(self.EDITOR)
        self.loads.record("imports", time.time() - t)

        # The kernels are prebuilt in the image; this loads them and the CUDA context.
        t = time.time()
        _warm_gsplat("RGB+ED")
        self.loads.record("warm_gsplat", time.time() - t)

        t = time.time()
        try:
            editor.load()
        except Exception as e:  # a gated model whose licence the token has not accepted
            if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
                from splat_studio import models
                from splat_studio.reconstruction import errors

                repo_id = models.editor_spec(self.EDITOR).models[0].repo_id
                raise errors.model_access(repo_id, repr(e)) from e
            raise
        self.loads.record("load_models", time.time() - t)

        # "Object to change" (add-object-masking): detector + SAM next to the editor.
        from splat_studio import models

        spec = models.editor_spec(self.EDITOR)
        if spec.can_mask and self.masker is None:
            from splat_studio.editing import masking

            t = time.time()
            masker = masking.ObjectMasker(*spec.mask_models)
            masker.load()
            self.masker = masker
            self.loads.record("load_mask_models", time.time() - t)
        self.editor = editor
        models_volume.commit()  # keep downloaded weights for the next container

    @modal.method()
    def warm(self) -> dict:
        """Start a container (models load in `load`) before the user clicks Make."""
        return {"warm": self.editor is not None}

    def _render(self, repo: str, scene_path: str, camera: dict, stage) -> tuple:
        """Render colour and depth of `scene_path` from `camera`.

        Returns (render, control, view info); `stage(name, seconds)` records timings.
        Raises `errors.preview_empty` for a view that shows almost nothing.
        """
        import numpy as np
        import torch
        from gsplat import rasterization
        from PIL import Image

        from splat_studio.editing import preview
        from splat_studio.reconstruction import errors

        cam = preview.ViewCamera.from_dict(camera)
        width, height = preview.output_size(cam.aspect)

        t = time.time()
        _reload_volume()
        splat_file, cached = _cached_scene_file(repo, f"{scene_path}/scene.splat")
        g = preview.read_splat(Path(splat_file).read_bytes())
        stage("download_scene", time.time() - t)

        t = time.time()

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
        seconds = time.time() - t
        stage("render", seconds)
        if (alpha >= preview.MIN_ALPHA).mean() < 0.05:
            raise errors.preview_empty()
        info = {
            "camera": camera,
            "size": [width, height],
            "source_scene": scene_path,
            "num_gaussians": len(g),
            "scene_cached": cached,
            "render_seconds": round(seconds, 2),
            "created_at": time.time(),
        }
        return render, control, info

    def _stored_view(self, repo: str, view_prefix: str) -> tuple | None:
        """A viewpoint's (render, control, info) from memory, else from the dataset."""
        if view_prefix in self.views:
            return self.views[view_prefix]
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        from PIL import Image

        try:
            with tempfile.TemporaryDirectory() as tmp:
                paths = {
                    n: hf_hub_download(repo, f"{view_prefix}/{n}", repo_type="dataset",
                                       cache_dir=tmp)
                    for n in VIEW_FILES
                }  # fmt: skip
                render = Image.open(paths["render.png"]).convert("RGB")
                control = Image.open(paths["depth.png"]).convert("RGB")
                info = json.loads(Path(paths["view.json"]).read_text())
        except EntryNotFoundError:
            return None
        self.views[view_prefix] = (render, control, info)
        return self.views[view_prefix]

    def _input_image(self, repo: str, path: str):
        """A parent future's restyled image (dataset path), from memory or the dataset."""
        if path in self.images:
            return self.images[path]
        from huggingface_hub import hf_hub_download
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            local = hf_hub_download(repo, path, repo_type="dataset", cache_dir=tmp)
            return Image.open(local).convert("RGB")

    def _keep_image(self, path: str, img) -> None:
        self.images[path] = img
        while len(self.images) > KEEP_IMAGES:
            self.images.pop(next(iter(self.images)))

    @modal.method()
    def render_view(self, scene_path: str, camera: dict, out_prefix: str) -> dict:
        """Render a viewpoint once (add-viewpoint-futures) and store it under `out_prefix`
        (`viewpoints/<scene>/<vp>`): render.png, depth.png and view.json in one commit.

        Returns {"status", "view" (view.json), "image_jpeg" (the render)} or a failed
        {"status", "error"}.
        """
        from huggingface_hub import HfApi

        from splat_studio.hub_outputs import commit_files
        from splat_studio.reconstruction import errors

        repo = os.environ["SPLAT_STUDIO_DATASET"]
        stages: dict[str, float] = {}
        try:
            if self.editor is None:
                self._load_models()  # the next future then finds the models loaded
            render, control, info = self._render(
                repo, scene_path, camera, lambda n, s: stages.__setitem__(n, round(s, 2))
            )
            t = time.time()
            with tempfile.TemporaryDirectory() as tmp:
                render.save(Path(tmp) / "render.png")
                control.save(Path(tmp) / "depth.png")
                files = {f"{out_prefix}/render.png": Path(tmp) / "render.png",
                         f"{out_prefix}/depth.png": Path(tmp) / "depth.png",
                         f"{out_prefix}/view.json": info}  # fmt: skip
                commit_files(HfApi(), repo, files, f"Viewpoint {out_prefix}")
            stages["upload"] = round(time.time() - t, 2)
            self.views[out_prefix] = (render, control, info)
            result = {"status": "done", "view": info, "stages": stages,
                      "image_jpeg": _jpeg(render)}  # fmt: skip
        except errors.ReconstructionError as e:
            result = {"status": "failed", "error": e.to_dict(), "stages": stages}
        except Exception as e:
            result = {"status": "failed", "error": errors.preview_failed(repr(e)).to_dict(),
                      "stages": stages}  # fmt: skip
        print(_printable(result))
        return result

    @modal.method()
    def detect(self, view_prefix: str, object_query: str) -> dict:
        """Detect and segment `object_query` in a viewpoint's plain render, without editing
        (add-detection-preview, "Detect objects"). Stores nothing.

        Returns {"status", "mask" (as run.json's), "stages", "detections_jpeg"} or a failed
        {"status", "error", "stages"} (`object_not_found` when nothing matches).
        """
        from splat_studio import models
        from splat_studio.editing import annotate, masking
        from splat_studio.reconstruction import errors

        repo = os.environ["SPLAT_STUDIO_DATASET"]
        stages: dict[str, float] = {}
        query = " ".join((object_query or "").split())
        try:
            if self.editor is None:
                self._load_models()
            if self.masker is None:
                name = models.editor_spec(self.EDITOR).name
                raise errors.preview_failed(f"{name} cannot limit changes to an object")
            if not query:
                raise errors.preview_failed("no object named")
            t = time.time()
            view = self._stored_view(repo, view_prefix)
            if view is None:
                raise errors.preview_failed(f"viewpoint {view_prefix} has no stored render yet")
            render = view[0]
            stages["load_view"] = round(time.time() - t, 2)
            t = time.time()
            found = self.masker.detect(render, query)
            stages["detect"] = round(time.time() - t, 2)
            if not found:
                raise errors.object_not_found(query)
            t = time.time()
            masks = self.masker.segment(render, found)
            stages["segment"] = round(time.time() - t, 2)
            t = time.time()
            labelled = annotate.draw_detections(render, found, masks)
            info = masking.mask_info(query, found, masks.any(0), False, self.masker.labels)
            stages["annotate"] = round(time.time() - t, 2)
            result = {"status": "done", "mask": info, "stages": stages,
                      "detections_jpeg": _jpeg(labelled, 85)}  # fmt: skip
        except errors.ReconstructionError as e:
            result = {"status": "failed", "error": e.to_dict(), "stages": stages}
        except Exception as e:
            result = {"status": "failed", "error": errors.preview_failed(repr(e)).to_dict(),
                      "stages": stages}  # fmt: skip
        print(_printable(result))
        return result

    @modal.method()
    def render(
        self,
        job_id: str,
        scene_path: str,
        camera: dict,
        prompt: str,
        seed: int = 0,
        strength: float = 0.7,
        control_scale: float = 0.6,
        steps: int | None = None,
        view_prefix: str | None = None,
        input_path: str | None = None,
        object_query: str | None = None,
        area_png: bytes | None = None,
        instructions: dict | None = None,
    ) -> dict:
        """Restyle a view; uploads preview.png to previews/<job_id>/ and returns the image.

        Without `view_prefix` (the Preview tool): renders `camera`, and also uploads
        render.png and depth.png to previews/<job_id>/.
        With `view_prefix` (a future): starts from the viewpoint's stored render and depth
        and does not render; if the viewpoint has no stored images yet (a new viewpoint
        from the current view), renders `camera` and stores them under `view_prefix` in
        the same commit. `input_path` (a parent future's preview.png) replaces the render
        as the image to restyle; the depth always comes from the viewpoint.
        The result carries the restyled image as JPEG bytes in `image_jpeg`.
        `object_query` (add-object-masking, editors with a masker only) names the object to
        change: it is detected and segmented in the start image first (fails with
        `object_not_found` if absent), and the editor's result is kept only inside the
        grown, feathered mask; mask.png goes up too and comes back as `mask_png`.
        `area_png` (add-area-layout, same editors) is a painted area instead: the editor
        gets a crop around it (`editing.area.crop_box`), and its result is kept only
        inside the grown, feathered area; area.png and mask.png go up too.
        `instructions` (edit-editor-instructions) are the scene's own texts added to the
        prompt and SDXL's negative prompt (`editing.instructions`); None uses the defaults.
        `camera` is `editing.preview.ViewCamera.to_dict()`.
        """
        import threading

        import torch
        from huggingface_hub import HfApi

        from splat_studio import models
        from splat_studio.editing import annotate, area, masking
        from splat_studio.editing import instructions as instr
        from splat_studio.hub_outputs import finish_with_outputs
        from splat_studio.reconstruction import errors
        from splat_studio.reconstruction.cost import RunLog, gpu_from_device_name

        repo = os.environ["SPLAT_STUDIO_DATASET"]
        api = HfApi()
        prefix = f"previews/{job_id}"
        object_query = " ".join((object_query or "").split()) or None
        gpu = gpu_from_device_name(torch.cuda.get_device_name(0), self.GPUS[0])
        log = RunLog(job_id=job_id, gpu=gpu)
        # The "running" run.json goes up in the background; it is joined before the final
        # commit, so it can never overwrite the finished one.
        started = threading.Thread(
            target=_upload_json, args=(api, repo, f"{prefix}/run.json", log.to_dict())
        )
        started.start()
        spec = models.editor_spec(self.EDITOR)
        steps = int(steps or spec.steps)  # None: the editor's own default

        try:
            if self.editor is None:
                self._load_models()  # enter failed: load here so the error is reported
            self.loads.apply(log)
            files: dict = {}
            view = self._stored_view(repo, view_prefix) if view_prefix else None
            if view is None:
                if camera is None:
                    raise errors.preview_failed(f"viewpoint {view_prefix} has no stored images")
                view = self._render(repo, scene_path, camera, log.stage)
                render, control, view_info = view
                out = view_prefix or prefix
                files[f"{out}/render.png"], files[f"{out}/depth.png"] = render, control
                if view_prefix:
                    files[f"{view_prefix}/view.json"] = view_info
                    self.views[view_prefix] = view
            render, control, view_info = view

            image = render
            if input_path:
                t = time.time()
                image = self._input_image(repo, input_path)
                if image.size != render.size:
                    image = image.resize(render.size)
                log.stage("download_input", time.time() - t)

            mask = mask_meta = detections_jpeg = None
            if object_query:  # before the edit, so "not found" fails in about a second
                if self.masker is None:
                    raise errors.preview_failed(f"{spec.name} cannot limit changes to an object")
                t = time.time()
                found = self.masker.detect(image, object_query)
                log.stage("detect", time.time() - t)
                if not found:
                    raise errors.object_not_found(object_query)
                t = time.time()
                masks = self.masker.segment(image, found)  # one per detection
                hard = masks.any(0)
                removal = masking.is_removal(prompt)
                mask = masking.soft_mask(hard, removal)
                mask_meta = masking.mask_info(object_query, found, hard, removal,
                                              self.masker.labels)  # fmt: skip
                log.stage("segment", time.time() - t)
                t = time.time()  # labelled image of what was detected (add-detection-preview)
                detected = annotate.draw_detections(image, found, masks)
                files[f"{prefix}/detections.jpg"] = detected
                detections_jpeg = _jpeg(detected, 85)
                log.stage("annotate", time.time() - t)

            box = area_meta = None
            if area_png:  # "Area to change" (add-area-layout)
                if object_query:
                    raise errors.preview_failed("an object and a drawn area were both given")
                if self.masker is None:
                    raise errors.preview_failed(f"{spec.name} cannot limit changes to an area")
                t = time.time()
                try:
                    hard = area.check(area.from_png(area_png, image.size))
                except area.AreaError as e:
                    raise errors.area_empty() from e
                removal = masking.is_removal(prompt)
                mask = masking.soft_mask(hard, removal)
                box = area.crop_box(hard)
                area_meta = area.area_info(hard, box, removal)
                files[f"{prefix}/area.png"] = masking.mask_image(hard.astype("float32"))
                log.stage("area", time.time() - t)

            # With a named object, Kontext gets an explicit instruction (masking.object_prompt);
            # with a drawn area, one that puts the change in the middle of the crop
            # (area.area_prompt); both add the scene's instruction texts. `prompt` stays
            # what the user wrote and is what info records as "prompt".
            texts = instr.resolve(instructions)
            if box is not None:
                editor_prompt = area.area_prompt(prompt, texts)
            elif mask is not None:
                editor_prompt = masking.object_prompt(prompt, object_query, texts)
            else:
                editor_prompt = prompt

            t = time.time()
            styled = self.editor.restyle(
                image.crop(box) if box else image,
                control.crop(box) if box else control,
                editor_prompt,
                seed,
                strength=strength,
                control_scale=control_scale,
                steps=steps,
                negative_prompt=texts["negative"],
            )
            if box is not None:
                styled = area.paste(image, styled, box)
            log.stage("restyle", time.time() - t)
            mask_png = None
            if mask is not None:
                t = time.time()
                styled = masking.composite(image, styled, mask)
                mask_img = masking.mask_image(mask)
                files[f"{prefix}/mask.png"] = mask_img
                mask_png = _png(mask_img)
                log.stage("composite", time.time() - t)
            files[f"{prefix}/preview.png"] = styled
            self._keep_image(f"{prefix}/preview.png", styled)

            info = {
                "scene_path": prefix,
                "source_scene": scene_path,
                "model": spec.label,
                "editor": spec.key,
                "prompt": prompt,
                "editor_prompt": editor_prompt,
                "negative_prompt": texts["negative"] if spec.key == "sdxl-depth" else None,
                "instructions": texts,
                "seed": seed,
                "strength": strength,
                "control_scale": control_scale,
                "steps": steps,
                "size": view_info["size"],
                "num_gaussians": view_info.get("num_gaussians"),
                "scene_cached": view_info.get("scene_cached"),
                "view_prefix": view_prefix,
                "input_path": input_path,
                "view_rendered": f"{view_prefix or prefix}/render.png" in files,
                "object_query": object_query,
                "mask": mask_meta,
                "area": area_meta,
            }
            jpeg = _jpeg(styled)
            started.join()
            with tempfile.TemporaryDirectory() as tmp:
                upload = {}
                for path, value in files.items():
                    if hasattr(value, "save"):  # a PIL image
                        local = Path(tmp) / path.replace("/", "_")
                        value.save(local)
                        value = local
                    upload[path] = value
                result = finish_with_outputs(
                    api, repo, prefix, log, upload, info, f"Preview {job_id}"
                )
            result["image_jpeg"] = jpeg
            if mask_png:
                result["mask_png"] = mask_png
            if detections_jpeg:
                result["detections_jpeg"] = detections_jpeg
        except errors.ReconstructionError as e:
            result = log.finish("failed", e.to_dict()).to_dict()
        except Exception as e:
            import traceback

            err = errors.preview_failed(repr(e))
            err.detail = traceback.format_exc()[-errors.DETAIL_CHARS :]
            result = log.finish("failed", err.to_dict()).to_dict()

        started.join()
        if result["status"] != "done":  # a done run.json went in with the images
            _upload_json(api, repo, f"{prefix}/run.json", result)
        print(_printable(result))
        return result


@app.cls(
    image=preview_image,
    gpu=PREVIEW_GPUS,
    timeout=PREVIEW_TIMEOUT_S,
    secrets=[hf_secret],
    volumes={MODELS_DIR: models_volume},
    retries=0,
    scaledown_window=WARM_S,
    max_containers=1,
)
class PreviewRenderer(_ViewRestyler):
    """Preview tool and futures with SDXL + depth ControlNet (the default editor)."""

    EDITOR = "sdxl-depth"
    GPUS = PREVIEW_GPUS


@app.cls(
    image=kontext_image,
    gpu=KONTEXT_GPUS,
    timeout=PREVIEW_TIMEOUT_S,
    secrets=[hf_secret],
    volumes={MODELS_DIR: models_volume},
    retries=0,
    scaledown_window=WARM_S,
    max_containers=1,
)
class KontextRenderer(_ViewRestyler):
    """Futures with FLUX.1 Kontext dev: instruction edits that keep the rest of the view.
    Test:  modal run modal_app/app.py::KontextRenderer.warm"""

    EDITOR = "flux-kontext"
    GPUS = KONTEXT_GPUS


@app.cls(
    image=flux_image,
    gpu=IMAGE_GPU,
    memory=48 * 1024,
    timeout=IMAGE_TIMEOUT_S,
    secrets=[hf_secret],
    volumes={MODELS_DIR: models_volume},
    retries=0,
    scaledown_window=WARM_S,
    max_containers=1,
)
class ImageStudio:
    """Image editing tab: FLUX.1-schnell text-to-image, and image-to-image that shares
    the same weights. Test:  modal run modal_app/app.py::ImageStudio.generate
      --job-id test --prompt "a green roof with wildflowers"
    """

    @modal.enter()
    def load(self) -> None:
        from splat_studio.reconstruction.cost import LoadTimes

        self.loads = LoadTimes()
        self.t2i = self.i2i = None
        try:
            self._load_models()
        except Exception as e:  # reported by the next job, which loads again
            print(f"model load failed in enter: {e!r}")

    def _load_models(self) -> None:
        import torch

        from splat_studio import models
        from splat_studio.reconstruction import errors

        t = time.time()
        from diffusers import FluxImg2ImgPipeline, FluxPipeline

        self.loads.record("imports", time.time() - t)
        t = time.time()
        try:
            self.t2i = FluxPipeline.from_pretrained(
                models.TEXT_TO_IMAGE.repo_id,
                revision=models.TEXT_TO_IMAGE.revision,
                torch_dtype=torch.bfloat16,
            ).to("cuda")
        except Exception as e:
            if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
                raise errors.model_access(models.TEXT_TO_IMAGE.repo_id, repr(e)) from e
            raise
        self.i2i = FluxImg2ImgPipeline(**self.t2i.components)  # same weights, no copy
        models_volume.commit()  # keep downloaded weights for the next container
        self.loads.record("load_image_model", time.time() - t)

    @modal.method()
    def generate(
        self,
        job_id: str,
        prompt: str,
        seed: int = 0,
        source_path: str | None = None,
        strength: float = 0.7,
        aspect: float = 1.0,
    ) -> dict:
        """Uploads images/<job_id>/result.png. With `source_path` (a dataset file) the
        image is changed from it (`strength`: 0 keeps it, 1 ignores it); without, it is
        made from the prompt at about one megapixel with the given width/height ratio."""
        import torch
        from huggingface_hub import HfApi, hf_hub_download

        from splat_studio import models
        from splat_studio.editing import images
        from splat_studio.hub_outputs import finish_with_outputs
        from splat_studio.reconstruction import errors
        from splat_studio.reconstruction.cost import RunLog

        repo = os.environ["SPLAT_STUDIO_DATASET"]
        api = HfApi()
        prefix = f"images/{job_id}"
        log = RunLog(job_id=job_id, gpu=IMAGE_GPU)
        _upload_json(api, repo, f"{prefix}/run.json", log.to_dict())
        generator = torch.Generator("cpu").manual_seed(int(seed))

        try:
            if self.t2i is None:
                self._load_models()  # enter failed: load here so the error is reported
            self.loads.apply(log)
            t = time.time()
            if source_path:
                from PIL import Image, ImageOps

                local = hf_hub_download(repo, source_path, repo_type="dataset")
                source = ImageOps.exif_transpose(Image.open(local)).convert("RGB")
                width, height = images.output_size(source.width / source.height)
                source = source.resize((width, height), Image.LANCZOS)
                steps = images.img2img_steps(strength)
                result = self.i2i(
                    prompt=prompt.strip(),
                    image=source,
                    strength=float(strength),
                    guidance_scale=0.0,
                    num_inference_steps=steps,
                    max_sequence_length=256,
                    width=width,
                    height=height,
                    generator=generator,
                ).images[0]
            else:
                width, height = images.output_size(aspect)
                steps = images.BASE_STEPS
                result = self.t2i(
                    prompt.strip(),
                    guidance_scale=0.0,
                    num_inference_steps=steps,
                    max_sequence_length=256,
                    width=width,
                    height=height,
                    generator=generator,
                ).images[0]
            log.stage("image", time.time() - t)

            info = {
                "scene_path": prefix,
                "model": models.TEXT_TO_IMAGE.label,
                "mode": "image" if source_path else "text",
                "prompt": prompt,
                "seed": seed,
                "steps": steps,
                "size": [width, height],
            }
            if source_path:
                info.update(source_path=source_path, strength=strength)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "result.png"
                result.save(out)
                result_info = finish_with_outputs(
                    api, repo, prefix, log, {f"{prefix}/result.png": out}, info, f"Image {job_id}"
                )
        except errors.ReconstructionError as e:
            result_info = log.finish("failed", e.to_dict()).to_dict()
        except Exception as e:
            import traceback

            err = errors.image_failed(repr(e))
            err.detail = traceback.format_exc()[-errors.DETAIL_CHARS :]
            result_info = log.finish("failed", err.to_dict()).to_dict()

        if result_info["status"] != "done":  # a done run.json went in with the image
            _upload_json(api, repo, f"{prefix}/run.json", result_info)
        print(json.dumps(result_info))
        return result_info
