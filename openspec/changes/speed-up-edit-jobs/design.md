# Design: speed-up-edit-jobs

## Baseline
The numbers come from the runs recorded in MVP tasks.md (4.2 to 4.4), on the test scene with 1.55 M Gaussians (`.ply` 367 MB, `.splat` 50 MB).

| Job (GPU) | Total | Largest stages |
|---|---|---|
| Edit scene (A10G) | 7.5 to 8 min, about $0.15 | download 40 s, render views 70 to 90 s, load editor 14 s, first edits 62 s, train 280 s, upload 25 s |
| Preview (A10G, as L4 stand-in) | 2 min 40 s | render 75 s, model download + load 63 s (download one-time), restyle 11 s, upload 9 s |
| Add object (L40S, weights cached) | 2 min 11 s | FLUX load 22 s, image 4 s, TRELLIS load 22 s, Gaussians 13 s, about 65 s in imports and uploads |

Inside Edit scene's train stage, the 150 re-edits (one every 10 of 1,500 steps, about 1.3 s each) take an estimated 200 s. The gsplat steps take only about 80 s.

## gsplat kernels
- **Cause:** the gsplat wheel index (`docs.gsplat.studio/whl/pt24cu124`) has 1.5.3 only for cp310. The image runs Python 3.11, so pip installs PyPI's `gsplat-1.5.3-py3-none-any.whl`. That wheel has no compiled kernels, so gsplat JIT-compiles them on first use into a cache that disappears with the container. This explains the 70 to 90 s render stages in Preview and Edit scene.
- **Choice:** build gsplat from the clone the image already makes in `/opt/gsplat`:
  - `git clone --recursive`, which brings the glm submodule.
  - `pip install --no-build-isolation /opt/gsplat`, with `TORCH_CUDA_ARCH_LIST="8.6;8.9"` to cover A10G, L4 and L40S.
  - This is the same way fused-ssim already builds in this image: the devel image has nvcc, and no GPU is needed at build time.
  - The build step ends with `python -c "from gsplat import csrc"`, so an image that would JIT-compile at runtime fails at build time instead.
  - `examples/requirements.txt` must not put the PyPI wheel back. Install it before gsplat, or check the version afterwards.
- **Rejected:**
  - A Python 3.10 image, so the prebuilt wheel would fit. The package targets 3.11, and CI does not test 3.10.
  - Caching the JIT output on the volume (`TORCH_EXTENSIONS_DIR`). The first job would still pay, and parallel containers would race on the build lock.
- **Side effect:** reconstruction training (`simple_trainer.py`) uses the same gsplat, so its compile time goes too.

## Warm containers
- Preview, Edit scene and Add object become `@app.cls` classes (`Previewer`, `SceneEditor`, `ObjectMaker`), with `scaledown_window=WARM_S` (300 s by default, one constant).
  - `@modal.enter()` loads the models.
  - One `@modal.method()` per class runs a job. It takes the same arguments and writes the same `run.json` and outputs as today's function.
  - Each method calls `models_volume.reload()` first, so it sees cache entries committed by other containers.
- **Space side:** `modal_client.spawn` gets a class form, `modal.Cls.from_name(APP_NAME, cls)().run.spawn(...)`. `poll` stays as it is, because a method call also returns a `FunctionCall` id. Reconstruction and the check functions stay plain functions.
- **Edit scene:** the InstructPix2Pix pipeline does not depend on the prompt, so `enter` loads it and each job encodes its own prompt. `Editor` splits into load and prompt parts.
- **Add object:** FLUX (about 33 GB in bf16 with its text encoders) and TRELLIS together come close to the L40S's 48 GB once activations are added.
  - Measure first. If both fit, keep both on the GPU.
  - Otherwise, keep both in CPU RAM (raise `memory` to 64 GB) and move each model to the GPU only for its step. That takes seconds instead of the 22 s each now takes to load from disk.
  - `enter` loads the two models one after the other. Loading them in parallel stays unsafe, because accelerate's `init_empty_weights` patches torch for the whole process.
- **Cost:** Modal bills a warm container while it idles. The run log's cost estimate keeps covering the job's own time only, and the README states the idle cost per GPU type.
- **Rejected for now:**
  - Modal memory snapshots: revisit only if cold starts still dominate after this change.
  - `min_containers`: it bills all day, which does not fit the free credit.

## Scene file cache on the volume
- Versions never change, so a cached copy stays valid.
  - Layout: `/models/scene-cache/scenes/<scene>/v<n>/<file>`.
  - `scene_cache.fetch(repo_path, download)` returns the local file. On a miss it downloads to a temporary name, renames it when complete (so another container never reads half a file), touches a `last_used` stamp and commits the volume.
- Edit scene also puts its own `scene.ply` and `scene.splat` into the cache under the new version's path, once `add_version` has succeeded. An edit on that version then skips the download. Preview caches the `.splat` the same way.
- **Size cap:** `CACHE_CAP_GB`, 10 GB by default. After a write, the least recently used versions are deleted until the cache is under the cap.
- **`hf_transfer`** (Apache-2.0): added to the images with `HF_HUB_ENABLE_HF_TRANSFER=1`. Keep it only if the measured download and upload times drop. If the dataset is stored on Xet, `hf_xet` handles the transfers and `hf_transfer` changes nothing.
- The plain module has no Modal import, so it is unit tested with a fake downloader.

## Edit scene budget
- **Settings:** `EditSettings` already holds the settings: `n_views` 40, `steps` 1500, `edit_every` 10, `denoise_steps` 20. `edit_scene` gets an optional `budget` dict that overrides them, for the comparison runs from the CLI. The Space keeps sending only the prompt, region and seed. The values used are recorded in `version.json` (`extra`), as `steps` and `seed` are today.
- **Batching:** the first edits run 4 views per UNet batch (batch 12 with guidance). Keep this only if it is at least 20 % faster than one view at a time.
- **Comparison runs:** each run uses the same camera, region, prompt and seed as job `20261113-165311-5ec895` (v3, "make the wooden table bright green").

  | Run | Views | Steps | Re-edit every | Denoise steps |
  |---|---|---|---|---|
  | A | 40 | 1500 | 10 | 20 (today) |
  | B | 40 | 1500 | 10 | 10 |
  | C | 40 | 1500 | 20 | 10 |
  | D | 24 | 1000 | 20 | 10 |

  Pick the cheapest run whose `after.png` and three orbit renders still show the table clearly green, the rest of the room unchanged and no more spill onto the floor than v3.

## GPU choice
- **Approximate Modal prices per hour:** L4 $0.80, A10G $1.10, L40S $1.95. An L40S costs about 1.8 times an A10G per second.
- **Edit scene:** switch `EDIT_GPU` to L40S if the chosen settings run at least 1.5 times faster there. The cost per edit then stays within about 1.2 times the A10G's.
- **Preview:** it stays on L4 unless it queues for more than 1 min again during the checks. In that case it moves to A10G, which the 4.3 run already used.

## Targets
Times count from container start and leave out time spent waiting for a free GPU. They are checked on the test scene.

| Job | Today | Target |
|---|---|---|
| Edit scene | 7.5 to 8 min | ≤ 4 min, ≤ $0.15 per edit |
| Preview, cold | 2 min 40 s | ≤ 90 s |
| Preview, within 5 min of the last | 2 min 40 s | ≤ 30 s |
| Add object, within 5 min of the last | 2 min 11 s | ≤ 60 s |
