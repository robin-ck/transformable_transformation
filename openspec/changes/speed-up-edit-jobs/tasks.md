# Tasks: speed-up-edit-jobs

Order: 1 first. It needs no quality decision, and it makes every later measurement fair.
Steps 2 to 5 each end with a measured run, recorded under the task as in MVP tasks.md.
This change can run in parallel with `redesign-space-gui`, which works in `space/app.py` and the viewer.
MVP task 5.3 (object merge on Modal) should use `scene_cache` once task 2 is done.

## 1. gsplat kernels in the image
- [ ] 1.1 Add a GPU probe `check_gsplat` (T4 or L4, a few cents): in a fresh container, time `import gsplat` and the first `rasterization` call on a small random scene. Run it on today's image and record the baseline. A first call near 70 s confirms the JIT compile.
- [ ] 1.2 Build gsplat from `/opt/gsplat` with its CUDA kernels:
  - `git clone --recursive`
  - `pip install --no-build-isolation`
  - `TORCH_CUDA_ARCH_LIST="8.6;8.9"`
  - drop the wheel-index install
  - end the build step with a `from gsplat import csrc` check
- [ ] 1.3 Rerun `check_gsplat`: the first call takes < 5 s. Run `check_edit_image` too, and a short `reconstruct` on the test job (500 steps) to confirm training still runs.
- [ ] 1.4 Run one preview on the real scene and record the stage times.

## 2. Scene file cache on the volume
- [ ] 2.1 `src/splat_studio/scene_cache.py`:
  - `fetch(repo_path, download)` with an atomic rename and a `last_used` stamp
  - `put(repo_path, local_file)`
  - an LRU trim to `CACHE_CAP_GB`
  - unit tests with a fake downloader in `tests/test_scene_cache.py`
- [ ] 2.2 Use it in Edit scene (`scene.ply`) and Preview (`scene.splat`). After `add_version` succeeds, Edit scene puts its result into the cache under the new version. Commit the volume after writes.
- [ ] 2.3 Add `hf_transfer` to the preview/edit and object images (`HF_HUB_ENABLE_HF_TRANSFER=1`). Keep it only if download or upload time drops in 2.4.
- [ ] 2.4 Measure an edit on v3, then an edit on the version it produced. The second job shows a cache hit and no download stage time.

## 3. Warm containers
- [ ] 3.1 Turn `render_preview` into `Previewer`:
  - `@app.cls`, `scaledown_window=WARM_S` (300 s)
  - `@modal.enter` loads SDXL and the ControlNet
  - `run` keeps today's arguments and outputs
  - `models_volume.reload()` at the start of each job
- [ ] 3.2 Turn `edit_scene` into `SceneEditor`. Split `Editor` into a load step (pipeline, scheduler) and a per-job prompt encoding.
- [ ] 3.3 Turn `generate_object` into `ObjectMaker`. Measure GPU memory with FLUX and TRELLIS both loaded. If they do not fit, keep both in CPU RAM (`memory` 64 GB) and move each to the GPU for its step. Load them one after the other in `enter`.
- [ ] 3.4 `modal_client`:
  - add a class form of `spawn`
  - `submit_preview`, `submit_edit` and `submit_object` call the class methods
  - `poll` stays unchanged
  - unit tests with mocked `modal` in `tests/test_modal_client.py`
- [ ] 3.5 Measure two previews back to back and two objects back to back. The second run of each shows no model load stage. Record cold and warm times.

## 4. Edit scene budget
- [ ] 4.1 Add an optional `budget` dict to `edit_scene` and `SceneEditor.run` that overrides `EditSettings` fields (views, steps, edit_every, denoise_steps). Record the values used in `version.json`. Unit test the override and validation in `tests/test_scene_edit.py`.
- [ ] 4.2 Batch the first edits 4 views at a time. Keep it only if it is at least 20 % faster in run A.
- [ ] 4.3 Run comparison runs A to D (design.md, same camera, region and seed as job `20261113-165311-5ec895`). Save `after.png`, `edit_example.png` and three orbit renders per run, and record the time and cost of each.
- [ ] 4.4 Pick the cheapest run that keeps the reference edit's look and make it the default `EditSettings`. Note the comparison in design.md.

## 5. GPU choice
- [ ] 5.1 Run the chosen Edit scene settings on an L40S (`with_options(gpu="L40S")`) and compare time and cost with the A10G.
- [ ] 5.2 Switch `EDIT_GPU` only if the L40S is at least 1.5 times faster. Keep the timeouts in `space/jobs.py` in sync.
- [ ] 5.3 Preview GPU: if the L4 queued for more than 1 min in 3.5, move `PREVIEW_GPU` to A10G.

## 6. Wrap-up
- [ ] 6.1 ruff + pytest pass
- [ ] 6.2 Deploy Modal, then the Space. Check one edit, one preview and one object from the Space's own buttons.
- [ ] 6.3 README, GPU jobs section:
  - warm window and idle cost per GPU type
  - the scene cache and its cap
  - the deploy order
- [ ] 6.4 Record final times against the targets in design.md: edit ≤ 4 min, cold preview ≤ 90 s, warm preview ≤ 30 s, warm object ≤ 60 s
- [ ] 6.5 MVP tasks.md:
  - under 4.3 and 4.4, mark the "render step" open items as fixed by this change
  - under 5.3, note that the Modal merge should use `scene_cache`

## 7. Later: object addition
- [ ] 7.1 Something more for the object (Add object) is to be added later. The scope is not decided yet. Write it up as its own tasks, or as a separate change if it changes behaviour, before starting.
