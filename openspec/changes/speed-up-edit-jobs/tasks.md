# Tasks: speed-up-edit-jobs

Order: 1 first. It needs no quality decision, and it makes every later measurement fair.
Steps 2 to 5 each end with a measured run, recorded under the task as in MVP tasks.md.
This change can run in parallel with `redesign-space-gui`, which works in `space/app.py` and the viewer.
MVP task 5.3 (object merge on Modal) should use `scene_cache` once task 2 is done.

## Status
2026-11-15: 1.2, 2.1, 2.2, 3.2, 3.4 and 4.1 written and unit-tested (ruff + pytest pass); nothing deployed or measured yet, so every "measure" task (1.1, 1.3, 1.4, 2.4, 3.5, 4.3, 5.x) is still open.
- 1.2: the build check is `from gsplat import csrc` plus the version; `MAX_JOBS=4` keeps the nvcc build within the builder's memory. Preview's `enter` now only warms the prebuilt kernels (stage `warm_gsplat`, was `compile_gsplat`).
- 2.2: downloads go through a temporary HF cache, so scenes no longer also pile up in the volume's `HF_HOME`. Each job calls `models_volume.reload()` first and skips it with a log line if the volume refuses (open files).
- 2.3 (`hf_transfer`) not done: the dataset likely transfers through `hf_xet`, where it changes nothing. Revisit only if 2.4 shows slow downloads.
- 4.1: instead of a CLI-only override, the budget is in the Space too: an Edit tab preset (Fast = run D, Balanced = run C, Best = today's settings), four sliders, and a time estimate from `scene_edit.estimate_edit_breakdown` (fitted to the baseline stage times). Balanced is the default until 4.3/4.4 pick one. `version.json` records the four values and the preset.
- First measured run, 2026-11-15, job `20261115-210709-5bd8b4` (A10G, Balanced, cold container, scene not cached, "make seats green" on v2, no region): job 190 s, est. $0.057. Stages: imports 2.0, warm_gsplat 1.7, load_editor 11.3, download_scene 7.6, load_scene 0.5, render_views 0.5, encode_prompt 0.2, first_edits 38.7, train 93.4, export 1.1, upload 33.1. Earlier edits on the old code (`20261115-194302-b9a225`, `-195421-146424`, old settings): 505 s / $0.155 and 442 s / $0.136, render_views 68 to 72 s, train 267 to 325 s. Confirms 1.2 (render_views 0.5 s). The Space's estimate for this run was about 4.5 min, so `scene_edit` time constants are too high; recalibrate after a warm, cached run. Quality not yet compared.
- 4.2 (batched first edits) not done: worth it only if measured, and first edits are a small share once re-edits are cut.

## 1. gsplat kernels in the image
- [ ] 1.1 Add a GPU probe `check_gsplat` (T4 or L4, a few cents): in a fresh container, time `import gsplat` and the first `rasterization` call on a small random scene. Run it on today's image and record the baseline. A first call near 70 s confirms the JIT compile.
- [x] 1.2 Build gsplat from `/opt/gsplat` with its CUDA kernels:
  - `git clone --recursive`
  - `pip install --no-build-isolation`
  - `TORCH_CUDA_ARCH_LIST="8.6;8.9"`
  - drop the wheel-index install
  - end the build step with a `from gsplat import csrc` check
- [ ] 1.3 Rerun `check_gsplat`: the first call takes < 5 s. Run `check_edit_image` too, and a short `reconstruct` on the test job (500 steps) to confirm training still runs.
- [ ] 1.4 Run one preview on the real scene and record the stage times.

## 2. Scene file cache on the volume
- [x] 2.1 `src/splat_studio/scene_cache.py`:
  - `fetch(repo_path, download)` with an atomic rename and a `last_used` stamp
  - `put(repo_path, local_file)`
  - an LRU trim to `CACHE_CAP_GB`
  - unit tests with a fake downloader in `tests/test_scene_cache.py`
- [x] 2.2 Use it in Edit scene (`scene.ply`) and Preview (`scene.splat`). After `add_version` succeeds, Edit scene puts its result into the cache under the new version. Commit the volume after writes.
- [ ] 2.3 Add `hf_transfer` to the preview/edit and object images (`HF_HUB_ENABLE_HF_TRANSFER=1`). Keep it only if download or upload time drops in 2.4.
- [ ] 2.4 Measure an edit on v3, then an edit on the version it produced. The second job shows a cache hit and no download stage time.

## 3. Warm containers
Decided 2026-11-15: `warm-gpu-workers` owns the object and preview classes (`ObjectGenerator`, `PreviewRenderer`, done there in group 1). Here only `SceneEditor` (3.2) and its part of 3.4 remain.
- [ ] 3.1 (moved to `warm-gpu-workers` 1.4) Turn `render_preview` into `Previewer`:
  - `@app.cls`, `scaledown_window=WARM_S` (300 s)
  - `@modal.enter` loads SDXL and the ControlNet
  - `run` keeps today's arguments and outputs
  - `models_volume.reload()` at the start of each job
- [x] 3.2 Turn `edit_scene` into `SceneEditor`. Split `Editor` into a load step (pipeline, scheduler) and a per-job prompt encoding.
- [ ] 3.3 (moved to `warm-gpu-workers` 1.2, 1.3) Turn `generate_object` into `ObjectMaker`. Measure GPU memory with FLUX and TRELLIS both loaded. If they do not fit, keep both in CPU RAM (`memory` 64 GB) and move each to the GPU for its step. Load them one after the other in `enter`.
- [x] 3.4 `modal_client`:
  - add a class form of `spawn`
  - `submit_preview`, `submit_edit` and `submit_object` call the class methods
  - `poll` stays unchanged
  - unit tests with mocked `modal` in `tests/test_modal_client.py`
- [ ] 3.5 Measure two previews back to back and two objects back to back. The second run of each shows no model load stage. Record cold and warm times.

## 4. Edit scene budget
- [x] 4.1 Add an optional `budget` dict to `edit_scene` and `SceneEditor.run` that overrides `EditSettings` fields (views, steps, edit_every, denoise_steps). Record the values used in `version.json`. Unit test the override and validation in `tests/test_scene_edit.py`.
- [ ] 4.2 Batch the first edits 4 views at a time. Keep it only if it is at least 20 % faster in run A.
- [ ] 4.3 Run comparison runs A to D (design.md, same camera, region and seed as job `20261113-165311-5ec895`). Save `after.png`, `edit_example.png` and three orbit renders per run, and record the time and cost of each.
- [ ] 4.4 Pick the cheapest run that keeps the reference edit's look and make it the default `EditSettings`. Note the comparison in design.md.

## 5. GPU choice
- [ ] 5.1 Run the chosen Edit scene settings on an L40S (`with_options(gpu="L40S")`) and compare time and cost with the A10G.
- [ ] 5.2 Switch `EDIT_GPU` only if the L40S is at least 1.5 times faster. Keep the timeouts in `space/jobs.py` in sync.
- [ ] 5.3 Preview GPU: if the L4 queued for more than 1 min in 3.5, move `PREVIEW_GPU` to A10G.
  - 2026-11-16: handled in `reopen-previews` (2.1, 2.2) as a fallback list instead: `PREVIEW_GPUS = ["L4", "A10G", "L40S"]`. Previews from the Space waited 4 to 41 min for an L4 (start waits 227 to 2487 s on 2026-11-15 and 16). Deployed as v16 on 2026-11-16; the first preview after it (cold, L4) took 37.8 s from click to finish, `warm_gsplat` 1.0 s instead of `compile_gsplat` 120 s.

## 6. Wrap-up
- [x] 6.1 ruff + pytest pass (152 passed after rebasing on main, 2026-11-15)
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
