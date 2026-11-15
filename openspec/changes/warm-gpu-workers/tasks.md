# Tasks: warm-gpu-workers

Runs alongside `redesign-space-gui`. Groups 0, 1 and task 2.1-2.2 can start now (no shared files). Task 2.3 and 4.2 wait until the redesign is merged: they edit `space/app.py` and the README, and need its Scene tab and tool sub-tabs.

## Status
0.1 done (baseline below). Group 1 written and unit-tested except 1.3, which needs a deploy (Modal token). Owner of the warm object and preview classes: this change; `speed-up-edit-jobs` group 3 keeps `SceneEditor` only.

## 0. Baseline
- [x] 0.1 Run `python scripts/job_timings.py`; record cold start (`start_wait_s`) and stage times of recent object and preview jobs in this file
  - 2026-11-15. The script read only `jobs/`, where scene jobs live; fixed to read `objects/` and `previews/`, and to include direct test runs (no `job.json`, so no start wait). All runs are from 2026-11-13, before any warm container work.
  - Object, from the Space (`20261113-153134-71f6e0`, L40S, "put a plant onto the table"): start wait 11.7 s, run 124.1 s, click to finish 148.4 s. Stages: imports 50.9, load FLUX 20.4, image 3.7, load TRELLIS 23.3, gaussians 6.6, simplify 0.9, upload 17.4.
  - Object, direct runs (L40S): `oak-tree-v2` run 131.4 s (load FLUX 22.3, image 3.6, load TRELLIS 22.3, gaussians 13.2, simplify 2.5). `oak-tree` 301.9 s, of which 154.4 s was the one-time FLUX download.
  - Preview, from the Space (L4): both runs failed in gsplat (`AssertionError(torch.Size([1, 5]))`), so no full Space-side timing. The first had a start wait of 113.9 s (cold start); the second, 3 min later, 1.1 s, so the container was reused.
  - Preview, direct runs (L4): `preview-test` run 154.7 s (render 118.0, load SDXL + ControlNet 16.0, restyle 14.1, upload 5.7, imports 3.3); `preview-test-a10g` (logged as L4) run 159.5 s (render 74.8, load 62.7, restyle 10.6, upload 9.2).
  - Reading: in the object job, imports and model loading take about 95 of 124 s; the work itself (image, gaussians, simplify) is about 11 s and upload 17 s. A warm container should therefore land near 30 s, with upload the largest remaining stage. In the preview, `render` (75 to 118 s) dominates, not model loading: this is most likely the gsplat JIT compile that `speed-up-edit-jobs` task 1 removes. Warm containers alone will not bring the preview under 30 s.
  - Overlap: `speed-up-edit-jobs` group 3 (on `main`) also turns preview and object jobs into warm Modal classes. Decide which change owns that work before starting group 1.

## 1. Load once per container
- [x] 1.1 Bump `modal>=1.0`; confirm `@modal.cls`, `@modal.enter`, `scaledown_window`, `max_containers` against the installed version
  - Checked on modal 1.6.4: `App.cls` takes `scaledown_window`, `max_containers`; `modal.enter`, `modal.method`, `Cls.from_name` exist. The app loads locally and registers both classes.
- [x] 1.2 `ObjectGenerator` class: models in `@modal.enter`, `generate()` method; drop prefetch thread and `del flux`
  - If loading fails in `enter`, the container stays up and the next job loads again, so a gated-model error still reaches the user as today.
- [x] 1.3 Check GPU memory with FLUX + TRELLIS resident; apply fallbacks from design.md if needed
  - 2026-11-15, job `20261115-130233-5da883` (L40S, cold, "add a plant onto the table"): peak 39.1 of 44.4 GB usable with both resident. Fits; no fallback needed, but only about 5 GB headroom.
  - Same run: start wait 129.2 s (container start plus `enter`: imports 74.5, load FLUX 21.6, load TRELLIS 25.9), job 18.6 s (image 2.7, gaussians 8.0, simplify 0.7, upload 6.9), click to finish 149.5 s. Baseline job time was 124.1 s.
  - Note: `est_cost_usd` counts only the job, not the load in `enter` or idle time.
- [x] 1.4 `PreviewRenderer` class: SDXL + ControlNet in `@modal.enter`, `render()` method
  - Added: `enter` also renders a 16-Gaussian scene to build gsplat's kernels (stage `compile_gsplat`), since 0.1 shows the first render, not model loading, dominates the preview.
- [x] 1.5 `RunLog`: `warm` flag; load stages only on the loading call
  - `LoadTimes` in `reconstruction/cost.py`; `warm` is `None` for plain functions.
- [x] 1.6 `modal_client`: submit via class methods; unit tests with a fake client stay green
  - `spawn_method(class, method, ...)`; tests in `tests/test_reconstruction.py`. ruff + pytest: 120 passed.

## 2. Pre-warm
- [ ] 2.1 `warm()` method on both classes; `modal_client.warm(kind)`
- [ ] 2.2 `space/warmup.py`: throttle per kind (4 min); unit tests with a fake clock
- [ ] 2.3 Space events: scene opened, Add object tool, Preview tool → warm-up; failures logged only

## 3. Check
- [ ] 3.1 Deploy; cold job, then warm job: compare with the baseline (`scripts/job_timings.py`)
  - Object, 2026-11-15 (L40S, local Space): cold `20261115-130233-5da883` click to finish 149.5 s (start wait 129.2, job 18.6); warm `20261115-130551-2ae3be` ("put a pizza onto the table") 16.3 s (start wait 0.1, job 14.6: image 2.3, gaussians 7.8, simplify 1.1, upload 3.0), `warm: true`. Baseline 148.4 s. Warm meets the 30 s target; cold is unchanged, so pre-warm (group 2) matters.
  - Preview: open.
- [ ] 3.2 Open scene, wait 1 min, add "a mature oak tree": done in under 30 s?
- [ ] 3.3 If over 30 s: note which stage dominates and propose next step (faster GPU, fewer TRELLIS steps, smaller image)

## 4. Wrap-up
- [ ] 4.1 ruff + pytest pass
- [ ] 4.2 README: note on warm-up, idle window and its cost
