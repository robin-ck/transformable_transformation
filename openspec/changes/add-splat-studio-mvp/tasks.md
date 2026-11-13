# Tasks: add-splat-studio-mvp

## 1. Repo
- [x] 1.1 Create private GitHub repo; add README, licence, `.gitignore`, `pyproject.toml`
- [x] 1.2 GitHub Actions: ruff + pytest on push
- [x] 1.3 Document secrets setup (Modal token, HF token) in README

## 2. Reconstruction (Modal)
- [x] 2.1 Spike: Modal GPU function called from a Space, with a cost/runtime log per run
  - 2026-11-13: `gpu_probe` ran on a Tesla T4 from the local CLI (Space-side call comes with task 3).
    In-container work 0.1 s; end-to-end with cold start 14-17 s, so about $0.003 per probe at
    $0.000164/s. First image build (gsplat + CUDA extensions) took about 2 min; later runs reuse the cache.
  - Image build fixes: `--no-build-isolation` for fused-ssim/fused-bilagrid, `CC=gcc CXX=g++`,
    `TORCH_CUDA_ARCH_LIST=8.6`. The gsplat 1.5.3 pin installs and builds.
- [x] 2.2 Modal function: frame extraction, COLMAP poses, `gsplat` training, `.ply` + compressed export
- [x] 2.3 Readable failure messages for pose failure
- [x] 2.4 Test on 2 sample videos and 1 photo set; record time per run
  - Scope reduced to one successful video run (decided 2026-11-13); a photo-set run is still
    untested and can be done as part of 5.1.
  - 2026-11-13: job `20261113-132210-3749ae`, 1 video, 180 frames, A10G, 7,000 steps:
    frames 10 s, poses 503 s, training 337 s, export 1 s; 14 min 26 s in total, about $0.31.
    1.55 M Gaussians (`.ply` 367 MB, `.splat` 50 MB). COLMAP poses (CPU) take the most time.
  - Earlier, the first two runs failed as `pose_failed`. Cause on our side: `sparse/` was never
    created, so `colmap mapper` exited with an error, and every mapper error was reported as bad
    footage. Fixed; COLMAP crashes now report `pose_tool_failed` and the log tail goes to `run.json`.

## 3. Space and viewer
- [x] 3.1 Gradio Space with upload + validation (unit tests for limits)
  - `space/app.py` + `splat_studio/space/validation.py`. Smoke-tested locally with Gradio 5.50
    (8 photos rejected, 20 accepted). Not yet deployed to a Space.
- [x] 3.2 Job client: spawn Modal job, poll, store results on HF Hub (unit tests with mocked Modal calls)
  - `splat_studio/space/jobs.py`: job state in `jobs/<id>/job.json`, queued/running from the
    worker's `run.json`, Modal function timeout mapped to the timeout message.
  - open: not yet run against real Modal + HF Hub (do together with 2.4)
- [x] 3.3 Embed JS splat viewer; orbit/pan/zoom/reset
  - `splat_studio/space/viewer.py`: `@mkkellogg/gaussian-splats-3d` 0.4.7 (MIT, Three.js 0.170)
    from jsDelivr, in an `<iframe srcdoc>`; the `.splat` is served via `/gradio_api/file=` from the
    store's download folder (`allowed_paths`). Start camera from the median and 80th-percentile
    radius of the Gaussians, z-up (gsplat normalises the world to z-up). Reset button + R key.
  - 2026-11-13: checked locally with the real 1.55 M-Gaussian scene from job 20261113-132210-3749ae:
    loads, orbit works, reset returns to the exact start camera. Scene opens upright. The start view
    is from outside the scene, so for room captures the walls are in front; zoom in to enter.
  - open: touch navigation not tried on a real phone; not yet tried on a deployed Space.
- [ ] 3.4 Versions list, switch, download

## 4. Text editing
- [ ] 4.1 Licence check and pick text-to-3D, image-edit and preview models
- [ ] 4.2 Add object: generate, convert to Gaussians, place/scale/rotate, merge
- [ ] 4.3 Preview: render view, image-to-image with depth control
- [ ] 4.4 Edit scene: Modal job with iterative dataset update; optional region mask
- [ ] 4.5 Prompt log per version

## 5. Wrap-up
- [ ] 5.1 End-to-end test: upload → scene → add object → edit → preview
- [ ] 5.2 README usage guide with screenshots
