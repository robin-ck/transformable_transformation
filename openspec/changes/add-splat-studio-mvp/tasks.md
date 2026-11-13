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
- [x] 3.4 Versions list, switch, download
  - `splat_studio/space/versions.py`: each version is an immutable folder `scenes/<id>/v<n>/`
    with `version.json` (date, mode, prompt, model, parent); `add_version` always appends, so
    switching back never loses later versions. v1 from the worker has no `version.json` and
    falls back to the job's finish time. Downloads are hard-linked as `<id>-v<n>.ply/.splat`.
  - Space: version radio list (newest first, shows date + prompt), viewer follows the selected
    version, "Download .ply" / "Download .splat" buttons fill a file link.
  - 2026-11-13: checked locally with the real scene plus a temporary fake v2: list, switch and
    `.splat` download (47.5 MB) work. `.ply` download only covered by unit tests (the local copy
    has no `.ply`); not yet tried against HF Hub, where the `.ply` (367 MB) is fetched on click.

## 4. Text editing
- [x] 4.1 Licence check and pick text-to-3D, image-edit and preview models
  - 2026-11-13: licences read from the HF model cards; choice and rejected models in design.md
    (Models). Add object: FLUX.1-schnell image, then TRELLIS-image-large to Gaussians
    (TRELLIS-text-xlarge as fallback). Edit scene: InstructPix2Pix. Preview: SDXL + depth ControlNet.
  - `splat_studio/models.py` pins repo ids and commits; a test keeps licences in the allow list.
  - Hunyuan3D rejected: its licence excludes the EU. No model has been run yet (comes with 4.2-4.4).
- [x] 4.2 Add object: generate, convert to Gaussians, place/scale/rotate, merge
  - Modal `generate_object` (L40S, 15 min cap): FLUX.1-schnell reference image, then
    TRELLIS-image-large straight to Gaussians (no mesh step), stored normalised (base centre at
    the origin, largest side 1) in `objects/<job>/` with `reference.png` and `run.json`.
  - `splat_studio/editing/objects.py`: placement p' = size * Rz(turn) * p + (x, y, z), merge into
    the scene's PLY layout; `space/add_object.py` writes the result as a new version with prompt,
    model, object job and placement in `version.json`. Cancel writes nothing.
  - Space: "Add object" section; the job is polled every 5 s; when done the viewer shows the
    scene plus the object, and the sliders (move x/y, height, turn, size) move it live through
    `postMessage`, without reloading the scene.
  - 2026-11-13: TRELLIS step run on Modal with TRELLIS's own mailbox example image (job
    `20261113-trellis-test`): 2 min 17 s, about $0.07 (first run: 52 s to download weights,
    10 s for the Gaussians); 918 k Gaussians. Placed into the real scene locally (Modal call
    replaced by that object): live preview, sliders, merge and cancel work; the merged object sits
    exactly at the slider values.
  - open: the text-to-image step is not run yet: FLUX.1-schnell is gated and the HF account
    behind the Modal secret has not accepted its terms. Not yet deployed (`modal deploy`) or tried
    on the Space with HF Hub.
  - open: objects are large (0.9 M Gaussians, 60 MB `.ply`), and they shift the viewer's start
    camera towards the object. Pruning or capping object Gaussians may be worth doing.
- [ ] 4.3 Preview: render view, image-to-image with depth control
- [ ] 4.4 Edit scene: Modal job with iterative dataset update; optional region mask
- [ ] 4.5 Prompt log per version

## 5. Wrap-up
- [ ] 5.1 End-to-end test: upload → scene → add object → edit → preview
- [ ] 5.2 README usage guide with screenshots
