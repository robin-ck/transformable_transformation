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
  - 2026-11-13: full run "a mature oak tree" (job `20261113-oak-tree`, FLUX terms accepted):
    5 min 2 s, about $0.16, of which 154 s was the one-time FLUX download into the volume;
    image 4.5 s, TRELLIS load 30.5 s, Gaussians 35.1 s. Clean reference image; 1.45 M Gaussians
    (`.ply` 99 MB). Placed into the real scene locally: v2 with both model names in `version.json`.
  - Fix (same day): objects were saved lying on their side. TRELLIS's `save_ply` turns its
    z-up frame to y-up by default; now saved with `transform=None`. Checked: trunk below leaves.
  - Speed-up: TRELLIS's weight files are read into memory while FLUX loads (building both models
    in parallel threads is not safe: diffusers' init_empty_weights patches torch for the whole
    process). `editing/simplify.py` merges the object's Gaussians per grid cell (moment
    matching) down to <= 250 k. Oak: 1.45 M -> 208 k Gaussians, `.ply` 99 -> 14 MB, 2.5 s;
    side by side it looks nearly the same (leaf clumps slightly softer).
  - 2026-11-13 rerun (job `20261113-oak-tree-v2`, weights cached): 2 min 11 s in the container,
    about $0.07. FLUX load 22 s, image 4 s, TRELLIS load 22 s, Gaussians 13 s, trim 2.5 s; about
    65 s was not covered by a stage, so imports and uploads are now logged too.
  - Deployed with `modal deploy` (2026-11-13). The Space reads a local `.env` for HF_TOKEN and
    SPLAT_STUDIO_DATASET. open: not yet tried from the Space against HF Hub.
- [x] 4.3 Preview: render view, image-to-image with depth control
  - Space: "Preview" section (look prompt, change strength 0.3-0.95, seed). The button asks
    the viewer iframe for its camera (`postMessage` `get_camera`; Three.js camera-to-world,
    fov, aspect), starts a job under `previews/<job>/`, polls every 5 s and shows the
    result (downloadable) in a column next to the viewer, with the plain render under it.
    Nothing is written under `scenes/`, so the version is unchanged.
  - Modal `render_preview` (L4, 10 min cap): gsplat renders the version's `.splat` at the
    viewer's camera (Three.js -> OpenCV axes, same vertical fov, about 1 MP with sides in
    multiples of 64), colour + depth; inverse depth (near bright) is the control image for
    SDXL img2img + depth ControlNet (fp16, 30 steps, control scale 0.6). Uploads
    `render.png`, `depth.png`, `preview.png`. A view that shows < 5 % of the scene fails with
    a tip to turn the camera. Camera maths, `.splat` decoding and the depth image live in
    `splat_studio/editing/preview.py` (unit tested).
  - 2026-11-13: first run failed: gsplat's `backgrounds` argument does not fit the RGB+ED
    mode (it pads channels); the background is now blended in after rendering. The UI showed
    the failure as a readable message.
  - 2026-11-13: full run on the real scene, "a cosy carpentry workshop with warm evening
    light, wooden walls, plants" (job `20261113-preview-test-a10g`, run on an A10G via
    `with_options` because no L4 was free for over 7 min): 2 min 40 s, about $0.05.
    Render 75 s (most likely gsplat compiling its CUDA kernels on first use in a fresh
    container), model download + load 63 s (one-time download into the volume), restyle
    11 s, upload 9 s. 1344x768 output. The render matches the viewer's view exactly
    (same objects at the same places); the restyle keeps the layout and warms the light,
    but drops the "plants" part of the prompt at strength 0.7.
  - open: the 75 s render step; check on a second cold start whether gsplat's kernels are
    cached, else build them into the image. L4 availability: if it keeps queueing, move
    the default to A10G. The Space's own button click reached Modal (the camera arrived
    correctly) but the successful run was spawned from the CLI with that recorded camera.
- [x] 4.4 Edit scene: Modal job with iterative dataset update; optional region mask
  - Views: the training cameras are not stored and edited versions have no photos, so
    the job renders its own dataset: 40 views orbiting the viewer's camera (+-30 deg yaw,
    +-10 deg pitch, +-15 % distance) around the point it looks at (median depth of the view).
    The edit is therefore strongest around the user's view. `editing/scene_edit.py`
    (numpy, unit tested): cameras, region, PLY <-> gsplat arrays.
  - `editing/edit_worker.py` (Modal only): render the 40 views, edit each once with
    InstructPix2Pix (fp16, DDIM 20 steps, guidance 7.5 / image 1.5), then 1,500 Adam steps
    (L1 + 0.2 D-SSIM, gsplat's learning rates, no densification); every 10 steps one view
    is re-rendered, re-noised (t in 0.70-0.98) and re-edited, as in Instruct-NeRF2NeRF.
  - Region: the user switches on "Limit to a region" and drags a box in the viewer
    (overlay; the camera stays still). Gaussians whose centre is in the box and not more
    than 20 % behind the first dense surface of its 16 px screen cell (15th percentile of
    centre depths) are selected; only they get gradients, and each edited target keeps the
    original render outside the selection's footprint (coverage-normalised soft mask).
    The orbit pivot is the median centre depth in view. Both use centres, not a depth
    render, because haze in front of surfaces pulls rendered depth towards the camera.
  - Modal `edit_scene` (A10G, 30 min cap) saves the result itself as the next version
    (`version.json`: prompt, model, parent, region, camera, seed, steps), so it survives a
    closed page; `edits/<job>/` keeps before/after renders, one edited view and the scene.
    Stage timings are published while it runs, for the Space's progress line.
  - Space: "Edit scene" section (instruction, region switch, seed), polled every 10 s; when
    done the version list and viewer switch to the new version. Running and failed edits
    appear in the prompt log.
  - 2026-11-13 runs on the real scene, "make the wooden table bright green", A10G:
    - Whole view (job `20261113-162643-6273b5`): 8 min 12 s, about $0.15. Download 40 s,
      render views 89 s, load editor 14 s, first edits 62 s, training 278 s. Training
      reproduced the edit consistently, but InstructPix2Pix tinted the whole room green,
      not just the table. Saving the version failed: the image's `huggingface_hub` 0.36
      has no top-level `RepoFolder`; fixed (import from `hf_api`), checked in the image,
      and the scene now goes to `edits/<job>/` before the version is written.
    - Region around the table (job `20261113-162648-52f1a8`): same timings and failure;
      only a faint tint, because the depth test was too strict (5 %) and the blend mask
      was opacity-weighted. Now 20 % and the mask is normalised by coverage.
    - Rerun with the save fix (job `20261113-164258-d27726`): saved as v2, but only 155
      Gaussians selected and pivot 0.74: the rendered depth sat in the haze (0.7) while
      the table is at 1.2-2.0. Switched surface and pivot to centre depths (see above;
      checked locally on the real scene: 413 k selected, table and floor, not the wall).
    - Rerun (job `20261113-165311-5ec895`): 7 min 33 s, about $0.14 (render views 71 s,
      first edits 63 s, training 281 s, upload 25 s). 413 k Gaussians selected, pivot 1.83.
      Saved as v3 (parent v1): the table is clearly green, the rest of the room unchanged;
      some green spills onto the floor next to the table, inside the box.
  - open: InstructPix2Pix tints the whole view for object-level instructions without a
    region, so the region is needed for local edits. `render_views` takes 70-90 s in
    every fresh container (same as the preview's render: probably gsplat compiling its
    CUDA kernels; build them into the image). The runs were started from a script with the
    camera recorded in the Space; the Space's own Edit button was checked up to the
    request (camera and box arrive), not through a full run. Two edits finishing at the
    same moment could pick the same version number. v2 (155-Gaussian edit) stays in
    the dataset as a nearly unchanged version.
- [x] 4.5 Prompt log per version
  - `splat_studio/space/prompt_log.py` reads what is already stored: each version's
    `version.json` (prompt, mode, model, parent, date) and each preview's `job.json`.
    Nothing new is written. Preview jobs record their planned model at creation, so failed
    previews still show it.
  - Space: under the version list, "History of vN" lists the chain of edits that led to the
    shown version (follows `parent`), with prompt and model of each step. A "Prompt log"
    panel lists every prompt for the scene, newest first: versions and previews (with source
    version and failed/running state). It is refreshed after picking a scene or version,
    placing an object, and when a preview starts or finishes.
  - 2026-11-13: checked locally against HF Hub: v1 history and the two failed preview
    attempts from 4.3 show with prompt and source version.

## 5. Wrap-up
- [ ] 5.1 End-to-end test: upload → scene → add object → edit → preview
- [ ] 5.2 README usage guide with screenshots
- [ ] 5.3 Move the object merge from the Space to Modal, so the UI server stays light
  - Why: `place_object` (`space/add_object.py`) runs `merge_files` (`editing/objects.py`) in the
    Space process. It downloads the full scene PLY (367 MB for the test scene), loads it into
    numpy and uploads the result: 1–1.5 GB peak RAM. On the planned Hetzner server (2 vCPU,
    4 GB shared with other services, see `docs/handoff-hetzner-deploy.md`) that can crowd out
    other services or get OOM-killed.
  - Do: a CPU-only Modal function (no GPU) that reads the scene and object PLY from the HF
    dataset, applies the `Placement`, merges, writes `scene.ply` + `scene.splat` as the next
    version, and returns the version info. The Space only sends job id, scene id, source
    version and placement, then polls like the other jobs.
  - Keep: the version and prompt-log records stay as they are (`version.json` with prompt,
    mode, model, parent). The placement step in the viewer is unchanged.
  - Done when: adding an object never loads a full scene PLY on the Space (peak RSS of
    `space/app.py` stays under ~500 MB during a merge), unit tests cover the new job with
    mocked Modal calls, and one real merge on the test scene is recorded here with time and
    Modal cost.
