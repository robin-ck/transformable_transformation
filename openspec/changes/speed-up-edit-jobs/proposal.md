# Proposal: speed-up-edit-jobs

## Why
Edits are too slow for trying out ideas. On the test scene (1.55 M Gaussians), Edit scene takes 7.5 to 8 min, Preview 2.7 min and Add object 2 to 5 min (MVP tasks.md, 4.2 to 4.4). Most of that is overhead. Every fresh container compiles gsplat's CUDA kernels before its first render (70 to 90 s), because prebuilt gsplat 1.5.3 wheels exist only for Python 3.10 and the image runs 3.11. Every job also starts a new container and loads its models again. On top of that, Edit scene spends about 260 s (estimated) in InstructPix2Pix: 40 first edits plus 150 re-edits.

## What changes
- Build gsplat's CUDA kernels into the Modal image, so no job compiles them.
- Keep Preview, Edit scene and Add object containers warm for a few minutes after each job, with their models loaded once per container. The Space starts these jobs through the new class methods.
- Cache scene files on the Modal volume, since versions never change. Edit scene also writes its result there, so an edit on a freshly edited version skips the 367 MB download. Try `hf_transfer` for the remaining transfers.
- Cut InstructPix2Pix work in Edit scene (fewer denoising steps, fewer re-edits, fewer views). Pick the settings with side-by-side runs of the reference edit.
- Measure Edit scene on an L40S, and switch to it if the cost per edit stays close to the A10G's.

## Impact
- Specs: adds "Job turnaround" to `text-editing`, building on the MVP delta (not yet archived).
- Files: `modal_app/app.py` (image, job classes, GPU choice), `src/splat_studio/modal_client.py` (class methods), `src/splat_studio/editing/edit_worker.py` (settings, batching), new `src/splat_studio/scene_cache.py`, README.
- Cost: a warm container bills idle GPU time for up to 5 min after each job. That is about $0.07 on an L4, $0.09 on an A10G and $0.16 on an L40S (approximate). A burst of jobs gets cheaper, while a single job costs a bit more.
- Risks:
  - Fewer InstructPix2Pix steps may weaken or blur edits, which is why the settings are chosen by comparison runs.
  - Building gsplat adds several minutes to one image build.
  - FLUX and TRELLIS together may not fit in an L40S's GPU memory for warm object jobs. The design has a fallback.
- Deploy order: deploy Modal first, then the Space, because the job names change.
- Not in scope:
  - Object merge on Modal (MVP task 5.3). It should use the new scene cache.
  - Reconstruction speed (COLMAP poses), though it gains from the gsplat build too.
  - The Space's polling intervals.
- No breaking changes: stored scenes, versions and job records keep their format.
