# Proposal: warm-gpu-workers

## Why
Adding an object or making a preview takes far longer than the 30 s target. Each job starts a new Modal container, which waits for a GPU, starts the image and loads the model weights (FLUX.1-schnell alone is about 24 GB) before any work begins. Model loading happens inside the function body, so even a reused container would reload everything. Scene creation (COLMAP + training) is allowed to stay slow and is out of scope.

## What changes
- **Load once per container (option 1):** `generate_object` and `render_preview` become methods of Modal classes (`ObjectGenerator`, `PreviewRenderer`). Weights load in `@modal.enter`, so a warm container answers without loading.
- **Pre-warm from the Space (option 3):** when the user opens a scene in the Scene tab, or switches to the Add object or Preview tool, the Space starts a container in the background with a no-op `warm()` call. The cold start then overlaps with the user looking around and typing the prompt.
- **Short idle window:** containers stay alive for a set time after their last call (`scaledown_window`, default 5 min), so the pre-warmed container is still there when the request comes and between edits in one session.
- **Timing check first:** before and after the change, read the stage times in `run.json` to see whether a warm job fits in 30 s.

## Impact
- Specs: `text-editing` gains a requirement on response time for Add object and Preview.
- Files: `modal_app/app.py` (classes, enter hooks, `warm()`), `src/splat_studio/modal_client.py` (call methods on classes, `warm()`), `src/splat_studio/space/jobs.py` (no change to job records), `space/app.py` (warm-up events), `pyproject.toml` and `space/requirements.txt` (`modal>=1.0` for `scaledown_window`).
- Cost: idle GPU time of up to one idle window per session and per warm-up that is never used. With the default 5 min on an L40S, that is a few tens of cents per session (check current Modal prices).
- Risks: FLUX and TRELLIS may not both fit in the L40S's 48 GB at once (see design); a warm container still may not reach 30 s if TRELLIS itself is slow; the Space must not send warm-ups on every click (throttled).
- No breaking changes: stored jobs, objects and versions keep their format.
