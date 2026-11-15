# Design: warm-gpu-workers

## Load once per container
- `ObjectGenerator` (L40S, `object_image`, models Volume). `@modal.enter` builds FLUX and TRELLIS and keeps them on the GPU. `generate(job_id, prompt, seed, reference_image)` keeps today's body minus the loading, the prefetch thread and `del flux`.
- `PreviewRenderer` (L4, `preview_image`). `@modal.enter` builds SDXL + depth ControlNet; `render(...)` keeps today's body minus loading.
- `RunLog` keeps a `load_*` stage only on the call that loaded, and records `warm: true/false`, so `run.json` shows whether a job hit a warm container.
- Weights still come from the `splat-studio-models` Volume; the first container after a deploy fills it as today.

### GPU memory (main risk)
FLUX.1-schnell in bf16 is about 33 GB with its text encoders; TRELLIS-image-large needs a few GB more. Both together should fit in 48 GB, but this is not yet measured. Order of fallbacks if it does not fit:
1. Keep FLUX's T5 text encoder on the CPU and move only the prompt embeddings to the GPU (T5 runs once per prompt, small cost).
2. Keep FLUX in CPU RAM (pinned) and move it to the GPU per call; slower than resident, much faster than loading from disk.
3. Move to an 80 GB GPU (A100-80GB / H100), at a higher price per hour.

## Pre-warm from the Space
- New method `warm()` on both classes: returns at once. Calling it starts a container, and `@modal.enter` loads the models.
- `modal_client.warm(kind)` spawns `warm()` and does not wait.
- Triggers in `space/app.py`: scene opened in the Scene tab → warm `object`; Add object tool selected → warm `object`; Preview tool selected → warm `preview`.
- Throttle: at most one warm-up per kind per 4 minutes per Space process (`space/warmup.py`, pure logic, unit-tested with a fake clock). Within the idle window a second warm-up adds nothing.
- Warm-up failures are logged and ignored; the real job still cold starts as today.

## Idle window
- `scaledown_window=300` on both classes, as a constant next to the GPU settings. Longer windows cut more cold starts but bill idle GPU time; 300 s covers typing a prompt and placing an object.
- `max_containers=1` on both classes: one user, and it keeps a burst of clicks from starting several GPUs.

## Not done here
- Always-on containers (`min_containers=1`): too costly while idle.
- Memory snapshots: can be added later if cold starts still hurt.
- Merging in memory instead of via the HF Hub: separate change.
