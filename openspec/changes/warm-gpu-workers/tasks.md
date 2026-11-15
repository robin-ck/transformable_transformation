# Tasks: warm-gpu-workers

## 0. Baseline
- [ ] 0.1 Read `run.json` of recent object and preview jobs; note time per stage (cold start, load, image, gaussians, upload)

## 1. Load once per container
- [ ] 1.1 Bump `modal>=1.0`; confirm `@modal.cls`, `@modal.enter`, `scaledown_window`, `max_containers` against the installed version
- [ ] 1.2 `ObjectGenerator` class: models in `@modal.enter`, `generate()` method; drop prefetch thread and `del flux`
- [ ] 1.3 Check GPU memory with FLUX + TRELLIS resident; apply fallbacks from design.md if needed
- [ ] 1.4 `PreviewRenderer` class: SDXL + ControlNet in `@modal.enter`, `render()` method
- [ ] 1.5 `RunLog`: `warm` flag; load stages only on the loading call
- [ ] 1.6 `modal_client`: submit via class methods; unit tests with a fake client stay green

## 2. Pre-warm
- [ ] 2.1 `warm()` method on both classes; `modal_client.warm(kind)`
- [ ] 2.2 `space/warmup.py`: throttle per kind (4 min); unit tests with a fake clock
- [ ] 2.3 Space events: scene opened, Add object tool, Preview tool → warm-up; failures logged only

## 3. Check
- [ ] 3.1 Deploy; cold job, then warm job: compare `run.json` with the baseline
- [ ] 3.2 Open scene, wait 1 min, add "a mature oak tree": done in under 30 s?
- [ ] 3.3 If over 30 s: note which stage dominates and propose next step (faster GPU, fewer TRELLIS steps, smaller image)

## 4. Wrap-up
- [ ] 4.1 ruff + pytest pass
- [ ] 4.2 README: note on warm-up, idle window and its cost
