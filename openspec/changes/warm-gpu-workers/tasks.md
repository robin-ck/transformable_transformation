# Tasks: warm-gpu-workers

Runs alongside `redesign-space-gui`. Groups 0, 1 and task 2.1-2.2 can start now (no shared files). Task 2.3 and 4.2 wait until the redesign is merged: they edit `space/app.py` and the README, and need its Scene tab and tool sub-tabs.

## Status
Plan only; nothing implemented. Start with 0.1 on a machine with `HF_TOKEN` and `SPLAT_STUDIO_DATASET` (in `.env`); the cloud session that wrote this plan had neither. Group 1 also needs a Modal token to deploy and test.

## 0. Baseline
- [ ] 0.1 Run `python scripts/job_timings.py`; record cold start (`start_wait_s`) and stage times of recent object and preview jobs in this file

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
- [ ] 3.1 Deploy; cold job, then warm job: compare with the baseline (`scripts/job_timings.py`)
- [ ] 3.2 Open scene, wait 1 min, add "a mature oak tree": done in under 30 s?
- [ ] 3.3 If over 30 s: note which stage dominates and propose next step (faster GPU, fewer TRELLIS steps, smaller image)

## 4. Wrap-up
- [ ] 4.1 ruff + pytest pass
- [ ] 4.2 README: note on warm-up, idle window and its cost
