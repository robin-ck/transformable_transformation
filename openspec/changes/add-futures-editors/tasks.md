# Tasks: add-futures-editors

## 1. Registry
- [x] 1.1 `models.py`: `ModelSpec`s for Qwen-Image-Edit-2511, Qwen Lightning LoRAs, FLUX.2 klein 4B, Step1X-Edit v1.2, union ControlNet and SDXL-Lightning, with the pinned commits from design.md; `EditorSpec.kind`; four `EDITORS` entries; all in `ALL`; tests in `tests/test_models.py` (licences allowed, keys stable, `kind` and `can_mask` consistent, default unchanged)
  - `EditorSpec.kind` (`RESTYLE`/`INSTRUCT`), `short`/`tag` for tiles. `tests/test_models.py`: 3 new tests.
- [x] 1.2 `restyle.py`: `QwenEditEditor`, `Flux2KleinEditor`, `Step1XEditor`, `SdxlUnionEditor` (lazy imports; same `restyle` signature; LoRA adapter per step count); `canny_control(render)`; `make_editor` for every key; CPU tests without weights
  - `canny_control` (opencv), `lightning_adapter`, `multiple_of`, `_Base.last_info`. Qwen Lightning scheduler from ModelTC/Qwen-Image-Lightning's `generate_with_diffusers.py`. Canny test needs opencv (skipped without; passed with it).

## 2. Modal
- [x] 2.1 Images `editors_image`, `step1x_image`, `sdxl_union_image`; first check that each pipeline imports and loads on torch 2.4.1 (`modal run` of a small check function); if not, a gsplat build on a newer torch for that image (design)
  - Torch 2.4.1 was too old for Qwen, klein and the Step1X fork: diffusers' custom ops fail with `infer_schema(func): Parameter q has unsupported type`, also with diffusers 0.37.0. They now run on `gsplat_image_t26` (gsplat built against torch 2.6.0); SDXL union stays on torch 2.4.1 with diffusers 0.35.1. diffusers 0.39.0 (the newest that still accepts huggingface_hub < 1). The Step1X fork also needs `megfile` and `qwen-vl-utils`. `check_*_image` functions import each pipeline; all four load on their GPUs (5.1).
- [x] 2.2 `_ViewRestyler` subclasses `QwenEditRenderer`, `Flux2KleinRenderer`, `Step1XRenderer`, `SdxlUnionRenderer` with the GPUs from design.md; switch on `spec.kind` instead of the `sdxl-depth` key; store Step1X's rewritten prompt; extend `tests/test_preview_worker.py`
  - `_restyler_cls` for the shared options; run info gets the editor's `last_info`. `tests/test_preview_worker.py`: 4 new tests (3 instruction classes mask and record `editor_prompt_rewritten`, union refuses a mask).

## 3. Space
- [x] 3.1 `futures.py`: messages and the sent-text preview by `kind` ("instruction editors"); Lightning note instead of a negative prompt for `sdxl-union`; `steps_for` for the new keys; tests
  - `instruct_names`, `editor_tag`, `try_editors`; details show Step1X's rewritten prompt. 2 new tests.
- [x] 3.2 `jobs.py` and `modal_client.py`: submit to the editor's Modal class for every key; "Try editors" starts one job per editor and skips restyle editors when an object or area is given; tests
  - No change needed in `jobs.py`/`modal_client.py`: they already route by `modal_class`. Try editors is one `create_preview` per editor in `on_make_future`; the running-future state holds comma-separated job ids.
- [x] 3.3 `space/app.py`: editor picker lists all six; "Try editors" checkbox group next to Make future; editor name on gallery tiles and in Compare captions; Kontext-only labels become "instruction editors"
  - Checked in the local Space that the Futures sub-tab builds and shows the six editors; not clicked through with a scene (needs the deployed classes).
- [x] 3.4 `instructions.py` descriptions: "instruction editors" instead of "Kontext"

## 4. Check
- [x] 4.1 ruff check, ruff format --check, pytest
  - ruff clean; 286 passed, 1 skipped. `test_scene_cache.py::test_rejects_paths_outside_the_cache[/etc/passwd]` fails on Windows on unchanged HEAD too.
- [x] 4.2 README Futures bullet (editors, Try editors, GPUs and licences)

## 5. Deploy
- [x] 5.1 `modal deploy modal_app/app.py`; `warm` each new class once, so the weights land on the models volume; record cold-start and load times
  - Deployed 2026-11-18. Cold start including the first weight download: SDXL union 47 s, klein 90 s, Step1X 192 s, Qwen 296 s.

## 6. Trial
- [x] 6.1 On one test scene with 2 saved views, make futures with all six editors (same seed) for: a whole-view restyle ("the same street in 2050 with trees and no cars"), an object change ("make the chairs red", object "chairs"), a removal ("remove the bench", object "bench") and an area addition ("a weeping willow", painted area); Fast and Quality
  - On the kitchen scene (no street scene exists yet): restyle, recolour stools, remove the green bowl, add an olive tree in a painted area. 28 futures, none failed.
- [x] 6.2 `trial.md` in this change: a table of time (warm and cold), GPU, cost per future and a short verdict per editor and task, with links to the futures in the dataset; tune step counts if needed
  - `trial.md` with contact sheets in `trial/`. Step counts not changed; SDXL union needs a higher strength (follow-up).
- [ ] 6.3 Proposal for the follow-up change (new default, editors to drop) based on the trial
  - The verdict and follow-ups are in `trial.md` (Qwen as the default); the follow-up proposal itself is not written yet.
