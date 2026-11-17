# Tasks: add-object-masking

## 1. Models and errors
- [x] 1.1 `models.py`: `MASK_DETECTOR` (grounding-dino-base) and `MASK_SEGMENTER` (sam-vit-large) with pinned commits and Apache-2.0; `EditorSpec.mask_models` (Kontext: both; SDXL: none); in `ALL`; tests
  - `EditorSpec.can_mask`; the mask models stay out of `label`, so Kontext's stored label is unchanged (test). Commits from `huggingface.co/api/models/<repo>` on 2026-11-17.
- [x] 1.2 `errors.object_not_found(query)`; test
  - Message: 'Nothing matching "chairs" was found in this view, so nothing was changed.' plus a tip.

## 2. Mask logic (pure, numpy and PIL)
- [x] 2.1 `editing/masking.py`: `detector_text`, `is_removal`, `grow` (integral-image max filter), `soft_mask` (grow + feather, sizes from the image), `composite`, `changed_area`, `mask_info`; unit tests
  - Feather sigma is a third of the grow (7 px, removals 19 px), so the object's own edge stays fully inside and the mask is exactly 0 beyond about twice the grow. `tests/test_masking.py`: 8 tests.
- [x] 2.2 `ObjectMasker` (Grounding DINO + SAM, imports torch/transformers only in `load`/`detect`/`segment`); checked on CPU with transformers 4.53.3 and torch 2.4.1 on a test image
  - Checked on CPU (Python 3.11, torch 2.4.1, transformers 4.53.3) with the pinned configs, tokenizer and processors but **random weights**: the session's network blocks the Hub's weight CDN. `detect` (post-processing with `threshold`/`text_threshold`) and `segment` (box prompts, `post_process_masks`, best-IoU pick) ran on a 1408 x 768 image and gave boxes in image pixels and a (768, 1408) bool mask. Detection quality and GPU timings are part of 5.2.

## 3. Worker
- [x] 3.1 `_ViewRestyler._load_models` loads the masker for editors with `mask_models`; `render(object_query=)` runs detect, segment, restyle, composite as separate stages; `mask.png` in the same commit; `mask` in run info; `mask_png` returned; not found fails before Kontext
  - The masker loads after the editor in `_load_models` (`load_mask_models` in the load stages). Grow and feather are timed in `segment`. A query on an editor without a masker fails with `preview_failed`.
- [x] 3.2 `tests/test_preview_worker.py`: masked future (pixels outside the mask equal the start image, `mask.png` committed, stages recorded), not found, editor without masker
  - `_worker` helper shared with the existing test; new test runs `KontextRenderer` with a fake masker and an editor that paints everything red.

## 4. Space
- [x] 4.1 `futures.future_params(object_query=)` (validate, store), `details_markdown` shows the object; `jobs.create_preview` and `modal_client.submit_preview` pass it on; the returned `mask_png` is kept like `image_jpeg`; tests
  - `JobManager.masks` keeps the returned PNG (never in `job.json`). `tests/test_futures.py`: 1 new test.
- [x] 4.2 `space/app.py`: "Object to change" field; "Changed area" image for a just-made future and for gallery tiles
  - "Changed area" (`_changed_area`) is a 512 px JPEG cached under `thumbs/changed/`, from the returned mask or `mask.png`; hidden for futures without a query. Checked that the Space builds; not clicked through in a browser.

## 5. Check
- [x] 5.1 ruff check, ruff format --check, pytest
  - Clean; 248 passed (incl. `tests/test_preview_worker.py`, which needs `modal`).
- [ ] 5.2 (user) `modal deploy modal_app/app.py`; on a scene with Kontext: "make the chairs red" with "chairs", a removal, and a query that finds nothing; check `run.json` stages (`detect` + `segment` + `composite` ≤ about 2 s warm) and that the area outside the mask is unchanged
- [x] 5.3 README Futures bullet
