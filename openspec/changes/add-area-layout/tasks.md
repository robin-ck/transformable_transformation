# Tasks: add-area-layout

## 1. Area logic (pure, numpy and PIL)
- [x] 1.1 `editing/area.py`: `from_layers`, `check`, `to_png`/`from_png`, `bounds`, `crop_box`, `paste`, `area_prompt`, `area_info`, `summary`; `tests/test_area.py` (7 tests)
- [x] 1.2 `errors.area_empty()`

## 2. Worker
- [x] 2.1 `render(area_png=)`: area stage, crop edit, paste, composite; `area.png` and `mask.png` in the same commit; `area` in run info; `mask_png` returned; refuses empty areas, area plus object, editors without a masker
- [x] 2.2 `tests/test_preview_worker.py`: area future (crop size and prompt, pixels outside the mask equal the start image, area drawn at half size scaled up), the three refusals

## 3. Space
- [x] 3.1 `futures.future_params(area_mask=)`, `futures.is_limited`, details line; `jobs.create_preview(area_png=)` and `modal_client.submit_preview(area_png=)` (bytes never in `job.json`); tests
- [x] 3.2 `space/app.py`: "Area to change" accordion with an image editor (brush, eraser); reloads the start image when the view, the parent or the accordion changes; Make refuses an area drawn on another start image; "Changed area" for area futures
  - Checked in Chromium (Gradio 5.50) that a brush stroke on a 1408 x 768 background gives a mask of that size at the stroke's place. The full Space was built but not clicked through with a real scene.

## 4. Check
- [x] 4.1 ruff check, ruff format --check, pytest
- [ ] 4.2 (user) `modal deploy modal_app/app.py`; on a scene with Kontext: paint a patch and add "a weeping willow", paint the bank and "a reed bed", a removal ("remove the fence" over a fence); check the area outside is unchanged and that the object sits inside the patch
