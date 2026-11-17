# Tasks: add-viewpoint-futures

## 1. Editor interface
- [x] 1.1 `editing/restyle.py`: `Editor` protocol and `SdxlDepthEditor` (today's SDXL + depth ControlNet call, same settings); `models.EDITORS` and `DEFAULT_EDITOR`
  - `models.EditorSpec` (key, name, models, uses_depth, `modal_class`; `label` and `licence` derived from its models) and `models.editor_spec(key)`. `restyle.make_editor(key)` builds an editor without loading it; torch and diffusers are imported only in `load`/`restyle`.
- [x] 1.2 `PreviewRenderer` uses `SdxlDepthEditor`; a preview from the Scene tab gives the same image for the same seed as before (compare one stored preview)
  - The pipeline call moved unchanged (same models, revisions, negative prompt, guidance 7.0, CPU generator seeded the same way, same strength/control scale/steps defaults); the stored model label is unchanged (test).
  - **Not verified on a GPU:** the image comparison needs the deployed worker (6.2); no GPU or Modal token in this container.
- [x] 1.3 Unit tests: registry lookup, unknown editor key, licence field present
  - `tests/test_models.py`: 4 new tests; 213 passed.

## 2. Viewpoints
- [ ] 2.1 `space/viewpoints.py`: read and write `viewpoints/<scene>/setup.json` (add, rename, reorder, remove viewpoints; add, edit, remove starters; editor per scene); pure logic, tests with `LocalStore`
- [ ] 2.2 Worker: `PreviewRenderer.render_view(scene_path, camera, out_prefix)` uploads `render.png`, `depth.png`, `view.json` in one commit
- [ ] 2.3 Space: Futures sub-tab with the viewpoint strip, "Save current view" (uses the viewer's camera and the shown version), rename, move up/down, remove; picking a viewpoint moves the viewer to it
- [ ] 2.4 `scenes.delete_plan` includes `viewpoints/<scene_id>`; test

## 3. Futures
- [ ] 3.1 `space/futures.py`: build the job params (viewpoint, starter id and text, parent, editor), compose the prompt (starter first, 300-character limit), list a viewpoint's futures (newest first, in-memory cache of finished records); tests
- [ ] 3.2 Worker: `render(...)` accepts `view_prefix`, a camera for a new viewpoint (render, store view, restyle in one job) and `input_path`; keeps viewpoint images in memory per container; returns the restyled image as a JPEG in its result
- [ ] 3.3 Space: starters, prompt, Make, status and result in the Futures sub-tab; Make from an unsaved current view saves it as a viewpoint; 1 s poll for the running future; shows the returned image before writing `job.json`; warm-up when the sub-tab opens
- [ ] 3.4 "Start from this" on a future: parent's `preview.png` as input, parent recorded
- [ ] 3.5 Prompt log lists futures as previews with the viewpoint name; test in `tests/test_prompt_log.py`

## 4. Comparison
- [ ] 4.1 Gallery: plain render first, then futures newest first with the start of their prompt; 512 px thumbnails cached locally
- [ ] 4.2 Compare: select two (or one and the plain render), show side by side with full prompt, starter and date

## 5. Optional speed-up
- [ ] 5.1 Compare 30 vs 20 restyle steps (and SDXL-Lightning if its licence fits) on four starters of one viewpoint; keep the cheapest that looks the same

## 6. Check
- [ ] 6.1 ruff + pytest pass
- [ ] 6.2 Deploy Modal; on a river scene, save 4 viewpoints and 4 starters, make futures from them and from an unsaved view, build on one, compare two; check `run.json` stage times and the click-to-image time (target ≤ 20 s warm, ≤ 90 s cold)
- [ ] 6.3 README: how to save viewpoints, set starters, make and compare futures
