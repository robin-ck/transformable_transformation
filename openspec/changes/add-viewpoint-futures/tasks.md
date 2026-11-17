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
- [x] 2.1 `space/viewpoints.py`: read and write `viewpoints/<scene>/setup.json` (add, rename, reorder, remove viewpoints; add, edit, remove starters; editor per scene); pure logic, tests with `LocalStore`
  - A scene without `setup.json` starts with four default starters (nature first, flood safe, recreation, housing), which the user can change. Starters are edited as one "label: prompt text" per line; a label that existed keeps its id. Viewpoints also store the viewer's orbit centre (`target`), so picking one puts the viewer back exactly. `same_view` tells whether the viewer is still at a viewpoint (position and direction; fov and aspect ignored).
  - `tests/test_viewpoints.py`: 6 tests; 219 passed.
- [x] 2.2 Worker: `PreviewRenderer.render_view(scene_path, camera, out_prefix)` uploads `render.png`, `depth.png`, `view.json` in one commit
  - The render step is now `PreviewRenderer._render`, shared with `render`. `render_view` returns the render as JPEG bytes (for the thumbnail) and keeps the images in the container's memory. Also a `warm()` method: the design called it "existing", but there was none; it only starts a container (models load in `@modal.enter`). `modal_client.submit_view` and `warm_preview`.
  - `tests/test_preview_worker.py` runs `render_view` and `render` with the GPU, editor and Hub stubbed and checks every commit's file list (skipped where `modal` is not installed, e.g. CI). Not run on a GPU (6.2).
- [x] 2.3 Space: Futures sub-tab with the viewpoint strip, "Save current view" (uses the viewer's camera and the shown version), rename, move up/down, remove; picking a viewpoint moves the viewer to it
  - Sub-tab after Preview (the tabs are History, Objects, Edit scene, Preview; the spec delta's "Add object" is the Objects tab). The viewer got a `set_camera` message (`viewer.py`); a view on another version reopens the viewer on that version at the view's camera. Setups are cached in the Space (it is the only writer); a scene without views shows the "futures are made from saved views" hint.
  - Browser check (2026-11-17): the local Space with a fake GPU client (jobs finish 4 s after they start) on a 3k-Gaussian test scene, driven with Playwright, since this cloud session has no browser pane; headless Chromium renders splats in software, so the viewer's render loop was paused during the check (its camera still moves). Saved "bridge" and "east bank" at set cameras; picking "bridge" put the viewer at exactly (-2, -4, 2) looking at the origin, picking "east bank" at its camera; move up, rename ("east bank, low") and remove updated the strip; the removed view's future stayed on disk.
- [x] 2.4 `scenes.delete_plan` includes `viewpoints/<scene_id>`; test
  - Only when the folder exists (`viewpoints=` flag; the Space checks `list_dirs("viewpoints")`), since the Hub refuses to delete a missing folder. Futures are previews of the scene, so already covered. Test in `tests/test_scenes.py`; 221 passed. (Done before 2.3, which needs the whole Futures sub-tab.)

## 3. Futures
- [x] 3.1 `space/futures.py`: build the job params (viewpoint, starter id and text, parent, editor), compose the prompt (starter first, 300-character limit), list a viewpoint's futures (newest first, in-memory cache of finished records); tests
  - The 300 characters count the whole prompt (starter included); a longer one is refused with a message, not cut. `params` also keep `own_prompt` (the user's text; `prompt` is what the editor got) and `view_prefix`. `FutureIndex` reads only `job.json`s it has not seen finished. Also `caption`, `details_markdown` and `thumbnail` (512 px JPEG) for 4.1/4.2.
  - `JobManager.create_preview(..., future=, input_path=)`; the worker's returned `image_jpeg` is kept in `JobManager.images`, never in `job.json`; `refresh(job, save=False)` + `save(job)` let the Space show the image before writing `job.json`.
  - `tests/test_futures.py`: 6 tests; 227 passed.
- [x] 3.2 Worker: `render(...)` accepts `view_prefix`, a camera for a new viewpoint (render, store view, restyle in one job) and `input_path`; keeps viewpoint images in memory per container; returns the restyled image as a JPEG in its result
  - Written with 2.2 (same code). A future of an existing viewpoint commits only `preview.png` and `run.json`; a new viewpoint's render, depth and `view.json` go into the same commit as the future. The last 16 restyled images also stay in memory, so building on a recent future downloads nothing. The "running" `run.json` is now uploaded in a background thread (joined before the final commit, so it cannot overwrite the finished one), which takes one Hub commit off the click-to-image path.
  - Checked with stubs in `tests/test_preview_worker.py` (cold container reads the view from the dataset; missing parent fails cleanly). Not run on a GPU (6.2).
- [x] 3.3 Space: starters, prompt, Make, status and result in the Futures sub-tab; Make from an unsaved current view saves it as a viewpoint; 1 s poll for the running future; shows the returned image before writing `job.json`; warm-up when the sub-tab opens
  - The 1 s `gr.Timer` runs only while a future or a view render is pending. The future being polled stays in memory, so a tick reads only `run.json`, and `job.json` is written in a background thread after the image is shown. Make does not list versions (each `version.json` read is a Hub download); it builds the version path directly. Warm-up runs at most every 4 minutes.
  - Browser check (see 2.3): Make from the open camera saved "view 1" first; Make after picking "bridge" used it (no new view); a failing job showed the error and "try again", and the gallery did not change. Space overhead from click to image: 0.3 s (fake 4 s job shown 4.3 s after the click). With the viewer drawing in software, headless Chromium itself delayed the click by about 20 s; this is a test-browser artefact, not the Space.
- [x] 3.4 "Start from this" on a future: parent's `preview.png` as input, parent recorded
  - Works on the future just made and on any gallery tile; "Start fresh instead" clears it. The browser check found and fixed that the button did nothing on a just-made future (only gallery clicks set it). The child's details show "Built on: future <job>"; the worker gets `input_path` (tested in `tests/test_futures.py`, `tests/test_preview_worker.py`).
- [x] 3.5 Prompt log lists futures as previews with the viewpoint name; test in `tests/test_prompt_log.py`
  - Result column: `preview of v2 · view "bridge" (job …)`; a removed viewpoint shows as "removed view". The Space passes the scene's viewpoint names. 228 passed. (Done before the UI tasks, as it is pure logic.)

## 4. Comparison
- [x] 4.1 Gallery: plain render first, then futures newest first with the start of their prompt; 512 px thumbnails cached locally
  - Thumbnails under `<file_dir>/thumbs/` (views and futures); a just-made future's thumbnail comes from the returned JPEG. Browser check: plain render, then "add people on th…", then "a wooden footbri…".
- [x] 4.2 Compare: select two (or one and the plain render), show side by side with full prompt, starter and date
  - The last two tiles clicked are compared. The browser check found the two images stacked in the narrow tool panel (Gradio's column minimum width); fixed, now two 230 × 298 px images in one row.

## 5. Optional speed-up
- [ ] 5.1 Compare 30 vs 20 restyle steps (and SDXL-Lightning if its licence fits) on four starters of one viewpoint; keep the cheapest that looks the same
  - Optional; not done: it needs the deployed worker and a GPU.

## 6. Check
- [x] 6.1 ruff + pytest pass
  - Linux, Python 3.13 venv: ruff check and format clean; 229 passed (incl. `tests/test_preview_worker.py`, which needs `modal` and is skipped in CI).
- [ ] 6.2 Deploy Modal; on a river scene, save 4 viewpoints and 4 starters, make futures from them and from an unsaved view, build on one, compare two; check `run.json` stage times and the click-to-image time (target ≤ 20 s warm, ≤ 90 s cold)
  - Open: needs `modal deploy modal_app/app.py` from a machine with the Modal token (none in the cloud session) and a real river scene. Also covers 1.2's same-seed image comparison.
- [x] 6.3 README: how to save viewpoints, set starters, make and compare futures
  - "Futures" bullet under Using the Space; scene deletion now names futures and saved views.
