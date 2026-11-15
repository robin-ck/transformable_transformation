# Tasks: redesign-space-gui

Start after task 4.3 (Preview) of `add-splat-studio-mvp` is committed.

## 1. Layout
- [x] 1.1 Split `build()` into New scene and Scene tabs; viewer left, tool sub-tabs (History, Add object, Preview) right; downloads under the viewer
- [x] 1.2 Pick the start tab on load (Scene if a finished scene exists, opened at its newest version)
- [x] 1.3 Finished-job notice (`gr.Info`) and "Open newest scene" button
- [ ] 1.4 Check locally: switching tools keeps scene, version, camera and a running placement

## 2. Version history
- [x] 2.1 `versions.py`: history rows (version, date, mode, prompt, model, from); unit tests incl. v1 without `version.json`
- [x] 2.2 Replace the version radio with the history table; row click opens the version

## 3. Click and drag placement
- [x] 3.1 Viewer: click to put the object's base on the hit point (splat raycast, floor-plane fallback); floor height from `start_placement` (unit test)
- [x] 3.2 Viewer: drag in the horizontal plane, Shift-drag to turn, orbit paused while dragging
- [x] 3.3 iframe → Space bridge updates the sliders; "Place" reads the placement from the viewer first
- [ ] 3.4 Check locally with the oak tree on the real scene: merged object sits where it was released

## 4. Hideable objects
- [x] 4.1 Segmented `.splat` export and `objects` list in `version.json` (inherited from parent); unit tests for ranges over two added objects
- [x] 4.2 Filtered `.splat` for hidden objects, cached per version + hidden set; unit tests
- [x] 4.3 Show/hide controls per object for the open version; none for older versions
- [x] 4.4 "Show object" toggle during placement
- [ ] 4.5 Check locally: hide and show on a version with two objects; downloads stay complete

## 5. Wrap-up
- [x] 5.1 ruff + pytest pass
- [x] 5.2 README: update the usage section for the new layout (screenshots come with MVP task 5.2)
- [x] 5.3 Note in MVP tasks that the History tab covers the UI part of task 4.5

## Notes (2026-11-15)
- Built in a sandbox without Gradio or pytest (no package registry access). Checked with ruff,
  the unit tests run through a small pytest stand-in (58 tests in `test_space`, `test_objects`,
  `test_prompt_log` pass), `node --check` on all viewer and bridge scripts, and the event
  handlers run against a `LocalStore` with Gradio stubbed out. 5.1: later run with real
  pytest and Gradio 5.50: 109 passed.
- Fixed an existing test helper: `_files` in `tests/test_space.py` wrote into a folder it never
  created, so 9 upload/job tests failed with `FileNotFoundError`.
- 1.1: preview images moved into the Preview sub-tab. The History tab keeps "History of vN"
  and the prompt log (as an accordion), so 4.5 of the MVP lives there now.
- 1.3: the 10 s timer remembers which jobs were done at the last listing; the first listing
  after load only records them, so reloading the page does not repeat old notices.
- 3.1: the viewer's own click handler (which moves the orbit centre to the clicked splat) is
  replaced during placement; the clicked point comes from the library's splat raycast with
  the object hidden, else from the floor plane (5th percentile of Gaussian heights).
- 3.2: a drag starts only on the object's box (normalised: x, y in +-0.5, z in 0..1, times
  size); listeners run in the capture phase so orbiting never sees it.
- 3.3: values dragged beyond a slider's range are still used by "Place" (it reads the viewer
  first); the slider itself may show the clamped value.
- 4.3: hiding reloads the viewer and keeps the camera (the `get_camera` reply now carries the
  orbit target). During placement the object stays in the reloaded viewer.
- open: 1.4, 3.4 and 4.5 need a real browser with Gradio (local WSL run).
