# Tasks: redesign-space-gui

Can start now: MVP tasks 4.3 (Preview), 4.4 (Edit scene) and 4.5 (prompt log) are committed.
If MVP task 5.3 (object merge on Modal) is done first or in parallel, keep the segmented
`.splat` writing (4.1 below) inside `editing/objects.merge_files`, which both use.

## 1. Layout
- [x] 1.1 Split `build()` into New scene and Scene tabs; viewer left, tool sub-tabs (History, Add object, Preview, Edit scene) right; downloads under the viewer
- [x] 1.2 Pick the start tab on load (Scene if a finished scene exists, opened at its newest version)
- [x] 1.3 Finished-job notice (`gr.Info`) for reconstructions and scene edits; "Open newest scene" button; a finished edit offers its new version
- [ ] 1.4 Check locally: switching tools keeps scene, version, camera, a running placement and a running edit's progress line

## 2. Version history
- [x] 2.1 History table from `space/prompt_log.scene_log` (versions, previews, running and failed edits; columns Date, Mode, Prompt, Model, Result) plus a From column for versions; unit tests extend `tests/test_prompt_log.py`
- [x] 2.2 Replace the version radio with the history table; clicking a version row opens it (preview and unfinished-edit rows open nothing); keep "History of vN" (`history_markdown`) under the table; drop the separate Prompt log accordion

## 3. Click and drag placement
- [x] 3.1 Viewer: click to put the object's base on the hit point (splat raycast, floor-plane fallback); floor height from `start_placement` (unit test)
- [x] 3.2 Viewer: drag in the horizontal plane, Shift-drag to turn, orbit paused while dragging
- [x] 3.3 iframe → Space bridge updates the sliders; "Place" reads the placement from the viewer first
- [ ] 3.4 One pointer mode at a time in the viewer: placement (Add object), region box (Edit scene) or orbit; leaving the Edit scene sub-tab switches the region box off, starting a placement does too
- [ ] 3.5 Check locally with the oak tree on the real scene: merged object sits where it was released; the region box still works in Edit scene

## 4. Hideable objects
- [x] 4.1 Segmented `.splat` export and `objects` list in `version.json` (inherited from parent); unit tests for ranges over two added objects
- [x] 4.2 Filtered `.splat` for hidden objects, cached per version + hidden set; unit tests
- [x] 4.3 Show/hide controls per object for the open version; none for older versions and none for versions made by Edit scene
- [x] 4.4 "Show object" toggle during placement
- [ ] 4.5 Check locally: hide and show on a version with two objects; downloads stay complete

## 5. Wrap-up
- [x] 5.1 ruff + pytest pass
- [x] 5.2 README: update the usage section for the new layout (screenshots come with MVP task 5.2)
- [x] 5.3 MVP tasks.md: note under 4.5 that the History tab now shows the prompt log

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
- Merged with main (2026-11-15): main's Edit scene (MVP 4.4, run on Modal) replaced a second,
  untested implementation from this branch; its box region overlay now sits in the redesigned
  viewer next to click-and-drag placement, and its controls in the Edit scene sub-tab.
- 2.1: done as a table of versions (Version, Date, Mode, Prompt, Model, From); previews and
  running or failed edits stay in the prompt log accordion under it, not in the table.
- 1.3: reconstructions get a toast; a finished edit shows it in the status line and opens its
  version (if its scene is shown and no object is being placed), with a toast.

## 6. Objects and edits tables (2026-11-15)
- [x] 6.1 Objects sub-tab: one table (Object, Status, Action) for placed objects (Hide/Show,
  click to move or resize) and objects generated for the scene (Add); replaces the show/hide
  checkboxes and the "Generated objects" dropdown
- [x] 6.2 Move a placed object: `add_object.move_object` cuts its Gaussians out and merges its
  generated file again at the new placement, as a new version (mode `move_object`); object
  entries keep their placement (older entries take it from the version that added them)
- [x] 6.3 Edit scene sub-tab: table of the edits in the open version's line of parents (plus running or failed ones started from it); one shown at a time
- [x] 6.4 Checked in Chromium against a local scene with two objects: hide, add, cancel, move
  and save (v4 with the bench moved, the oak unchanged)

Notes: Gradio 5 maps a js hook's return values onto the non-State inputs only, so a State
in the middle shifted the values after it. This broke Preview and Edit scene (the prompt
arrived as None) and "Place" (sliders shifted). All hooked events now list States last.
Sliders in a hidden group have no value in the browser, so the objects table reads the
placement from the viewer instead. A table click can fire `select` twice: Hide/Show follow
the clicked label and a repeated Add is ignored.
