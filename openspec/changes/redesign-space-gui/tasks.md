# Tasks: redesign-space-gui

Can start now: MVP tasks 4.3 (Preview), 4.4 (Edit scene) and 4.5 (prompt log) are committed.
If MVP task 5.3 (object merge on Modal) is done first or in parallel, keep the segmented
`.splat` writing (4.1 below) inside `editing/objects.merge_files`, which both use.

## 1. Layout
- [ ] 1.1 Split `build()` into New scene and Scene tabs; viewer left, tool sub-tabs (History, Add object, Preview, Edit scene) right; downloads under the viewer
- [ ] 1.2 Pick the start tab on load (Scene if a finished scene exists, opened at its newest version)
- [ ] 1.3 Finished-job notice (`gr.Info`) for reconstructions and scene edits; "Open newest scene" button; a finished edit offers its new version
- [ ] 1.4 Check locally: switching tools keeps scene, version, camera, a running placement and a running edit's progress line

## 2. Version history
- [ ] 2.1 History table from `space/prompt_log.scene_log` (versions, previews, running and failed edits; columns Date, Mode, Prompt, Model, Result) plus a From column for versions; unit tests extend `tests/test_prompt_log.py`
- [ ] 2.2 Replace the version radio with the history table; clicking a version row opens it (preview and unfinished-edit rows open nothing); keep "History of vN" (`history_markdown`) under the table; drop the separate Prompt log accordion

## 3. Click and drag placement
- [ ] 3.1 Viewer: click to put the object's base on the hit point (splat raycast, floor-plane fallback); floor height from `start_placement` (unit test)
- [ ] 3.2 Viewer: drag in the horizontal plane, Shift-drag to turn, orbit paused while dragging
- [ ] 3.3 iframe → Space bridge updates the sliders; "Place" reads the placement from the viewer first
- [ ] 3.4 One pointer mode at a time in the viewer: placement (Add object), region box (Edit scene) or orbit; leaving the Edit scene sub-tab switches the region box off, starting a placement does too
- [ ] 3.5 Check locally with the oak tree on the real scene: merged object sits where it was released; the region box still works in Edit scene

## 4. Hideable objects
- [ ] 4.1 Segmented `.splat` export and `objects` list in `version.json` (inherited from parent); unit tests for ranges over two added objects
- [ ] 4.2 Filtered `.splat` for hidden objects, cached per version + hidden set; unit tests
- [ ] 4.3 Show/hide controls per object for the open version; none for older versions and none for versions made by Edit scene
- [ ] 4.4 "Show object" toggle during placement
- [ ] 4.5 Check locally: hide and show on a version with two objects; downloads stay complete

## 5. Wrap-up
- [ ] 5.1 ruff + pytest pass
- [ ] 5.2 README: update the usage section for the new layout (screenshots come with MVP task 5.2)
- [ ] 5.3 MVP tasks.md: note under 4.5 that the History tab now shows the prompt log
