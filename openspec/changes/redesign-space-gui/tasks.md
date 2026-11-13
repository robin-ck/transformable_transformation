# Tasks: redesign-space-gui

Start after task 4.3 (Preview) of `add-splat-studio-mvp` is committed.

## 1. Layout
- [ ] 1.1 Split `build()` into New scene and Scene tabs; viewer left, tool sub-tabs (History, Add object, Preview) right; downloads under the viewer
- [ ] 1.2 Pick the start tab on load (Scene if a finished scene exists, opened at its newest version)
- [ ] 1.3 Finished-job notice (`gr.Info`) and "Open newest scene" button
- [ ] 1.4 Check locally: switching tools keeps scene, version, camera and a running placement

## 2. Version history
- [ ] 2.1 `versions.py`: history rows (version, date, mode, prompt, model, from); unit tests incl. v1 without `version.json`
- [ ] 2.2 Replace the version radio with the history table; row click opens the version

## 3. Click and drag placement
- [ ] 3.1 Viewer: click to put the object's base on the hit point (splat raycast, floor-plane fallback); floor height from `start_placement` (unit test)
- [ ] 3.2 Viewer: drag in the horizontal plane, Shift-drag to turn, orbit paused while dragging
- [ ] 3.3 iframe → Space bridge updates the sliders; "Place" reads the placement from the viewer first
- [ ] 3.4 Check locally with the oak tree on the real scene: merged object sits where it was released

## 4. Hideable objects
- [ ] 4.1 Segmented `.splat` export and `objects` list in `version.json` (inherited from parent); unit tests for ranges over two added objects
- [ ] 4.2 Filtered `.splat` for hidden objects, cached per version + hidden set; unit tests
- [ ] 4.3 Show/hide controls per object for the open version; none for older versions
- [ ] 4.4 "Show object" toggle during placement
- [ ] 4.5 Check locally: hide and show on a version with two objects; downloads stay complete

## 5. Wrap-up
- [ ] 5.1 ruff + pytest pass
- [ ] 5.2 README: update the usage section for the new layout (screenshots come with MVP task 5.2)
- [ ] 5.3 Note in MVP tasks that the History tab covers the UI part of task 4.5
