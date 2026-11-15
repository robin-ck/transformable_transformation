# Design: redesign-space-gui

## Layout in Gradio
- `gr.Tabs` with two tabs, New scene and Scene. Gradio cannot show one component in two tabs, and the tools need the viewer, so the tools are sub-tabs (`gr.Tabs` inside a `gr.Column(scale=2)`) next to the viewer (`scale=3`). An Edit tab of its own was rejected for this reason.
- Start tab: `demo.load` picks the tab (`gr.Tabs(selected=...)`) from the job listing it already makes.
- State that must survive tab switches (`obj_job`, `placing`, slider values) stays in `gr.State` as today; switching sub-tab does not fire events, so placement simply persists.
- Finished-job notice: the existing 10 s timer compares done jobs with the last listing and calls `gr.Info(...)`; an "Open newest scene" button next to the picker does the opening (a toast cannot hold a button).
- Small custom CSS (`gr.Blocks(css=...)`) for a fixed viewer height and a compact history list. No new Python dependencies.

## History list
- `gr.Dataframe` (read-only) built from `prompt_log.scene_log` (task 4.5): columns Date, Mode, Prompt, Model, Result, plus From for versions. Rows are versions, previews and edits that have not produced a version yet (running or failed). A row click (`.select`) opens a version row; other rows do nothing. It replaces the radio list and the Prompt log accordion. `history_markdown` ("History of vN": the chain of parents) stays under the table. v1 shows "reconstruction" and the input summary.
- No second history module: `prompt_log.py` already reads `version.json` and the preview and edit job records.

## Viewer ⇄ Space messages
Today the Space sends `placement`, `get_camera` and `region_mode` (Edit scene's box overlay, which captures the pointer while on), and the iframe answers `camera` (with `region`). New:
- iframe → Space `{type: "placement_changed", x, y, z, turn, size}` after a click or drag ends.
- Space → iframe `{type: "object_visible", id, visible}`.
- A small script in the page head listens for `placement_changed`, writes the JSON into a hidden `gr.Textbox` and dispatches an `input` event; that textbox's `.input` handler updates the sliders. Fallback if the bridge is flaky: "Place in new version" asks the iframe for its current placement first (same pattern as `GET_CAMERA_JS`), so the merged result always matches what the user sees.

## Pointer modes
- The viewer has one pointer mode at a time: orbit (default), placement (Add object, while an object is being placed) or region (Edit scene, while "Limit to a region" is on). The page keeps the mode in one variable; `region_mode` and the new placement messages set it, and turning one on turns the other off.
- Leaving the Edit scene sub-tab sends `region_mode` off and unticks the switch (sub-tab `.select` event); a new viewer page starts in orbit mode, as today.

## Click and drag
- Click: ray from the camera through the pointer; use the viewer library's splat raycast if it gives a hit, otherwise intersect the horizontal plane at the scene floor (the z of the 5th percentile of Gaussian heights, computed in `start_placement`). The object's base goes to the hit point.
- Drag: pointer-down on the object's bounding box starts a drag in the horizontal plane at the object's current base height; orbit controls are paused during the drag. Shift-drag turns the object around z. Size stays on the slider and the mouse wheel over the object is not used (it zooms the camera).

## Hiding objects
- Merged `.splat` files are sorted by size × opacity over all Gaussians, so object Gaussians end up scattered. Change: write the `.splat` in segments (base scene, then each object), each sorted on its own. `version.json` gets `objects: [{id, prompt, object_job, start, count}]`, inherited from the parent version plus the new object. Segment order only changes load order, not the picture.
- Hide: the Space builds a filtered `.splat` by dropping the hidden byte ranges (32 bytes per Gaussian, no `.ply` needed), caches it next to the version's download (key: version + hidden ids) and reloads the viewer. Downloads always use the full files.
- Placement toggle: the object is a separate splat scene in the viewer during placement, so "Show object" just sets its visibility in the iframe, no reload.
- Edit scene (task 4.4, done) rewrites Gaussians and re-sorts the `.splat`, so versions it makes have no `objects` list (objects become part of the scene); this already holds, as `edit_scene` writes no `objects` field.
- MVP task 5.3 moves the merge to a Modal function; the segmented writing and the `objects` list belong in `editing/objects.merge_files` and the version record of `place_object`, which both the Space and the Modal path use.
- Rejected: keeping objects as separate files per version and composing in the viewer. Cleaner hiding, but every version then needs several files, and downloads would need a merge step.
