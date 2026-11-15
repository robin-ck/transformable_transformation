# Proposal: redesign-space-gui

## Why
The Space is one long Gradio page: upload, jobs table, scene picker, viewer, a version radio list with a collapsed prompt log, then the Add object, Preview and Edit scene sections stacked below. Users scroll between the viewer and the tools they steer it with (Preview and Edit scene both read the viewer's camera, Edit scene also a box drawn in it), objects can only be placed with five sliders, and how a version came about is split between the radio label, a history note and a separate log.

## What changes
- Split the page into tabs per stage (Gradio, desktop first):
  - **New scene**: upload, validation message, "Create scene", jobs list.
  - **Scene**: scene picker, viewer on the left; on the right a panel with sub-tabs **History**, **Add object**, **Preview**, **Edit scene**. Downloads sit under the viewer.
  - The tools live in the Scene tab, not a separate Edit tab, because every tool needs the viewer next to it (placement, camera for Preview and Edit scene, region box for Edit scene).
- **Version history with prompts**: one table for the scene, built from the existing prompt log (`space/prompt_log.py`, task 4.5): versions with date, mode, prompt, model and parent, plus previews and running or failed edits; clicking a version opens it. It replaces the version radio list and the separate Prompt log panel; "History of vN" stays under it.
- **One pointer mode at a time** in the viewer: orbit, object placement (click and drag) or the Edit scene region box.
- **Click and drag to place objects**: click a point in the scene to put the object's base there; drag the object to move it; Shift-drag (or a turn handle) to rotate; sliders stay as fine control and stay in sync with the viewer.
- **Hideable objects**: during placement, a toggle hides the object being placed; on versions made with Add object, each added object can be shown or hidden in the viewer. Hiding is view-only: it changes no version and no download.
- When a reconstruction or a scene edit finishes, the user gets a notice with a button to open the new scene or version.

## Impact
- Specs: adds `space-layout`; modifies `scene-viewer` (Scene versions, new Hide objects) and `text-editing` (Add object from text). These build on the deltas in `add-splat-studio-mvp`, which is not yet archived.
- Files: `space/app.py` (layout, events), `src/splat_studio/space/viewer.py` (picking, drag, hide, pointer modes next to the existing region box), `src/splat_studio/space/prompt_log.py` (history rows), `src/splat_studio/space/versions.py` (object ranges), `src/splat_studio/editing/objects.py` and `reconstruction/export.py` (`.splat` written in segments so each object's Gaussians stay together).
- Order: MVP tasks 4.3-4.5 are committed, so this can start. MVP task 5.3 moves the object merge to Modal; the segmented export belongs in `merge_files`, which both paths use.
- Risks: picking a point on Gaussians in the viewer library may be imprecise on sparse areas (fallback: ground plane at the scene floor); passing drag results from the iframe back to Gradio sliders needs a JS bridge; versions made before this change have no object ranges, so their objects cannot be hidden.
- No breaking changes: stored scenes, versions and jobs keep their format; new fields are optional.
