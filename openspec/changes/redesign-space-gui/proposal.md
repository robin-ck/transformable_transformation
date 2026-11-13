# Proposal: redesign-space-gui

## Why
The Space is one long Gradio page: upload, jobs table, scene picker, viewer, then the Add object and Preview sections stacked below. Users scroll between the viewer and the tools they steer it with, objects can only be placed with five sliders, and the version list shows a one-line label, so it is hard to see how a version came about. With more tools coming (Edit scene, task 4.4) the page will only get longer.

## What changes
- Split the page into tabs per stage (Gradio, desktop first):
  - **New scene**: upload, validation message, "Create scene", jobs list.
  - **Scene**: scene picker, viewer on the left; on the right a panel with sub-tabs **History**, **Add object**, **Preview** (and later **Edit scene**). Downloads sit under the viewer.
  - The tools live in the Scene tab, not a separate Edit tab, because every tool needs the viewer next to it (placement, camera for Preview).
- **Version history with prompts**: a list of versions showing number, date, mode, prompt, model and parent; selecting one opens it. Gives the UI for the prompt log (task 4.5).
- **Click and drag to place objects**: click a point in the scene to put the object's base there; drag the object to move it; Shift-drag (or a turn handle) to rotate; sliders stay as fine control and stay in sync with the viewer.
- **Hideable objects**: during placement, a toggle hides the object being placed; on versions made with Add object, each added object can be shown or hidden in the viewer. Hiding is view-only: it changes no version and no download.
- When a finished job belongs to the open tab's scene, the user gets a notice with a link to open it.

## Impact
- Specs: adds `space-layout`; modifies `scene-viewer` (Scene versions, new Hide objects) and `text-editing` (Add object from text). These build on the deltas in `add-splat-studio-mvp`, which is not yet archived.
- Files: `space/app.py` (layout, events), `src/splat_studio/space/viewer.py` (picking, drag, hide), `src/splat_studio/space/versions.py` (history rows, object ranges), `src/splat_studio/editing/objects.py` and `reconstruction/export.py` (`.splat` written in segments so each object's Gaussians stay together).
- Order: start after task 4.3 (Preview, now uncommitted) is committed, so both don't edit `space/app.py` at once.
- Risks: picking a point on Gaussians in the viewer library may be imprecise on sparse areas (fallback: ground plane at the scene floor); passing drag results from the iframe back to Gradio sliders needs a JS bridge; versions made before this change have no object ranges, so their objects cannot be hidden.
- No breaking changes: stored scenes, versions and jobs keep their format; new fields are optional.
