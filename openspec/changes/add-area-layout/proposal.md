# Proposal: add-area-layout

## Why
"Object to change" (add-object-masking) limits an edit to something the detector can find. River-landscape futures are mostly about other things: adding what is not there yet (a willow, a footbridge, a reed bed) and changing areas without clear edges (turn this meadow into a wetland, widen this bank). For those the user has to say *where*, and Kontext alone decides that itself, often somewhere else in the view. This change lets the user paint the area, so the layout of a future is set by the user and the rest of the view stays as it is.

## What changes
- **Area to change:** an accordion under "Object to change" in the Futures sub-tab with an image editor showing the next future's start image (the selected view's plain render, or the parent future when building on one). The user paints over the area with a brush (eraser to correct). The note says how much of the view is painted. Empty keeps today's behaviour. An area and an object query cannot be combined.
- **Crop, edit, paste:** the worker crops the start image around the area with context on every side (60 % of the area's size, at least 35 % of the image's shorter side), so Kontext sees the area large and in the middle of its input. Kontext gets the user's prompt plus "Make the change in the middle of the image, matching the perspective, scale, lighting and style of the scene" (removals: fill with what is behind). The edited crop is pasted back and kept only inside the grown, feathered area, with the same grow, feather and composite as object masks.
- **Stored:** `area.png` (the painted area, 1-bit, at the start image's size) and `mask.png` (the blend mask) next to `preview.png`; `run.json` gains `area` (bounds, crop, cover, grow and feather in px, removal); the job params gain `area` (bounds, cover). The "Changed area" image and the future's details show it.
- **Checks:** an empty or tiny area (under 0.2 % of the view) is refused in the Space and at the worker (`area_empty`); an area drawn on another start image than the future's is refused; only editors that can mask (Kontext) accept an area.

## Impact
- Specs: `viewpoint-futures` gains "Drawn area (layout)"; "Future from a viewpoint" stores the area.
- Files: new `src/splat_studio/editing/area.py` (pure); `reconstruction/errors.py` (`area_empty`); `modal_app/app.py` (`render(area_png=)`); `space/futures.py`; `space/jobs.py`; `modal_client.py`; `space/app.py`; README.
- Models: none new. Kontext only.
- Speed: one CPU stage `area` (well under 0.1 s); the crop is smaller than the view, so Kontext is no slower.
- Deploy: needs `modal deploy modal_app/app.py`.
- Not in this change: boxes or several areas with their own prompts in one future; SDXL areas; snapping the painted area to detected objects; a scene graph of the view.
