# Proposal: add-viewpoint-futures

## Why
The Space is to be used to sketch how a river landscape could change. Full 3D edits (Edit scene) take minutes, cost cents each and only change colours, so they are too slow for trying many ideas. The Preview tool already turns a view into a restyled 2D image in about 15 to 20 s on a warm GPU, but each preview uses whatever camera the viewer happens to have, so two images of "the river in 2050" are hard to compare, and nothing groups them or lets one idea build on another. This change is layer 1 of the agreed plan: 2D futures from saved viewpoints, built on the existing preview. It is for one user in one session; several participants at once is a later change.

## What changes
- **Viewpoints:** in the 3D viewer, the user moves to a spot they want to change and saves the camera as a named viewpoint ("bridge", "east bank", "bird's-eye"; typically 4 to 8 per scene). Saving renders its colour and depth once and stores them, so later restyles skip rendering and every future of a viewpoint starts from exactly the same image. Viewpoints can be renamed, reordered and removed; picking one moves the viewer there.
- **Futures sub-tab** in the Scene tab, next to Preview: pick a viewpoint (or the current view, which is saved as a new viewpoint), optionally a starter direction, type a prompt and get a restyled image. Each future is stored with prompt, starter, viewpoint, scene version, editor model, seed and optional parent future.
- **Build on a future:** "Start from this" restyles a future's image again with a new prompt and records it as the parent.
- **Comparison:** a gallery of all futures of one viewpoint (plain render first, then newest first), and a side-by-side view of any two, or of one future and the plain render.
- **Starter prompts:** a few directions per scene (e.g. nature first, flood safe, recreation, housing), each a label plus prompt text that is put in front of the user's own text.
- **Exchangeable editor:** the restyle step goes behind one interface. SDXL + depth ControlNet stays the default; a stronger layout-keeping editor (e.g. Qwen-Image-Edit, Apache-2.0) can be added later without changing stored futures or the UI. Adding one is not part of this change.
- **Faster result:** the worker returns the restyled image directly, so the Space shows it without first downloading it from the Hub (today that adds 30 to 50 s).

## Impact
- Specs: new capability `viewpoint-futures`; `space-layout` (Tools next to the viewer) gains the Futures sub-tab. Text-to-image preview and the prompt log keep their behaviour; futures appear in the prompt log as previews.
- Files: `src/splat_studio/editing/preview.py` (split render from restyle), new `editing/restyle.py` (editor interface), new `space/viewpoints.py` and `space/futures.py` (pure logic, unit tested), `space/jobs.py` (future fields on preview jobs), `space/prompt_log.py`, `space/scenes.py` (delete plan includes `viewpoints/<scene>`), `space/app.py` (Futures sub-tab), `modal_app/app.py` (`render_view` method; `render` takes a cached view or a parent image and returns the image), `models.py` (editor registry), README.
- Storage: new `viewpoints/<scene_id>/` folder; futures are preview jobs under `previews/<job_id>/` with extra fields. Old previews stay valid (no viewpoint).
- Cost: about $0.003 of GPU time per future on an L4, plus the existing 5-minute warm idle window.
- Risks: SDXL's text encoders understand English best, so prompts are in English; SDXL img2img at 0.7 strength changes some layout, which the later editor upgrade addresses.
- Deploy: needs `modal deploy modal_app/app.py`.
- Not in this change (later changes): several participants at once (names, facilitator role, phones, live gallery, more GPUs), pre-generated 3D object kit, exact water-level plane, turning chosen futures into 3D (e.g. Marble, SpAItial), voting and comments.
