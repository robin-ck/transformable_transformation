# Design: add-viewpoint-futures

## Scope
One user, one browser, one session. No names, roles or passwords: whoever uses the Space can save viewpoints, edit starters and make futures. Group use is a later change; the storage below already records what it would need (viewpoint, parent), and an `author` field can be added then.

## Where it lives in the UI
- A **Futures** sub-tab in the Scene tab's tool panel, next to Preview, so the 3D viewer stays visible.
- Top: the scene's viewpoints as a strip of thumbnails, and "Save current view" (name box). Picking a viewpoint moves the viewer to its camera (the viewer already accepts an explicit camera when it opens a scene) and shows its version.
- Middle: starter directions (radio, with "none"), prompt box (English placeholder, e.g. "wide reed beds and a wooden footbridge"), Make, status line, result.
- Bottom: gallery of the selected viewpoint and Compare.
- **Make from the current view:** if the user has moved the viewer away from the selected viewpoint (or picked none), Make saves the current camera as a new viewpoint first (default name "view 5"), in the same job (below).
- Starters and the editor are set in a small "Settings" accordion in the sub-tab.

## Storage (private HF dataset)
Existing layout stays; one new top-level folder per scene.

```
viewpoints/<scene_id>/setup.json            viewpoints and starter prompts
viewpoints/<scene_id>/<vp_id>/render.png    plain colour render (viewer background behind empty parts)
viewpoints/<scene_id>/<vp_id>/depth.png     ControlNet control image (inverse depth, uint8)
viewpoints/<scene_id>/<vp_id>/view.json     camera, size, source version, num_gaussians, render time
previews/<job_id>/job.json                  a future: a preview job with the fields below
previews/<job_id>/preview.png               the restyled image
previews/<job_id>/run.json                  worker run log (unchanged)
```

- `setup.json`: `{"viewpoints": [{"id", "name", "version", "camera", "order", "created_at"}], "starters": [{"id", "label", "text"}], "editor": "sdxl-depth"}`. `vp_id` is a short random id, so renaming keeps the folder.
- A viewpoint is pinned to the scene version shown when it was saved. A new version does not change existing viewpoints; to compare futures on v2, save viewpoints on v2.
- Removing a viewpoint removes it from `setup.json` only; its folder and futures stay, so nothing is lost by a misclick.
- **Futures are preview jobs.** `job.json` `params` gains `viewpoint` (`vp_id`), `starter` (`id` and the text used), `parent` (job id or null) and `editor` (registry key). The Earlier previews list, the reopen machinery, the GPU fallback and the prompt log then work for futures as they are. Old previews have no `viewpoint` and do not appear in the Futures gallery.
- A future does not upload `render.png` and `depth.png` again; its `info` names the viewpoint folder they came from. A future with a parent uses the parent's `preview.png` as input.
- `scenes.delete_plan` adds `viewpoints/<scene_id>`; futures are already covered, as previews of the scene.
- Listing: the Space keeps finished future records in memory (they do not change) and reads only new `previews/` folders when the sub-tab refreshes. Gallery thumbnails (512 px JPEG) are made by the Space and kept in its local file cache.

## Rendering once per viewpoint
- `PreviewRenderer` gains `render_view(scene_path, camera, out_prefix)`: today's render step, uploading `render.png`, `depth.png` and `view.json` in one commit. It runs when the user saves a viewpoint, so the thumbnail shows what the restyle will start from.
- `render(...)` takes either a camera (today's Preview), a `view_prefix` (an existing viewpoint) or both (a new viewpoint from the current view: render, store the view, then restyle, all in one job, so there is one GPU call and one wait). It optionally takes an `input_path` (a parent's `preview.png`). The depth control always comes from the viewpoint.
- The worker keeps viewpoint images in memory per container (a few MB each), so a warm container serves later futures of the same viewpoint with no download.
- The time saved on Modal is small now that gsplat's kernels are prebuilt (render 0.1 s, scene download 0.9 s in job `20261116-105801-7a0452`), but it removes the `.splat` load for big scenes (50 MB+, 1.5 M Gaussians on the GPU) and guarantees that all futures of a viewpoint start from identical pixels, whatever later changes to the renderer.

## Latency
Measured (reopen-previews 2.2 and 5.3): a warm preview runs 13.9 s on Modal (restyle about 10.5 s, upload 2.8 s in one commit); a cold container adds about 10 s of model loading plus container start; the page then showed the result 30 to 50 s after the job finished, because the Space's poll writes `job.json` and downloads the images from the Hub first.

Targets for a future, from the click, with a free GPU:
- ≤ 20 s with a warm container (goal about 10 s), i.e. within 5 minutes of the previous preview or future.
- ≤ 90 s with a cold container.

How:
1. **Show the result without the Hub round trip.** The worker returns the restyled image (JPEG, about 200 kB) in its result; the Space shows it as soon as the poll sees the job done, and writes `job.json` and the thumbnail afterwards. Removes most of the 30 to 50 s.
2. **Poll every 1 s** instead of 5 s for the future the page is waiting on.
3. **No render, no scene download** for existing viewpoints (above).
4. **Warm-up:** opening the Futures sub-tab sends the existing `warm()` call (as the Preview sub-tab does), so the container is usually up by the time the prompt is typed.
5. **Optional, fewer restyle steps:** today 30 steps at strength 0.7 (21 denoising steps, 10.5 s on an L4). Compare 20 steps, and SDXL-Lightning 8-step if its licence (openrail++) fits, on the four starters of one viewpoint; keep the cheapest that looks the same. The target holds without it.

## Cost
- About 12 s on an L4 ($0.80/h) per future, about $0.003; 100 futures about $0.30.
- Plus up to 5 minutes of idle GPU after the last job, as for previews today. `max_containers=1` stays.

## Exchangeable editor
- `editing/restyle.py` defines the interface, in plain Python so the Space and tests can use it:
  ```python
  class Editor(Protocol):
      key: str              # "sdxl-depth", stored per future
      label: str            # shown in the prompt log and gallery
      licence: str          # checked against project.md before an editor is registered
      uses_depth: bool      # True: gets the viewpoint's control image
      def load(self) -> None: ...
      def restyle(self, image, depth, prompt: str, seed: int) -> Image: ...
  ```
- `models.py` keeps a registry `EDITORS = {"sdxl-depth": ...}` and `DEFAULT_EDITOR`. The SDXL pipeline code moves out of `PreviewRenderer` into an `SdxlDepthEditor`.
- Each editor runs in its own Modal class (GPU and image differ). The Space picks the class from the registry; the editor is chosen per scene in Settings (default `sdxl-depth`).
- **Later upgrade (not in this change):** Qwen-Image-Edit (Apache-2.0, about 20 B parameters; needs an L40S or larger and is likely slower than SDXL, so measure first). FLUX.1 Kontext dev keeps layout well but has a non-commercial licence: usable only while the project stays research use, and to be recorded as such.

## Prompt handling
- Prompts are in English (SDXL's text encoders follow English best); no translation step.
- The prompt sent to the editor is `starter text + ", " + own text`, trimmed; empty own text is allowed when a starter is chosen.
- A parent future's prompt is not repeated automatically; its image carries the idea, and the new prompt describes the next change.
- Prompts are limited to 300 characters; SDXL's encoders read at most 77 tokens, and the starter comes first so it is never cut off.

## Rejected
- **A separate "futures" folder outside `previews/`:** duplicates the reopen, status and prompt-log logic that previews already have.
- **Rendering each future from the scene again:** costs little time now, but two futures of a viewpoint could differ if the scene file or renderer changes.
- **A separate Futures top-level tab without the viewer:** only needed when participants use phones; with one user the viewer next to the tool is more useful for choosing views.
- **Multi-user features now** (names, facilitator password, phones, live gallery, session GPUs): not needed for one user; a later change.
