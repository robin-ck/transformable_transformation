# Design: add-object-masking

## Flow in the worker
`_ViewRestyler.render(..., object_query=None)` for a future with an object query:

1. Start image as today (the viewpoint's render, or the parent future's `preview.png`).
2. **detect** (stage): Grounding DINO on the start image with the query. Runs before Kontext, so "not found" fails in about a second instead of after the edit.
3. **segment** (stage): SAM with all boxes as prompts in one call; the best of SAM's three masks per box (highest predicted IoU), joined into one boolean mask.
4. Grow and feather the mask (pure numpy/PIL, `masking.soft_mask`; timed within the segment stage).
5. **restyle** (stage): Kontext on the full start image with the prompt, unchanged. Kontext sees the whole view, so lighting and perspective of the edit fit the scene.
6. **composite** (stage): `final = m * kontext + (1 - m) * start` in float, rounded to uint8, at the start image's size.
7. `preview.png` is the composite; `mask.png` (8-bit greyscale, the soft mask) goes into the same commit. The result also returns `mask_png` (bytes) so the Space can show the changed area without a Hub download.

Without a query nothing changes (no new stages, no `mask.png`).

## Query
- Grounding DINO expects lowercase phrases, each ending with a period: "chairs, Table" becomes "chairs. table.". `masking.detector_text` does this; at most 5 phrases and 60 characters (longer queries are refused in the Space with a message).
- Thresholds: box score 0.3, text score 0.25 (the transformers defaults used in the model card). All boxes above them are kept (several chairs), at most 20, highest scores first.

## Mask size
Proportional to the image's longer side `L` (1408 px for a 16:9 future):
- Grow: 1.5 % of `L` (21 px), so the soft edge sits outside the object.
- Removals: 4 % of `L` (56 px), so the background Kontext paints, and the object's shadow, are inside the mask. A removal is a prompt containing "remove", "delete", "erase", "take away", "get rid of" or "without" (`masking.is_removal`). This reads only the verb, never the object, so it does not conflict with "do not guess the noun".
- Feather: Gaussian blur after growing, with sigma a third of the grow (7 px; 19 px for removals), so the object's own edge stays fully inside the mask (the blur reaches 0.999 there) and the mask is exactly 0 beyond about twice the grow.
- Growing uses a square max filter computed with an integral image (O(pixels), any radius), so no SciPy or OpenCV is needed.

## Models
| Role | Repo | Licence | Size |
|---|---|---|---|
| detect | `IDEA-Research/grounding-dino-base` | Apache-2.0 | 233 M params, fp32 |
| segment | `facebook/sam-vit-large` | Apache-2.0 | 312 M params, fp32 |

- Both run through `transformers` 4.53.3 (`AutoModelForZeroShotObjectDetection`, `SamModel`), already in `kontext_image`, on torch 2.4.1. Grounding DINO's post-processing in 4.53 takes `threshold` and `text_threshold` (`box_threshold` is deprecated). Neither model calls `scaled_dot_product_attention` with `enable_gqa`, so the existing shim is not involved (it is installed before Kontext loads anyway).
- Base over tiny: better recall for small outdoor objects at about 0.1 s more. SAM large over base: cleaner edges on thin parts (chair legs, railings); huge is twice as slow for little gain at a feathered edge.
- They are `EditorSpec.mask_models` of the Kontext editor, not part of `EditorSpec.models`, so the editor's stored model label (and so old futures' labels) is unchanged. A masked future records the two labels in `run.json` `mask.models`.
- VRAM: about 2.2 GB in fp32 next to Kontext's 34 GB; fits the L40S (48 GB).
- Loaded in `_load_models` only when the editor has `mask_models`; weights are cached on the models volume like Kontext's.

## Speed
Kontext on Fast takes about 10 s warm on an H100. Expected on an H100: detect 0.1 to 0.3 s, segment 0.2 to 0.4 s (SAM's encoder runs once at 1024 px for all boxes), grow/feather/composite 0.1 to 0.3 s on the CPU at 1408 × 768. Target: at most about 2 s added, checked with the stage times in `run.json` (task 5.2).

## Storage
- `job.json` `params.object_query`: the query as typed (normalised whitespace), or absent.
- `run.json` `mask`: `{"query", "detections", "boxes" ([x0, y0, x1, y1] in start-image pixels), "scores", "labels", "area" (fraction of the image inside the hard mask), "grow_px", "feather_px", "removal", "models"}`.
- `previews/<job_id>/mask.png`.

## Space
- "Object to change (optional)" textbox under the prompt, with the info "Only that object changes; the rest of the view stays exactly as it is. Kontext editor only."
- `futures.future_params(..., object_query=)` validates (length, editor can mask) and stores it; `jobs.create_preview` passes it to `modal_client.submit_preview`, which passes `object_query` to `render`.
- Feedback: a small "Changed area" image under the result: the future with everything outside the mask dimmed to 35 % (`masking.changed_area`). Shown for a just-made future from the returned `mask_png`, and for gallery tiles from `mask.png` in the dataset. Hidden for futures without a mask.
- Details: "**Object:** chairs (3 found)".

## Failure
`errors.object_not_found(query)`: 'Nothing matching "chairs" was found in this view, so nothing was changed.' Tip: 'try another word (e.g. "chair" or "bench"), or leave "Object to change" empty to edit the whole view.' The failed `run.json` keeps the `detect` stage time.
