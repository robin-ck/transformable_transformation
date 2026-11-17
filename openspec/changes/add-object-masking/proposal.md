# Proposal: add-object-masking

## Why
FLUX.1 Kontext dev follows instructions such as "make the chairs red", but it still sometimes changes other parts of the view (the table, the light, the trees). Users had to add "keep everything else as it is, also the table" to their prompts, and even then the rest of the image is not guaranteed to stay the same. For comparing futures of one spot, the parts the user did not ask to change must stay identical to the start image.

## What changes
- **Object to change:** an optional field next to the prompt in the Futures sub-tab (e.g. "chairs"). Empty keeps today's behaviour (no mask). Several objects can be named, separated by commas ("chairs, table"). The noun is not guessed from the prompt in this change; automatic extraction is a later step.
- **Detect and segment:** in the Kontext worker, Grounding DINO (open-vocabulary detector) finds boxes for the query in the start image, and SAM turns the boxes into one mask. The mask is grown and feathered so edges blend; removals ("remove", "take away", ...) get a wider margin, so the background Kontext paints (including the shadow) fits in.
- **Composite:** `final = mask * kontext_output + (1 - mask) * start_image`, at the start image's size. Outside the mask the future is the start image, pixel for pixel.
- **Clear failure:** if nothing is found, the future fails with 'Nothing matching "chairs" was found in this view' (new error `object_not_found`) before Kontext runs, instead of editing the whole image.
- **Feedback:** `mask.png` is stored next to `preview.png`; the Space shows a small "Changed area" image (the future with the unchanged part dimmed) under the result, and the future's details show the object query and how many were found.
- **Only for editors that can mask:** the Kontext editor. With SDXL + depth, the field is refused with a message (SDXL restyles the whole view by design).

## Impact
- Specs: `viewpoint-futures` gains "Object-level masking"; "Future from a viewpoint" stores the object query.
- Files: new `src/splat_studio/editing/masking.py` (pure mask logic, plus a lazily importing `ObjectMasker`), `models.py` (two `ModelSpec`s, `EditorSpec.mask_models`), `reconstruction/errors.py` (`object_not_found`), `modal_app/app.py` (`_ViewRestyler` loads the masker and masks in `render`), `space/futures.py`, `space/jobs.py`, `modal_client.py`, `space/app.py`, README.
- Models: `IDEA-Research/grounding-dino-base` and `facebook/sam-vit-large`, both Apache-2.0, pinned commits; run with transformers 4.53.3 and torch 2.4.1 of `kontext_image` (no new packages).
- Storage: `previews/<job_id>/mask.png` (masked futures only); `job.json` params gain `object_query`, `run.json` gains `mask` (query, detections, boxes, scores, labels, area, grow and feather in px).
- Speed: two new run stages, `detect` and `segment`, plus `composite`; target at most about 2 s added to a warm future (about 10 s of GPU time today on Fast). Cold start loads about 1.5 GB more weights.
- Deploy: needs `modal deploy modal_app/app.py` and a GPU check by the user (not possible from the cloud session).
- Not in this change: guessing the object from the prompt; masks for the SDXL editor; editing the mask by hand (click to add or remove parts).
