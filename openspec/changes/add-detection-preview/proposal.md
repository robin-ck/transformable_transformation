# Proposal: add-detection-preview

## Why
With "Object to change" (add-object-masking), the user only sees the blended "Changed area". It does not say which objects the detector matched, how sure it was, or whether two chairs were taken as one. Without that, a wrong or missing detection is hard to spot, and the user cannot tell whether a poor future came from Kontext or from the mask.

## What changes
- **Labelled detection image:** for every future with an object query, the worker draws the start image with each detection's box and a semi-transparent coloured mask, labelled with a number, the matched phrase and the score ("1 chair 0.62"). Each detection has its own colour; labels sit on a filled background so they stay readable on any scene. Stored as `previews/<job_id>/detections.jpg` and returned to the Space with the result.
- **Shown in the Space:** under the result (next to "Changed area"), and for gallery tiles and both sides of Compare, with a caption such as "Detected: 3 × chair, 1 × stool".
- **Listed in the details:** each detection's number, label and score in the future's details text (`run.json` `mask` already records label, score and box).
- Drawing and caption are pure PIL/numpy code (`editing/annotate.py`) with unit tests.

## Impact
- Specs: `viewpoint-futures` "Object-level masking" gains the labelled detection image.
- Files: new `src/splat_studio/editing/annotate.py`; `editing/masking.py` (`ObjectMasker.segment` returns one mask per detection); `modal_app/app.py`; `space/jobs.py`; `space/futures.py`; `space/app.py`; README.
- Storage: `previews/<job_id>/detections.jpg` for masked futures (about 150 kB). Older masked futures have none and show the list only.
- Speed: one CPU drawing step (stage `annotate`), expected under 0.1 s.
- Deploy: needs `modal deploy modal_app/app.py`.
- Follow-up, not in this change: a "Detect only" button that shows the labelled image before Make future and lets the user untick detections.
