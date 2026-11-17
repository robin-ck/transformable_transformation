# Tasks: add-detection-preview

## 1. Drawing
- [x] 1.1 `editing/annotate.py`: `draw_detections(image, detections, masks)` (distinct colours, semi-transparent masks, boxes, "n label score" on a filled label background, kept inside the image) and `detections_caption(labels)`; unit tests
  - Tableau 10 colours, masks at 45 % opacity, white text on the detection's colour; the label sits above the box, or inside its top when the box touches the image's top edge, and is moved left to stay inside the image. Font is 1/45 of the longer side (31 px at 1408) so it stays readable when shown small. Pillow's scalable default font, no font file. `tests/test_annotate.py`: 3 tests; a 1408 x 768 sample checked by eye.

## 2. Worker
- [x] 2.1 `ObjectMasker.segment` returns one mask per detection; the worker joins them for the blend mask
  - Re-checked on CPU with torch 2.4.1 and transformers 4.53.3 (pinned configs and processors, random weights): `segment` gives (3, 768, 1408) for three boxes.
- [x] 2.2 `render` draws `detections.jpg` (stage `annotate`), commits it with the future and returns `detections_jpeg`; extend `tests/test_preview_worker.py`
  - JPEG at quality 85, in the same commit as the future.

## 3. Space
- [x] 3.1 `JobManager` keeps the returned `detections_jpeg`; `futures.details_markdown` lists the detections; tests
  - `JobManager.detections`; details line "**Detected:** 1 chair 0.62 · 2 stool 0.50"; `futures.detections_caption`. 1 new test in `tests/test_futures.py`.
- [x] 3.2 `space/app.py`: labelled image with caption under the result (next to "Changed area"), for gallery tiles and both sides of Compare
  - "Detected objects" image next to "Changed area" (both 240 px high), its label is the caption; also under each side of Compare. Hidden for futures without a query and for masked futures made before this change. Checked that the Space builds; not clicked through in a browser.

## 4. Check
- [x] 4.1 ruff check, ruff format --check, pytest
  - Clean; 253 passed.
- [ ] 4.2 (user) `modal deploy modal_app/app.py`; make a masked future and check the labelled image and caption
- [x] 4.3 README Futures bullet
