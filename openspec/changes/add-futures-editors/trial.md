# Trial: add-futures-editors (2026-11-18)

Scene `20261117-125819-e5905b` (office kitchen), saved view **front** (`ea77a2bb`, 1408 x 768), seed 0, scene editor instructions as saved. The futures were made with `modal_app/app.py` deployed from this change and are stored in the dataset under `previews/<job>/`. Each contact sheet has the plain render first, then one tile per editor.

## Tasks
| Task | Prompt | Limit | Editors | Speeds | Sheet |
|---|---|---|---|---|---|
| restyle | turn this office kitchen into a bright Scandinavian café with plants and warm lighting | none | all six | Fast, Quality | `trial/sheet_restyle_*.jpg` |
| recolour | make the stools red | object "stools" | 4 instruction | Fast, Quality | `trial/sheet_recolour_*.jpg` |
| remove | remove the green bowl | object "green bowl" | 4 instruction | Fast | `trial/sheet_remove_fast.jpg` |
| add-in-area | a large potted olive tree | painted area: floor band x 950-1300, y 600-760 | 4 instruction | Fast | `trial/sheet_add-in-area_fast.jpg` |

All 28 futures finished; none failed.

## Speed
The restyle stage only, in seconds, warm. Wall-clock times are not comparable, because one container per editor ran its futures one after the other.

| Editor | GPU | Fast | Quality | Cold start incl. first weight download |
|---|---|---|---|---|
| SDXL depth | L4 | 8.0 | 10.4 | (existing) |
| Kontext | H100 | 10.0-10.7 | 13.9-14.0 | (existing) |
| SDXL union | L4 | 2.7 | 2.6 | 47 s |
| Qwen | H100 | 2.6-3.9 | 4.9 | 296 s |
| klein | L4 | 1.7-7.2 | 12.2 | 90 s |
| Step1X | L40S | 51.5-52.7 | 70.9-71.8 | 192 s |

Step1X spends most of its time in the "thinking" step, its MLLM rewriting the prompt. Its rewrites were sensible, for example "Change the color of the stools to red."

## Results
- **Whole-view restyle:**
  - **Qwen** is the best by a clear margin: a photoreal café that keeps the island, stools, back wall and screen where they were.
  - **klein** is also photoreal but changes the layout more (the island's shape, extra furniture).
  - **Kontext** ignores the layout and draws a different kitchen.
  - **Step1X** keeps the layout, but Fast is washed out and Quality looks like an illustration.
  - **SDXL depth** changes little and keeps the render's artefacts.
  - **SDXL union** barely changes the render at all. At strength 0.7 the Lightning schedule runs only 2 to 5 steps without guidance, so this editor needs a higher strength before it is useful.
- **Recolour (stools):**
  - **Qwen, klein and Step1X** give red stools with the right shape.
  - **Kontext** turned them into deformed red shapes.
  - The mask kept the rest of the view identical for all four.
- **Remove (green bowl):**
  - **Qwen and klein** removed it cleanly.
  - **Kontext** removed it but left smeared artefacts on the island's right end.
  - **Step1X** left the bowl in place.
- **Add in area (olive tree):**
  - No editor drew a convincing olive tree. The painted band (160 px high, at the bottom edge) is too flat for a tree, so this test says more about the area than about the editors.
  - **Step1X** drew a leafy plant in a pot (cut off at the bottom).
  - **klein** and **Kontext** drew a pot with a stump; Kontext also left a box artefact.
  - **Qwen** drew an empty pot.
  - This needs a rerun with a taller area.

## Verdict
1. **Qwen-Image-Edit-2511 (Lightning)** was best or joint best on every task except the add-in-area test, and was the second fastest (3 to 5 s on an H100). It is the candidate for the default futures editor.
2. **FLUX.2 klein 4B** came close to Qwen on recolour and remove, and runs on an L4. It is a good cheap option, but it changes the layout more in whole-view restyles.
3. **Step1X-Edit v1.2** gives good recolours and keeps layouts. It is 10 to 20 times slower than Qwen, and it failed the removal.
4. **FLUX.1 Kontext dev** was worse than Qwen and klein on every task here, and its licence is research-only.
5. **SDXL + union (Lightning)** is not usable as configured; it needs a higher strength (for example 0.9) and a rerun.

## Follow-up (not in this change)
- Make Qwen the default editor, and consider dropping Kontext.
- Tune SDXL union's strength; rerun add-in-area with a taller area.
- Repeat on a second scene (outdoor) before changing the default.
