# Tasks: edit-editor-instructions

## 1. Logic
- [x] 1.1 `editing/instructions.py`: `DEFAULTS`, `LABELS`, `resolve`, `clean`, `tail`, `editor_prompt`; `masking.object_prompt` and `area.area_prompt` take the texts; SDXL `restyle(negative_prompt=)`
- [x] 1.2 `Setup.instructions`; `futures.future_params(texts=)` stores the full set; `futures.sent_preview`; tests (`tests/test_instructions.py`)

## 2. Worker
- [x] 2.1 `render(instructions=)`; run.json `instructions`, `negative_prompt`; test that changed texts reach the editor

## 3. Space
- [x] 3.1 Settings: one text box per instruction, Save, Reset to defaults; the "Sent to …" line above Make future
  - Checked that the Space builds; not clicked through with a real scene.

## 3b. Rules and Preview tool
- [x] 3.2 `removal_words` and `pronouns` as editable word lists (empty turns the rule off); `masking.is_removal(texts=)`
- [x] 3.3 `preview_negative`: box in the Preview sub-tab, saved per scene on Preview this view; the worker uses it when there is no viewpoint; tests

## 4. Check
- [x] 4.1 ruff check, ruff format --check, pytest
- [ ] 4.2 (user) `modal deploy modal_app/app.py`; change an instruction, check the line above Make future and the future's details
