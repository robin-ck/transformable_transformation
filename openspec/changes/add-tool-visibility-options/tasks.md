# Tasks: add-tool-visibility-options

## 1. Layout hooks
- [x] 1.1 `space/app.py`: `elem_id` on the tool column, the tool `gr.Tabs` and each sub-tab (history, objects, edit, preview, futures)

## 2. Global storage
- [x] 2.1 `space/tool_options.py`: tool ids, `clean`, `load`/`save` via `Store` (`settings/space.json`); unit tests
- [x] 2.2 `space/app.py`: hidden `#tool-options-bridge` textbox, filled on page load, saved on `input`

## 3. Client script
- [x] 3.1 Apply visibility: hide tab buttons; open the first visible tool if the open one is hidden; hide the tool column and let the viewer grow when all are hidden; trigger a viewer resize
- [x] 3.2 Inject the "Scene tools" checkboxes into Gradio's Settings dialog when it opens; selectors in one constant; fail silently

## 4. Tests and checks
- [x] 4.1 Unit test: the script's tool ids match the sub-tab `elem_id`s
- [x] 4.2 Playwright check on a local run: hide one tool, hide the open tool, hide all, reload in a second browser context; viewer camera kept
  - Ran on the local store (no scene there): hide one, hide the open one (switches to the first visible), hide all (panel gone, viewer 720 → 1216 px), show one again, second browser context sees the same choice. Camera not checked without a scene; the viewer iframe is not re-rendered.
- [x] 4.3 `ruff check .`, `ruff format --check .`, `pytest`

## 5. Docs
- [x] 5.1 README: one line on hiding scene tools via Settings
