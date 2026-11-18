# Tasks: add-tool-visibility-options

## 1. Layout hooks
- [ ] 1.1 `space/app.py`: `elem_id` on the tool column, the tool `gr.Tabs` and each sub-tab (history, objects, edit, preview, futures)

## 2. Client script (`BRIDGE_HEAD`)
- [ ] 2.1 Read/write hidden tools in `localStorage` (`splat-studio:hidden-tools`), safe on errors
- [ ] 2.2 Apply visibility: hide tab buttons; open the first visible tool if the open one is hidden; hide the tool column and let the viewer grow when all are hidden; trigger a viewer resize
- [ ] 2.3 Inject the "Scene tools" checkboxes into Gradio's Settings dialog when it opens; selectors in one constant; fail silently

## 3. Tests and checks
- [ ] 3.1 Unit test: `BRIDGE_HEAD` contains the storage key and the tool ids match the sub-tab `elem_id`s
- [ ] 3.2 Playwright check on a local run: hide one tool, hide the open tool, hide all, reload; viewer camera kept
- [ ] 3.3 `ruff check .`, `ruff format --check .`, `pytest`

## 4. Docs
- [ ] 4.1 README: one line on hiding scene tools via Settings
