# Proposal: add-tool-visibility-options

## Why
The Scene tab's tool panel (History, Objects, Edit scene, Preview, Futures) always shows all five tools, even when a user only needs one or two. This crowds the panel and takes width from the viewer. Users should be able to switch off tools they do not use.

## What changes
- Gradio's built-in **Settings** dialog (footer of the page) gets a "Scene tools" section with one checkbox per tool sub-tab.
- An unchecked tool's sub-tab is hidden; if it was open, the first visible tool opens instead.
- If all tools are unchecked, the whole tool panel is hidden and the viewer uses the full width.
- The choice is global: saved in the Space's dataset and applied for every user and browser. With nothing saved, all tools are shown.

## Impact
- Spec: `space-layout` (MODIFIED "Tools next to the viewer", ADDED "Tool visibility options").
- Code: new `src/splat_studio/space/tool_options.py` (load/save the choice), `space/app.py` (elem_ids, a hidden bridge textbox, page-load handler, JavaScript in the page head).
- Risk: the Settings dialog is Gradio-internal markup with no public API. A Gradio update can break the injection; the fallback is that the section does not appear and all tools stay visible (no loss of function). See `design.md`.
- Any user can change the choice for everyone (the Space has no accounts). No breaking changes.
