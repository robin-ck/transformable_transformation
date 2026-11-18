# Design: add-tool-visibility-options

## Injection into Gradio's Settings dialog (user's choice)
Gradio 5 has no API to add items to its footer Settings dialog. A script in `BRIDGE_HEAD` watches the page with a `MutationObserver`; when the Settings dialog opens, it appends a "Scene tools" section with checkboxes (once per opening).
- Trade-off: integrated look vs. fragility. Selectors are kept in one place and the script fails silently, so a Gradio update only removes the section; it never hides tools by itself.
- Alternative rejected by the user: own "View options" accordion at the page bottom (supported components, robust).

## Hiding is client-side only
Tools are hidden with CSS classes on the tab buttons and the tool column (found by `elem_id`), not with `gr.Tab(visible=...)`. Reasons: no server round trip, works before the app has loaded the Settings, and hidden tools keep their state (a placement in progress, form inputs) because the components stay mounted.

## Persistence
`localStorage` key `splat-studio:hidden-tools` (list of tool ids), wrapped in try/catch like the viewer's last-view storage. Unreadable or empty storage means all tools visible.
