# Design: add-tool-visibility-options

## Injection into Gradio's Settings dialog (user's choice)
Gradio 5 has no API to add items to its footer Settings dialog. A script in `BRIDGE_HEAD` watches the page with a `MutationObserver`; when the Settings dialog opens, it appends a "Scene tools" section with checkboxes (once per opening).
- Trade-off: integrated look vs. fragility. Selectors are kept in one place and the script fails silently, so a Gradio update only removes the section; it never hides tools by itself.
- Alternative rejected by the user: own "View options" accordion at the page bottom (supported components, robust).

## Hiding is client-side only
Tools are hidden with CSS classes on the tab buttons and the tool column (found by `elem_id`), not with `gr.Tab(visible=...)`. Reasons: no server round trip, works before the app has loaded the Settings, and hidden tools keep their state (a placement in progress, form inputs) because the components stay mounted. Until the saved choice arrives on page load, all tools show briefly.

## Global persistence
The choice is stored as `settings/space.json` (`{"hidden_tools": [...]}`) through the existing `Store` (`write_json`/`read_json`), cached in memory after the first read. A hidden textbox (`#tool-options-bridge`) carries the list as JSON: on page load the server fills it and the script applies it; a checkbox change makes the script write the new list into it, and its `input` event saves it on the server. Unknown tool ids are dropped; an unreadable file means all tools visible. Other open pages pick up a change on their next load.
