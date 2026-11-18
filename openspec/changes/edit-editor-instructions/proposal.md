# Proposal: edit-editor-instructions

## Why
A future's prompt is not sent to the editor as typed. The Space and the worker add fixed text: SDXL gets a negative prompt, and Kontext gets "Keep the shape, material, texture…" after an object prompt and "Make the change in the middle of the image…" after an area prompt (with other texts for removals). These texts shape every result, yet the user could only see them after the fact (in a future's details) and could not change them. For research on how planners steer the editor, every instruction must be visible before a future is made and changeable per scene.

## What changes
- **Editor instructions:** the five fixed texts (`negative`, `object_change`, `object_removal`, `area_change`, `area_removal`) move to one module, `editing/instructions.py`, with today's texts as defaults. `{target}` stands for the named object.
- **Per scene:** Settings in the Futures sub-tab shows each text with when it is used; the user can change or empty it, save, or reset to the defaults. `setup.json` stores only the changed texts.
- **Visible before Make:** a line above Make future shows exactly what the editor will get for the current starter, prompt, object and area (and SDXL's negative prompt), updated as the user types.
- **Recorded:** each future stores the full set of texts in its params; run.json gains `instructions` and `negative_prompt`; the future's details show the negative prompt.
- **Compatible:** only changed texts go to the worker, so a worker deployed before this change still runs futures with the default texts.

## Impact
- Specs: `viewpoint-futures` gains "Editor instructions".
- Files: new `editing/instructions.py`; `editing/masking.py`, `editing/area.py`, `editing/restyle.py` (take the texts); `modal_app/app.py` (`render(instructions=)`); `space/viewpoints.py`, `space/futures.py`, `space/jobs.py`, `modal_client.py`, `space/app.py`; tests.
- Not in this change: the removal-verb rule (which words count as a removal), the it/them replacement, and the Preview tool's negative prompt stay fixed.
- Deploy: changed texts need `modal deploy modal_app/app.py`.
