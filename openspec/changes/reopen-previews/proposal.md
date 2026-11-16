# Proposal: reopen-previews

## Why
A preview is shown only in the page that started it, and only while that page keeps polling. Previews often wait minutes for an L4 (start waits of 4 to 41 min on 2026-11-15 and 16), so users reload or leave. The job then finishes on Modal and its images are stored under `previews/<job_id>/`, but the Space never shows them, and the prompt log keeps listing them as queued because it never updates them. On 2026-11-15 this led to three runs of "make it yellow", none of which the user saw.

## What changes
- **Earlier previews list** in the Preview sub-tab: the open scene's previews, newest first, with date, version, prompt and status. Clicking a finished preview shows its stored images (restyled image and plain render) in the Preview sub-tab, as a new result would be shown. Clicking a running one follows it like a preview just started; a failed one shows its error.
- **Up-to-date status:** the list and the prompt log update unfinished previews from Modal when they are listed, so a preview that finished after the page was closed shows as done.
- **GPU fallback for Preview** (also `speed-up-edit-jobs` task 5.3): Preview may run on an A10G or L40S when no L4 is free, instead of waiting. The run log records the GPU it ran on.

## Impact
- Specs: `text-editing` (Text-to-image preview).
- Files: `src/splat_studio/space/prompt_log.py` (preview rows), `space/app.py` (list, click, refresh), `modal_app/app.py` (GPU list, GPU name in the run log), tests in `tests/test_prompt_log.py`.
- Deploy: the GPU fallback needs `modal deploy modal_app/app.py`; the list works without it.
- Cost: an A10G costs about 1.4 times an L4 per second, an L40S about 2.4 times; a warm preview runs about 20 s, so a few tenths of a cent more per preview.
- No stored format changes: previews keep their `job.json`, `run.json` and images.
