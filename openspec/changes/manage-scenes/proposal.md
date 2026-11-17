# Proposal: manage-scenes

## Why
Scenes are listed only as `<job id> · <input summary>`, which says little once there are several, and test or failed-looking scenes cannot be removed, so the list and the dataset keep growing.

## What changes
- **Scene details** (closed accordion under the scene list): the scene's id, date, input, settings and number of versions; a **Name** (max 80 characters, shown in the scene list and the jobs table) and **Notes** (max 2,000), saved with "Save details".
- **Delete scene:** a second click confirms and says what goes. One Hub commit removes `jobs/<id>/`, `scenes/<id>/` and the scene's edits, previews and generated objects; the Space's local downloads of it go too. Refused while one of those jobs is queued or running. Image editing jobs are kept.

## Impact
- Specs: `space-layout` (new requirement: Scene details).
- Files: new `src/splat_studio/space/scenes.py`; `space/jobs.py` (`Job.name`, `Job.notes`, `Store.delete_folders`), `space/versions.py` (`version_numbers`), `space/app.py`; tests in `tests/test_scenes.py`.
- Stored format: `job.json` of a reconstruction gains `name` and `notes`; older files read as unnamed. No Modal deploy needed.
- Deleted files stay in the dataset's git history until it is squashed.
