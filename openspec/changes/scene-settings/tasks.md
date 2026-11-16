# Tasks: scene-settings

## 1. Settings and estimate
- [x] 1.1 `reconstruction/settings.py`: limits, presets (Fast, Standard, High), `check_settings`, `preset_of`, time estimate per stage, `check_time` (45 min); unit tests
  - The estimate is an upper bound from the slowest stage times of the three Standard runs (`3749ae`, `7e524c`, `7a7de6`): Fast up to 6 min 50 s ($0.15), Standard 15 min ($0.33), High 30 min ($0.64). The longest settings (30,000 steps, 300 frames, 1,600 px) estimate 54 min and are refused.
- [x] 1.2 Video frames: frame count and image size from the settings (`frames.extract_frames`)
- [x] 1.3 Photos: scale down to the max image size, keep stored orientation and EXIF (`frames.copy_photos`); unit test with a real image
- [x] 1.4 `pipeline.run(settings=...)`: checks the settings and time (`bad_settings` error) before any work

## 2. Job flow
- [x] 2.1 `JobManager.create(upload, settings)` saves the checked settings in `params`; `modal_client.submit_reconstruction(job_id, settings)`
- [x] 2.2 Modal `reconstruct(job_id, settings)`; `run.json` records the settings used
  - Not deployed yet. The deployed worker takes `steps`, so deploy before the Space update.

## 3. Space
- [x] 3.1 New scene tab: Speed and quality preset, Scene settings sliders, estimate (photo count for photo sets)
- [x] 3.2 "Create scene" refuses settings over the time limit before uploading
- [x] 3.3 Jobs table: Settings column

## 4. Check and deploy
- [x] 4.1 ruff + pytest pass
  - 2026-11-16: 174 passed, 1 failed: the known Windows-only `test_scene_cache.py::test_rejects_paths_outside_the_cache[/etc/passwd]` (`PermissionError` instead of `ValueError`); not touched by this change.
- [x] 4.2 Local Space: presets, sliders, estimate and refusal work
  - 2026-11-16, local Space against the dataset: Fast sets 3,000 / 120 / 1,200; typing 30,000 steps shows Custom; 30,000 / 300 / 1,600 shows the 45 min warning; Standard resets the sliders. Older jobs show an empty Settings column. The photo path (300 photos) was checked by calling the handlers with a local store: the estimate uses the photo count, and "Create scene" over the limit uploads nothing.
- [ ] 4.3 Deploy Modal, then one Fast run from the Space; record times and cost here and update the estimate constants
- [x] 4.4 README usage line
