# Tasks: reopen-previews

## 1. Preview list
- [x] 1.1 `prompt_log.preview_rows(scene_id, previews)`: the scene's previews, newest first (Date, Version, Prompt, Status), plus their job ids in the same order; unit tests in `tests/test_prompt_log.py`
- [x] 1.2 Space: "Earlier previews" table in the Preview sub-tab, under the result; filled when a scene opens, when the Preview sub-tab is selected and when a preview starts or finishes
- [x] 1.3 Click a row: done shows its images in the result column; running or queued is followed by the existing 5 s poll; failed shows its error
- [x] 1.4 The list and the prompt log refresh unfinished previews from Modal (only unfinished ones are polled)
  - The list refreshes them and saves their `job.json`; the prompt log runs after it and reads the saved status, so it does not poll Modal itself (it already runs on every 5 s status change of a preview).

## 2. GPU fallback (`speed-up-edit-jobs` 5.3)
- [x] 2.1 `PREVIEW_GPUS = ["L4", "A10G", "L40S"]` (the image's gsplat kernels cover all three); the run log names the GPU the container got
  - `cost.gpu_from_device_name` maps the CUDA device name to a price key (unit test). The app definition loads locally with the list; not deployed yet.
- [x] 2.2 Deploy Modal; one preview from the Space, check `gpu` and the stage times in `run.json`
  - Before: the deployed worker (v15, from `9a5687c`) still compiled gsplat's kernels on each cold start (`compile_gsplat`, about 120 s) and had no `SceneEditor`, which `main`'s Space already calls for Edit scene.
  - 2026-11-16: deployed as v16 from this branch (uncommitted changes on `5d4572c`, so Modal shows no commit). On Windows, `modal deploy` with its output redirected to a file fails on the ✓ character; set `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`.
  - Job `20261116-105801-7a0452` ("same setting in winter with snow", cold container) from the local Space: start wait 16.6 s, run 19.4 s (imports 2.8, warm_gsplat 1.0, load_models 5.7, download 0.9, render 0.1, restyle 10.5, upload 7.6), click to finish 37.8 s, `gpu: "L4"`. An L4 was free, so the fallback itself was not exercised.
  - The page showed the result about 30 to 50 s after the job was done (still "Restyling" 66 s after the click). That time is spent in the Space, not on Modal: the poll that saw it finish also writes `job.json` and downloads both images from the Hub.

## 3. Check
- [x] 3.1 ruff + pytest pass
  - 164 passed, 1 failed: `test_scene_cache.py::test_rejects_paths_outside_the_cache[/etc/passwd]` raises `PermissionError` instead of `ValueError` on this Windows machine; `scene_cache` is not touched by this change.
- [x] 3.2 Local Space against the dataset: the earlier "make it yellow" previews show as done and open; a running preview can be followed after a reload
  - 2026-11-16, scene `20261115-184011-7e524c`: the three "make it (more) yellow" previews of 2026-11-15, stuck as queued in `job.json`, were listed as done and opened with both images. Clicking "same setting but more autumn like" (started from the hosted Space, waiting for an L4) followed it; its images appeared when it finished. Start wait 815.8 s (about 680 s for a GPU, 135 s container start), run 25.9 s, on an L4.
- [x] 3.3 README usage: one line on earlier previews; `speed-up-edit-jobs` 5.3 notes the fallback

## 5. One Hub commit for a worker job's outputs
(Numbered 5 so it does not clash with section 4 on `claude/magical-turing-b5a633`.)
- [x] 5.1 Preview, object and image jobs commit their files and the final `run.json` in one `create_commit` (`splat_studio.hub_outputs.finish_with_outputs`); the edit job commits its five files to `edits/<job>/` in one commit (its scene version and final `run.json` stay separate commits, since the result needs the version number). `reconstruct` and `add_version` are unchanged.
  - The returned result (stored in `job.json`) keeps the commit time as the `upload` stage. The committed `run.json` cannot contain its own commit time, so it has no `upload` stage and its `seconds` stop before the commit. A done `run.json` now arrives in the same commit as the files.
- [x] 5.2 ruff + pytest: 177 passed, 1 failed (the known Windows-only `test_scene_cache` case); new `tests/test_hub_outputs.py`.
- [x] 5.3 Deployed 2026-11-16; one preview from the local Space (job `20261116-154959-acf2a3`, scene `20261115-213245-7a7de6`, L40S, cold container): `render.png`, `depth.png`, `preview.png` and `run.json` all come from one commit (`2461a0be`, "Preview 20261116-154959-acf2a3", 15:50:51 UTC). Measured `upload` stage **2.8 s**, run 13.9 s. Earlier previews with four commits: `upload` 77.9 s (job `20261116-144131-95cb60`, slow Hub) and 23.7 s (job `20261116-150233-a0e4bb`, Hub back to 3 to 6 s a commit). The Hub was fast during this run, so the 2.8 s is one fast commit; at the afternoon's ~26 s a commit it would be one commit instead of four.
