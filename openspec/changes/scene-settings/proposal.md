# Proposal: scene-settings

## Why
Every scene is built with the same fixed settings: 7,000 training steps, about 180 frames from a video at 1,600 px on the long side, and photos at their full size. A quick test of new footage costs as much as a final scene, and a scene that needs more detail cannot get more training. The Modal function already takes a step count, but the Space never sends one.

## What changes
- **Speed and quality** choice on the New scene tab: Fast, Standard (today's settings) or High, laid out like the Edit tab's.
- **Scene settings** (closed by default): training steps (1,000 to 30,000), frames from a video (60 to 300) and max image size (800 to 1,600 px). Moving a slider shows the matching preset, or Custom.
- **Estimate** of GPU time and cost for the chosen settings and upload. It is an upper bound from the slowest stage times of the three Standard runs so far (589, 629 and 866 s in total; training time follows how many Gaussians the scene grows). For a photo set it uses the number of photos instead of the frames setting.
- **Time limit:** settings estimated above 45 min are refused, by the Space before it uploads and again by the Modal function before GPU work. This keeps a margin before the 60 min hard timeout.
- **Photos are scaled down** to the max image size, as video frames already are. Today they are trained at full size (a phone photo is about 4,000 px).
- **Recorded:** the settings are saved in the job's `job.json` (`params`) and `run.json`, and shown in the Jobs table.

## Impact
- Specs: `scene-reconstruction` (Remote reconstruction job).
- Files: new `src/splat_studio/reconstruction/settings.py`; `reconstruction/pipeline.py`, `frames.py`, `errors.py`; `space/jobs.py`, `modal_client.py`, `modal_app/app.py`, `space/app.py`; tests in `tests/test_reconstruction.py` and `tests/test_space.py`.
- Deploy: `modal deploy modal_app/app.py` before the Space update. The deployed `reconstruct` takes `steps`, not `settings`, so a new Space would fail to start jobs on the old worker.
- Cost: Standard is unchanged ($0.21 to $0.31 so far). Fast and High are estimates only, up to about $0.15 and $0.65; their quality has not been compared.
- No breaking changes: older jobs have no settings and show none in the Jobs table.
