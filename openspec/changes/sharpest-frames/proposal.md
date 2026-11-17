# Proposal: sharpest-frames

## Why
Frames are taken from a video at fixed times, so a frame blurred by a quick camera move goes into pose estimation and training as it is. Blurred frames are a common cause of soft scenes and floaters in handheld captures.

## What changes
- **Sharpest frame per window:** frames are taken at 3 times the needed rate (no more than the video's own frame rate). Each one is scored with the variance of the Laplacian on a 480 x 480 grayscale copy, and only the sharpest of each group of 3 is kept. The frame count and spread over the video stay as before, so the sequential matcher still sees neighbouring frames in order.
- One ffmpeg pass writes the candidate JPGs and the grayscale copies (on stdout); no new dependencies (numpy only).
- **Estimate:** the frames stage is raised from 15 to 25 s. Locally, 180 frames from a 2 min 1080p video took 5.1 s instead of 3.4 s.

## Impact
- Specs: `scene-reconstruction` (Remote reconstruction job).
- Files: `reconstruction/frames.py`, `reconstruction/settings.py`; tests in `tests/test_reconstruction.py`.
- Deploy: `modal deploy modal_app/app.py`; the Space needs no update.
- Photo sets are unchanged.
