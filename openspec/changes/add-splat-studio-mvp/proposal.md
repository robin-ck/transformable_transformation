# Proposal: add-splat-studio-mvp

## Why
Creating and changing a 3D Gaussian splat scene today means command-line tools, a local GPU and separate tools for generative edits. We want one browser interface where a user uploads a video or photos, gets a splat scene, and changes it with plain-language prompts. Free GPU on Kaggle makes this possible without owning hardware.

## What changes
- New **scene-reconstruction** capability: upload video/photos in the Space; a Kaggle GPU job estimates camera poses and trains a Gaussian splat; the result returns to the Space.
- New **scene-viewer** capability: interactive 3D viewer in the Space with orbit/fly controls, version history and download.
- New **text-editing** capability with three modes:
  - *Add object*: text-to-3D generates an object as Gaussians, which the user places, scales and rotates in the scene.
  - *Edit scene*: text-guided edit of the existing splat (e.g. "make the façade green"), run as a Kaggle job.
  - *Preview*: render the current view and restyle it as a 2D image from a prompt.
- Job status panel (queued, running, done, failed) because Kaggle jobs take minutes.
- Private GitHub repo with README, licence, CI, and the Kaggle notebook versioned as code.

## Impact
- Specs: adds `scene-reconstruction`, `scene-viewer`, `text-editing` (all new).
- Risks: Kaggle API job latency (minutes of queueing); 12 h / 30 h-per-week GPU caps; "Edit scene" is the least mature technique and may give blurry results; model licences (some text-to-3D models are non-commercial).
- No breaking changes (new project).
