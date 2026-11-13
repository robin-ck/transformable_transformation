# Project: Splat Studio

## Purpose
Turn a user's video or photo set into a 3D Gaussian splat scene, view it in a browser, and change it with text prompts (add objects, edit the scene, make 2D concept previews).

## Stack
- **Interface:** Hugging Face Space (Gradio 5, Python 3.11), with an embedded web splat viewer (JavaScript, e.g. a Three.js-based splat renderer).
- **Heavy compute:** Kaggle GPU notebooks (free tier, ~30 GPU h/week, sessions max 12 h), triggered and polled through the Kaggle API.
- **Light compute:** Hugging Face ZeroGPU (short calls) for text-to-image and text-to-3D where they fit.
- **3DGS:** `gsplat` (Apache-2.0) for training; COLMAP or a learned pose estimator for camera poses.
- **Storage:** Hugging Face Hub dataset repo (private) for inputs, scenes and job outputs.
- **Code hosting:** private GitHub repo; CI with GitHub Actions (lint + unit tests, no GPU).

## Conventions
- Python: `ruff` for lint/format, `pytest` for tests, type hints on public functions.
- Scene files: `.ply` (3DGS standard) as source of truth; compressed `.splat`/`.spz` for the viewer.
- Every edit creates a new scene version; originals are never overwritten.
- Secrets (Kaggle key, HF token) only in Space secrets / Kaggle secrets, never in the repo.

## Constraints
- Free tiers only for v1; jobs must survive minutes-long queues.
- Only open-licence models whose licence allows the intended (research) use.
- Inputs: one video (≤ 2 min, ≤ 500 MB) or 20 to 300 photos.
