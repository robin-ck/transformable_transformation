# Project: Splat Studio

## Purpose
Turn a user's video or photo set into a 3D Gaussian splat scene, view it in a browser, and change it with text prompts (add objects, edit the scene, make 2D concept previews).

## Stack
- **Interface:** Hugging Face Space (Gradio 5, Python 3.11), with an embedded web splat viewer (JavaScript, e.g. a Three.js-based splat renderer).
- **GPU compute:** Modal serverless functions (Python), called from the Space; jobs start in seconds and bill per second. Free tier: about $30 credit per month.
- **Fallback (later, optional):** Kaggle notebooks for long runs if Modal credit runs out.
- **3DGS:** `gsplat` (Apache-2.0) for training; COLMAP or a learned pose estimator for camera poses.
- **Storage:** Hugging Face Hub dataset repo (private) for inputs, scenes and job outputs.
- **Code hosting:** private GitHub repo; CI with GitHub Actions (lint + unit tests, no GPU).

## Conventions
- Python: `ruff` for lint/format, `pytest` for tests, type hints on public functions.
- Scene files: `.ply` (3DGS standard) as source of truth; compressed `.splat`/`.spz` for the viewer.
- Every edit creates a new scene version; originals are never overwritten.
- Secrets (Modal token, HF token) only in Space secrets / Modal secrets, never in the repo.

## Constraints
- Stay within Modal's free monthly credit for v1: cap job runtime and GPU type per job.
- Only open-licence models whose licence allows the intended (research) use.
- Inputs: one video (≤ 2 min, ≤ 500 MB) or 20 to 300 photos.
