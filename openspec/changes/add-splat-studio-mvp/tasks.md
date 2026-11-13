# Tasks: add-splat-studio-mvp

## 1. Repo
- [ ] 1.1 Create private GitHub repo; add README, licence, `.gitignore`, `pyproject.toml`
- [ ] 1.2 GitHub Actions: ruff + pytest on push
- [ ] 1.3 Document secrets setup (Kaggle key, HF token) in README

## 2. Reconstruction (Kaggle)
- [ ] 2.1 Spike: push, run and fetch a Kaggle notebook via API; record limits
- [ ] 2.2 Notebook: frame extraction, COLMAP poses, `gsplat` training, `.ply` + compressed export
- [ ] 2.3 Readable failure messages for pose failure
- [ ] 2.4 Test on 2 sample videos and 1 photo set; record time per run

## 3. Space and viewer
- [ ] 3.1 Gradio Space with upload + validation (unit tests for limits)
- [ ] 3.2 Job client: submit, poll, store results on HF Hub (unit tests with mocked API)
- [ ] 3.3 Embed JS splat viewer; orbit/pan/zoom/reset
- [ ] 3.4 Versions list, switch, download

## 4. Text editing
- [ ] 4.1 Licence check and pick text-to-3D, image-edit and preview models
- [ ] 4.2 Add object: generate, convert to Gaussians, place/scale/rotate, merge
- [ ] 4.3 Preview: render view, image-to-image with depth control
- [ ] 4.4 Edit scene: Kaggle job with iterative dataset update; optional region mask
- [ ] 4.5 Prompt log per version

## 5. Wrap-up
- [ ] 5.1 End-to-end test: upload → scene → add object → edit → preview
- [ ] 5.2 README usage guide with screenshots
