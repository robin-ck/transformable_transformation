# Tasks: add-splat-studio-mvp

## 1. Repo
- [x] 1.1 Create private GitHub repo; add README, licence, `.gitignore`, `pyproject.toml`
- [x] 1.2 GitHub Actions: ruff + pytest on push
- [x] 1.3 Document secrets setup (Modal token, HF token) in README

## 2. Reconstruction (Modal)
- [ ] 2.1 Spike: Modal GPU function called from a Space, with a cost/runtime log per run
  - code done (`gpu_probe`, `RunLog`, `modal_client`); open: deploy and record real cost per run
- [x] 2.2 Modal function: frame extraction, COLMAP poses, `gsplat` training, `.ply` + compressed export
- [x] 2.3 Readable failure messages for pose failure
- [ ] 2.4 Test on 2 sample videos and 1 photo set; record time per run
  - open: needs a Modal account (not reachable from the dev sandbox)

## 3. Space and viewer
- [ ] 3.1 Gradio Space with upload + validation (unit tests for limits)
- [ ] 3.2 Job client: spawn Modal job, poll, store results on HF Hub (unit tests with mocked Modal calls)
- [ ] 3.3 Embed JS splat viewer; orbit/pan/zoom/reset
- [ ] 3.4 Versions list, switch, download

## 4. Text editing
- [ ] 4.1 Licence check and pick text-to-3D, image-edit and preview models
- [ ] 4.2 Add object: generate, convert to Gaussians, place/scale/rotate, merge
- [ ] 4.3 Preview: render view, image-to-image with depth control
- [ ] 4.4 Edit scene: Modal job with iterative dataset update; optional region mask
- [ ] 4.5 Prompt log per version

## 5. Wrap-up
- [ ] 5.1 End-to-end test: upload → scene → add object → edit → preview
- [ ] 5.2 README usage guide with screenshots
