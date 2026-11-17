# Tasks: sharpest-frames

## 1. Frames
- [x] 1.1 `frames.sharpness` (variance of the Laplacian) and `frames.pick_sharpest` (sharpest per window, time order); unit tests
- [x] 1.2 `frames.extract_frames`: 3x candidates in one ffmpeg pass (JPGs + grayscale on stdout), keep the sharpest, number 00000.jpg, ...; candidates go to a temporary folder that is removed
- [x] 1.3 Test with a real video in which only 1 of 3 candidates is sharp: every kept frame is sharp
- [x] 1.4 Frames stage estimate 15 -> 25 s

## 2. Check and deploy
- [x] 2.1 ruff + pytest pass
  - 2026-11-17: 177 passed, 1 failed: the known Windows-only `test_scene_cache.py::test_rejects_paths_outside_the_cache[/etc/passwd]`; not touched by this change.
- [ ] 2.2 Deploy Modal, then one Standard run on a handheld video; record the frames time here and compare the scene with an earlier run of the same video
