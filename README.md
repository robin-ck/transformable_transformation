# Splat Studio (transformable_transformation)

Upload a video or photos, get a 3D Gaussian splat scene, and change it with text prompts (add objects, edit the scene, 2D concept previews).

Status: early development. See `openspec/` for the project description, specs and proposed changes.

## Architecture

- **Interface:** Hugging Face Space (Gradio) with an embedded web splat viewer.
- **GPU compute:** Modal serverless functions, called from the Space.
- **Storage:** private Hugging Face Hub dataset repo for inputs, scenes and versions.

## Development setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Lint, format check and tests (the same checks CI runs on every push):

```bash
ruff check .
ruff format --check .
pytest
```

## Secrets

Secrets never go into the repository. `.env` and `.modal.toml` are git-ignored.

| Secret | Used by | Where to get it |
|---|---|---|
| Modal token (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`) | Space, to start GPU jobs | modal.com → Settings → API Tokens |
| Hugging Face token (`HF_TOKEN`, write access) | Space and Modal, to read/write the private dataset | huggingface.co → Settings → Access Tokens |

**Local development**

```bash
pip install modal
modal token new
```

This stores the Modal token in `~/.modal.toml`. Set `HF_TOKEN` and `SPLAT_STUDIO_DATASET` in your shell, or put them in a `.env` file in the repo root (`KEY=value` per line); `space/app.py` reads it.

**Hugging Face Space:** Space → Settings → Variables and secrets → add `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` and `HF_TOKEN` as *secrets* (not variables).

**Modal functions:** see *GPU jobs (Modal)* below for the `huggingface-secret` Modal secret.

## GPU jobs (Modal)

GPU code lives in `modal_app/app.py`; the shared pipeline logic is in `src/splat_studio/reconstruction/`.

1. Create a private HF dataset repo (e.g. `<you>/splat-studio-data`).
2. Store the HF token and dataset name in one Modal secret:

```bash
modal secret create huggingface-secret HF_TOKEN=<your-token> SPLAT_STUDIO_DATASET=<you>/splat-studio-data
```

3. Run the GPU spike, which prints the GPU name, runtime and estimated cost:

```bash
modal run modal_app/app.py::gpu_probe
```

4. Deploy, so the Space can call the functions:

```bash
modal deploy modal_app/app.py
```

**Redeploy whenever `modal_app/app.py` changes**, including when you only test a branch locally: the Space (local or hosted) always calls the deployed Modal app, never your local copy. Deploy the worker before the Space, from a checkout of the same branch. A Space that is newer than the worker fails with errors such as `got an unexpected keyword argument`. Workers stay warm for a few minutes after a job; if one still runs old code after a deploy, stop it with `modal app stop splat-studio` and deploy again.

**Warm workers and caches.** Add object (`ObjectGenerator`), Preview (`PreviewRenderer`) and Edit scene (`SceneEditor`) keep their models loaded for `WARM_S` (5 min) after a job, so a job soon after the last one skips loading. An idle warm container still bills GPU time: up to about $0.07 (L4), $0.09 (A10G) or $0.16 (L40S) per 5 minutes (approximate). gsplat's CUDA kernels are built into the image (for A10G, L4 and L40S), so no job compiles them; the first deploy after this change rebuilds the image, which takes several minutes. Scene files that Edit scene and Preview read are cached on the `splat-studio-models` volume under `scene-cache/` (capped at 10 GB, least recently used versions go first); an edit also caches the version it creates, so editing it again skips the download.

Reconstruction reads inputs from `jobs/<job_id>/input/` in the dataset and writes `scenes/<job_id>/v1/scene.ply`, `scene.splat` and a run log with stage timings and estimated cost to `jobs/<job_id>/run.json`. GPU prices in `reconstruction/cost.py` are approximate; check modal.com/pricing.

## Space (Gradio UI)

The Space lives in `space/`; the logic it uses is in `src/splat_studio/space/`.

Run it locally (needs `pip install "gradio>=5,<6" huggingface_hub modal`):

```bash
python space/app.py
```

Run `modal deploy modal_app/app.py` from the same checkout first if `modal_app/app.py` changed (see above). With `SPLAT_STUDIO_DATASET` and `HF_TOKEN` set, jobs are stored in the HF dataset; without them, in `./local_data` (UI work only, since the Modal job reads its inputs from HF Hub).

Deploy to a Hugging Face Space (Gradio SDK) by uploading `space/` as the Space root and the package next to it:

```bash
hf upload <you>/splat-studio space/ . --repo-type space
hf upload <you>/splat-studio src/splat_studio splat_studio --repo-type space
```

Then add the four secrets (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN`, `SPLAT_STUDIO_DATASET`) in the Space settings, and make sure `modal deploy modal_app/app.py` has been run.

### Using the Space

The page has three tabs. It opens on **Scene** when a finished scene exists (at its newest version), otherwise on **New scene**.

- **New scene:** upload one video or 20 to 300 photos, click "Create scene", and follow the job in the jobs list. When a job finishes you get a notice; pick it in the "Finished scene" list of the Scene tab to open it. "Speed and quality" (Fast, Standard, High) and "Scene settings" set the training steps, the frames taken from a video and the max image size; the line under them estimates GPU time and cost, and settings estimated above 45 min are refused.
- **Scene:** pick a scene; the viewer is on the left, the tools on the right. Switching tools keeps the scene, version, camera and any object being placed.
  - **History:** every version with date, mode, prompt, model and the version it was made from. Click a row to open that version. "Prompt log" also lists the 2D previews.
  - **Objects:** a table of the objects in the open version and those generated for the scene. Click **Hide**/**Show** to drop an object from the view only (downloads always contain every object). Click **Add** on a generated object to place another copy. Click a placed object's name to move, turn, tilt or resize it; "Save as new version" re-places it there, the other objects stay put. To make a new object, describe it, or upload a photo of it (PNG, JPG or WebP up to 20 MB; the text then only names it), and generate it (a few minutes). Photos of one object on a plain background work best. Then click the scene to put its base on that spot, drag it to move it, Shift-drag to turn it, or use the sliders. "Show object" hides it while you look around; "Cancel" saves nothing.
  - **Edit scene:** a table lists every edit of the scene (running ones too). **Hide** on an edit the open version builds on shows the scene from before it, plus objects placed after it (view only, downloads unchanged; objects placed between two edits belong to the later edit). **Open** shows an edit from another branch. To make one, aim the camera, write an instruction (e.g. "make the table green"), pick **Speed and quality** (Fast, Balanced, Best, or your own values under "Edit settings") and start the edit; the estimate under the settings shows how long it takes (about 2.5 to 6.5 minutes on the test scene). The result is a new version, saved even if you close the page. Fast and Balanced have not yet been compared with Best for quality. The edit is strongest around the current view. Tick "Limit to a region" and drag a box in the viewer to change only that part (recommended for single objects).
  - **Preview:** aim the camera, describe a look, and get a restyled 2D image of that view. The scene is not changed. "Earlier previews" lists the scene's previews, also ones that finished after you closed the page; click one to show it, or to follow one that is still running. A preview takes the first free GPU of L4, A10G and L40S.
- **Image editing:** 2D images with FLUX.1-schnell, no scene needed. "Text to image" makes a picture from a prompt in the chosen format; "Text + image to image" changes an uploaded image towards the prompt ("Change" sets how far: low keeps more of the input). A result takes under a minute on a warm GPU; "Save as base (no change)" stores an uploaded image as it is, without a GPU step. "Use result as input" continues editing a result, and "Earlier images" (generated and saved images) loads when the tab opens and after each new result; a click puts that image in as the input. Files are stored under `images/<job_id>/` in the dataset.

## Licence

Apache-2.0, see `LICENSE`. Third-party models used by the project keep their own licences.
