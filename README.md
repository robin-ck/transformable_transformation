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

Reconstruction reads inputs from `jobs/<job_id>/input/` in the dataset and writes `scenes/<job_id>/v1/scene.ply`, `scene.splat` and a run log with stage timings and estimated cost to `jobs/<job_id>/run.json`. GPU prices in `reconstruction/cost.py` are approximate; check modal.com/pricing.

## Space (Gradio UI)

The Space lives in `space/`; the logic it uses is in `src/splat_studio/space/`.

Run it locally (needs `pip install "gradio>=5,<6" huggingface_hub modal`):

```bash
python space/app.py
```

With `SPLAT_STUDIO_DATASET` and `HF_TOKEN` set, jobs are stored in the HF dataset; without them, in `./local_data` (UI work only, since the Modal job reads its inputs from HF Hub).

Deploy to a Hugging Face Space (Gradio SDK) by uploading `space/` as the Space root and the package next to it:

```bash
hf upload <you>/splat-studio space/ . --repo-type space
hf upload <you>/splat-studio src/splat_studio splat_studio --repo-type space
```

Then add the four secrets (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN`, `SPLAT_STUDIO_DATASET`) in the Space settings, and make sure `modal deploy modal_app/app.py` has been run.

### Using the Space

The page has two tabs. It opens on **Scene** when a finished scene exists (at its newest version), otherwise on **New scene**.

- **New scene:** upload one video or 20 to 300 photos, click "Create scene", and follow the job in the jobs list. When a job finishes you get a notice; "Open newest scene" in the Scene tab opens it.
- **Scene:** pick a scene; the viewer is on the left, the tools on the right. Switching tools keeps the scene, version, camera and any object being placed.
  - **History:** every version with date, mode, prompt, model and the version it was made from. Click a row to open that version. "Prompt log" also lists the 2D previews.
  - **Add object:** describe an object and generate it (a few minutes). Then click the scene to put its base on that spot, drag it to move it, Shift-drag to turn it, or use the sliders (they follow the viewer). "Show object" hides it while you look around. "Place in new version" saves a new version; "Cancel" saves nothing.
  - **Edit scene:** aim the camera, write an instruction (e.g. "make the table green") and start the edit; about 10 minutes later the result is a new version, saved even if you close the page. The edit is strongest around the current view. Tick "Limit to a region" and drag a box in the viewer to change only that part (recommended for single objects).
  - **Preview:** aim the camera, describe a look, and get a restyled 2D image of that view. The scene is not changed.
- **Objects shown** (under the viewer, for versions made with Add object): untick an object to hide it in the viewer. This changes only the view; downloads always contain every object. Versions made before object tracking have no such controls.

## Licence

Apache-2.0, see `LICENSE`. Third-party models used by the project keep their own licences.
