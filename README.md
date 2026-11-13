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

## Licence

Apache-2.0, see `LICENSE`. Third-party models used by the project keep their own licences.
