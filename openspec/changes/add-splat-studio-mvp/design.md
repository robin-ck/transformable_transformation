# Design: add-splat-studio-mvp

## Architecture
```
Browser ── HF Space (Gradio UI + JS splat viewer)
              │  spawn job / poll status (Modal Python client)
              ├──────────────► Modal GPU functions: reconstruct, edit scene,
              │                text-to-3D object, text-to-image preview
              └── HF Hub private dataset: uploads, scenes (.ply), versions, prompt log
```

## Decisions
| Topic | Choice | Why / alternative |
|---|---|---|
| Training | `gsplat` | Apache-2.0, fast; original 3DGS code is non-commercial. |
| Camera poses | COLMAP first; learned estimator (e.g. VGGT/MASt3R-style) as fallback | COLMAP is robust and well documented; fallback helps sparse photo sets. |
| GPU backend | Modal serverless functions (`.spawn()` + poll), outputs written to HF Hub | Starts in seconds, callable from code, ~$30/month free credit. Alternatives: Kaggle (free but queued), ZeroGPU (tiny free quota). |
| GPU choice | T4/L4 for previews and text-to-3D; A10G/L40S for training and scene edits | Balances speed against credit use. |
| Text-to-3D | FLUX.1 [schnell] makes a reference image, TRELLIS-image-large turns it into Gaussians; TRELLIS-text-xlarge as one-step fallback | Decided in 4.1 (see Models). TRELLIS outputs Gaussians directly, so no mesh conversion. |
| Scene edit | Iterative dataset update (Instruct-NeRF2NeRF/GaussianEditor-style) with an instruction-following image editor | Most mature published approach; slow, so it runs as a long Modal job with a runtime cap. |
| COLMAP build | Ubuntu apt COLMAP, CPU SIFT; sequential matcher for video, exhaustive for photos | No CUDA COLMAP build to maintain; revisit if pose time dominates in task 2.4. |
| Training budget | 7,000 gsplat steps by default (gsplat default 30,000), 60 min hard timeout | Keeps cost per scene low; raise if quality is too poor. |
| Preview | SDXL img2img + SDXL depth ControlNet on Modal (L4); depth from the gsplat render | Fast, keeps scene layout; no depth estimator needed. |
| Storage | Private HF Hub dataset | Free, versioned, readable from both Space and Modal. |

## Models (task 4.1, licences checked 2026-11-13 on the Hugging Face model cards)
Rule: open licence that allows research use, with no non-commercial clause and no
territory limit. Repo ids and pinned revisions live in `src/splat_studio/models.py`.

| Use | Model | Licence | GPU |
|---|---|---|---|
| Add object: text to image | `black-forest-labs/FLUX.1-schnell` | Apache-2.0 (gated: accept terms with the HF token) | A10G, CPU offload |
| Add object: image to Gaussians | `microsoft/TRELLIS-image-large` | MIT | A10G |
| Add object: fallback text to Gaussians | `microsoft/TRELLIS-text-xlarge` | MIT | A10G |
| Edit scene: per-view editor | `timbrooks/instruct-pix2pix` | MIT (card; SD 1.5 base, so OpenRAIL-M use rules apply) | A10G |
| Preview: base | `stabilityai/stable-diffusion-xl-base-1.0` | OpenRAIL++-M | L4 |
| Preview: depth control | `diffusers/controlnet-depth-sdxl-1.0` | OpenRAIL++-M | L4 |

Rejected:
- Hunyuan3D-2 / 2.1: Tencent Hunyuan Community licence does not apply in the EU, UK or South Korea.
- FLUX.1 Kontext [dev], FLUX.1 Depth [dev]: non-commercial licence; also 12B, slow for hundreds of edits.
- Stable Fast 3D: Stability Community licence (revenue cap), image-to-mesh only.
- Qwen-Image-Edit (20B) and Step1X-Edit: Apache-2.0 and better edits, but need an A100/H100 class GPU;
  too slow and costly per view for the iterative dataset update within the free credit. Revisit if
  InstructPix2Pix quality is too poor.
- Shap-E: MIT but low quality.

Risks:
- TRELLIS code is MIT, but some of its rendering extensions (diffoctreerast, mip-splatting, nvdiffrast)
  are research-only licences. We only need `save_ply` of the Gaussian output; install only what that
  path needs, and note it in the README if a research-only extension is required.
- TRELLIS needs 16 GB+ VRAM and spconv/kaolin builds; expect an image-build spike like gsplat's in 2.1.
- Cache all weights in a Modal Volume (open question below).

## Open questions
- Real cost per scene is unknown; task 2.1 measures it, and we decide whether a Kaggle fallback is needed.
- Model weights should be cached in a Modal Volume to keep cold starts short.
