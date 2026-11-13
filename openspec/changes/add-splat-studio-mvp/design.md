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
| Text-to-3D | Open model that outputs Gaussians or mesh (e.g. TRELLIS-class); mesh is converted to Gaussians | Choose final model in task 4.1 after licence check. |
| Scene edit | Iterative dataset update (Instruct-NeRF2NeRF/GaussianEditor-style) with an instruction-following image editor | Most mature published approach; slow, so it runs as a long Modal job with a runtime cap. |
| COLMAP build | Ubuntu apt COLMAP, CPU SIFT; sequential matcher for video, exhaustive for photos | No CUDA COLMAP build to maintain; revisit if pose time dominates in task 2.4. |
| Training budget | 7,000 gsplat steps by default (gsplat default 30,000), 60 min hard timeout | Keeps cost per scene low; raise if quality is too poor. |
| Preview | Image-to-image/ControlNet-depth on Modal | Fast, keeps scene layout. |
| Storage | Private HF Hub dataset | Free, versioned, readable from both Space and Modal. |

## Open questions
- Real cost per scene is unknown; task 2.1 measures it, and we decide whether a Kaggle fallback is needed.
- Model weights should be cached in a Modal Volume to keep cold starts short.
