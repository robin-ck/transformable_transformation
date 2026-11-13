# Design: add-splat-studio-mvp

## Architecture
```
Browser ── HF Space (Gradio UI + JS splat viewer)
              │  submit job / poll status (Kaggle API, `kaggle kernels push/status/output`)
              ├──────────────► Kaggle GPU notebook: reconstruct, edit scene
              │  short GPU calls (ZeroGPU)
              ├──────────────► text-to-image preview, text-to-3D object
              └── HF Hub private dataset: uploads, scenes (.ply), versions, prompt log
```

## Decisions
| Topic | Choice | Why / alternative |
|---|---|---|
| Training | `gsplat` | Apache-2.0, fast; original 3DGS code is non-commercial. |
| Camera poses | COLMAP first; learned estimator (e.g. VGGT/MASt3R-style) as fallback | COLMAP is robust and well documented; fallback helps sparse photo sets. |
| Job bridge | Space pushes a parameterised Kaggle notebook via API, polls, pulls output to HF Hub | Only free route to Kaggle GPU; adds 1 to 5 min queue latency. |
| Text-to-3D | Open model that outputs Gaussians or mesh (e.g. TRELLIS-class); mesh is converted to Gaussians | Choose final model in task 4.1 after licence check. |
| Scene edit | Iterative dataset update (Instruct-NeRF2NeRF/GaussianEditor-style) with an instruction-following image editor | Most mature published approach; slow, so it runs on Kaggle. |
| Preview | Image-to-image/ControlNet-depth on ZeroGPU | Fast, keeps scene layout. |
| Storage | Private HF Hub dataset | Free, versioned, readable from both Space and Kaggle. |

## Open questions
- ZeroGPU needs a Hugging Face PRO account; without it text-to-3D and preview also go to Kaggle (slower).
- Kaggle's API for running notebooks programmatically has quotas; confirm in task 2.1 before building on it.
