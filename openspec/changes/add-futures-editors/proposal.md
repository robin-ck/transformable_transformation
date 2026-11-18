# Proposal: add-futures-editors

## Why
Futures have two editors. SDXL + depth ControlNet restyles the whole view and keeps the layout only loosely. FLUX.1 Kontext dev follows instructions well, but its licence is non-commercial, so it can never be the default (openspec/project.md). Newer open-licence editors (2025-2026) may follow instructions as well as Kontext or better, keep layouts tighter, or run much faster. We do not know which of them works best on our scenes, so we want to add several and compare them on the same views.

## What changes
- **Four new editors** in the Settings editor picker. Licences and revisions were read from the Hugging Face API on 2026-11-18:
  - **Qwen-Image-Edit-2511 + Lightning** (`qwen-edit`, Apache-2.0, 20B): instruction edits. Lightning LoRA, 4 steps for Fast and 8 for Quality. A100-80GB or H100.
  - **FLUX.2 klein 4B** (`flux2-klein`, Apache-2.0, 4B): fast instruction edits, about 13 GB of GPU memory. L4, A10G or L40S.
  - **Step1X-Edit v1.2** (`step1x-edit`, Apache-2.0, 12B): instruction edits with a "thinking" step that rewrites the prompt; reflection stays off. L40S, A100-80GB or H100.
  - **SDXL + union ControlNet + Lightning** (`sdxl-union`, Apache-2.0 / OpenRAIL++): whole-view restyle like the default, but guided by depth and edges (canny of the render), in 4 to 8 steps. L4, A10G or L40S.
- **Object and area for every instruction editor:** "Object to change", "Area to change" and "Detect objects" work with Qwen, klein and Step1X as they do with Kontext. The same Grounding DINO + SAM masker runs in each editor's container. Messages and labels that now say "Kontext" say "instruction editors" instead.
- **Try with several editors:** a "Try editors" picker next to **Make future**. It starts one future per ticked editor with the same start image, prompt, object or area and seed. Each runs on its own GPU class, in parallel. Every gallery tile names its editor, so Compare puts two editors side by side.
- The default editor stays **SDXL + depth ControlNet**. Whether one of the new editors becomes the default is decided after the trial, in a separate change.

## Impact
- Specs: `viewpoint-futures`. MODIFIED: "Exchangeable image editor", "Object-level masking" and "Drawn area (layout)" (no longer limited to Kontext). ADDED: "Try several editors".
- Files: `models.py` (5 `ModelSpec`s, 4 `EditorSpec`s, a new `EditorSpec.kind`), `editing/restyle.py` (4 editor classes, canny control image), `modal_app/app.py` (3 new images, 4 `_ViewRestyler` subclasses), `space/futures.py`, `space/jobs.py`, `modal_client.py`, `space/app.py`, `editing/instructions.py` (descriptions only), README, tests.
- Dependencies: diffusers 0.43.0 for Qwen and klein. Step1X-Edit v1.2 is not in released diffusers, so it uses the authors' fork pinned to a commit (see design). Each new editor has its own Modal image, so the existing editors' images do not change.
- Cost: a cold start downloads the weights once to the models volume: about 57 GB for Qwen, 16 GB for klein, 30 GB for Step1X and 3 GB for the union ControlNet and LoRAs. Qwen runs on the dearest GPUs (A100-80GB/H100). "Try editors" with all six ticked starts six containers. Each stays warm for 5 minutes after its last future.
- Risk: the gsplat image is built on torch 2.4.1, and diffusers 0.43 or the Step1X fork may need a newer torch. Task 2.1 checks this first. If they do, the editor's image gets a gsplat build on a newer torch (design).
- Deploy: `modal deploy modal_app/app.py`. The trial (section 6) runs on the GPUs and costs Modal credit.
- Not in this change: changing the default editor; removing Kontext; multi-image references (Qwen and klein can take a second image, for example a photo of a bench to add); video or multi-view futures.
