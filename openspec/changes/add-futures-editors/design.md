# Design: add-futures-editors

## Editor kinds
`EditorSpec.kind` is either `"restyle"` or `"instruct"`.
- **restyle** editors (`sdxl-depth`, `sdxl-union`) take the whole view, a control image and a negative prompt, plus strength and control scale. They cannot mask.
- **instruct** editors (`flux-kontext`, `qwen-edit`, `flux2-klein`, `step1x-edit`) take the start image and an instruction. They get `mask_models`, so they handle "Object to change", "Area to change" and "Detect objects".

`futures.py` and `app.py` switch on `kind` instead of `spec.key == "sdxl-depth"`. The `editor` key stored with each future does not change, so old futures read as before.

## Models (revisions from huggingface.co/api/models, 2026-11-18)
| Key | Models | Licence | Steps Fast / Quality | GPUs |
|---|---|---|---|---|
| `qwen-edit` | Qwen/Qwen-Image-Edit-2511 @6f3ccc0, lightx2v/Qwen-Image-Edit-2511-Lightning @107d238 (4- and 8-step bf16 LoRAs) | apache-2.0 | 4 / 8 | A100-80GB, H100 |
| `flux2-klein` | black-forest-labs/FLUX.2-klein-4B @e7b7dc2 | apache-2.0 | 4 / 8 | L4, A10G, L40S |
| `step1x-edit` | stepfun-ai/Step1X-Edit-v1p2 @ca85b97 | apache-2.0 | 20 / 28 | L40S, A100-80GB, H100 |
| `sdxl-union` | SDXL base (existing pin), xinsir/controlnet-union-sdxl-1.0 @801a4a3, ByteDance/SDXL-Lightning @c9a24f4 (4- and 8-step LoRAs) | openrail++ + apache-2.0 | 4 / 8 | L4, A10G, L40S |

The step counts are starting values and will be tuned in the trial. Qwen in bf16 needs the 20B transformer (about 41 GB) plus the Qwen2.5-VL text encoder (about 16 GB), so it needs 80 GB. An fp8 transformer on an L40S would be cheaper, but it needs extra quantisation code, so it waits until Qwen has proven itself. The two Lightning LoRAs are loaded as two adapters, and `set_adapters` picks one per future.

`sdxl-union` sends the union ControlNet two control images, depth and canny edges of the render, with control modes 1 and 3. The canny image is computed in the worker from the render, so saved views need no new files. Lightning runs with `EulerDiscreteScheduler(timestep_spacing="trailing")` and guidance 0, so the negative prompt has no effect. The sent-text preview says so.

Step1X runs with `enable_thinking_mode=True` and `enable_reflection_mode=False`, because reflection runs the edit several times. The rewritten prompt it returns is stored as `editor_prompt_rewritten` in `run.json` and shown in the future's details.

## Images and packages
The render (gsplat, torch 2.4.1) and the edit run in one container (`_ViewRestyler`), so every editor image starts from `gsplat_image`.
- `editors_image` (Qwen, klein): diffusers 0.43.0, a transformers release that has Qwen2.5-VL and Qwen3, peft, accelerate, and the existing `allow_sdpa_without_gqa` shim.
- `step1x_image`: the authors' fork `git+https://github.com/Peyton-Chen/diffusers@f5f1c98fa00cb4d0479af1b1b1c17d724345963a` (branch `step1xedit_v1p2`, 2026-01-13), with transformers 4.55.0 as its card asks. RegionE is not installed. When a diffusers release ships `Step1XEditPipelineV1P2`, the image moves to that release.
- `sdxl_union_image`: `preview_image` plus peft and opencv-python-headless (for canny).

Torch 2.4.1 is the open question. Task 2.1 imports each pipeline and loads it once on a GPU before any other work. If an image needs torch 2.5 or newer, that image gets its own gsplat build against that torch, using the same `GSPLAT_VERSION` and arch list. The build takes about 15 minutes, once.

## Try editors
`jobs.create_preview` already takes the editor key, so "Try editors" calls it once per ticked editor with the same params and seed. Each call is its own job and its own preview folder, and the jobs run in parallel because each editor has its own Modal class (`max_containers=1` each). When an object or area is given, the run skips restyle editors and says so. Opening the Futures sub-tab still warms only the scene's editor. The others start cold, and the status line says so.

## Not chosen
- A single image for all editors: one diffusers version cannot cover the 0.31 pin of the existing SDXL editor, Kontext's 0.35.1 and the Step1X fork.
- HiDream-E1.1 (MIT, 17B) and OmniGen2 (Apache-2.0): left out to keep the trial to four. Either can be added later as another `EditorSpec`.
- FLUX.2 klein 9B and FLUX.2 dev: non-commercial licences.
