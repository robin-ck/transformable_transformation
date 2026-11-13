"""Models for text editing, picked after the licence check in task 4.1.

One place for repo ids, pinned revisions, licences and GPU choice, so the Modal
functions (tasks 4.2-4.4) and the prompt log (task 4.5) use the same names.
`version.json` stores `ModelSpec.label` as its `model` field.

Licences were read from the Hugging Face model cards on 2026-11-13. The project
only allows open licences that permit research use (openspec/project.md).
"""

from __future__ import annotations

from dataclasses import dataclass

# Licences that allow our research use without a non-commercial clause or a
# territory limit. Excluded: FLUX.1 [dev] family (non-commercial), Hunyuan3D
# (licence does not apply in the EU, UK or South Korea), Stability Community
# models (revenue cap). OpenRAIL licences carry use restrictions we follow.
ALLOWED_LICENCES = frozenset(
    {"mit", "apache-2.0", "openrail", "openrail++", "creativeml-openrail-m"}
)


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str  # Hub commit; bump on purpose, never float on main
    licence: str
    gpu: str  # Modal GPU type
    role: str

    @property
    def label(self) -> str:
        """Short name for the prompt log, e.g. 'microsoft/TRELLIS-image-large@25e0d31'."""
        return f"{self.repo_id}@{self.revision[:7]}"


# Add object (4.2): text -> image -> Gaussians. TRELLIS recommends this route over
# its text-conditioned models, which it calls less detailed.
TEXT_TO_IMAGE = ModelSpec(
    repo_id="black-forest-labs/FLUX.1-schnell",
    revision="741f7c3ce8b383c54771c7003378a50191e9efe9",
    licence="apache-2.0",
    gpu="A10G",
    role="object reference image from the prompt (4 steps, CPU offload on 24 GB)",
)
IMAGE_TO_3D = ModelSpec(
    repo_id="microsoft/TRELLIS-image-large",
    revision="25e0d31ffbebe4b5a97464dd851910efc3002d96",
    licence="mit",
    gpu="A10G",
    role="3D Gaussians from the reference image; no mesh conversion needed",
)
TEXT_TO_3D_FALLBACK = ModelSpec(
    repo_id="microsoft/TRELLIS-text-xlarge",
    revision="e0b00432b8e3a8ecee0df806ab1df9f7281f2be4",
    licence="mit",
    gpu="A10G",
    role="one-step text-to-3D if the two-step route fails or is too slow",
)

# Edit scene (4.4): instruction-following editor for the iterative dataset update.
# It runs on hundreds of renders per edit, so speed decides over quality here.
IMAGE_EDIT = ModelSpec(
    repo_id="timbrooks/instruct-pix2pix",
    revision="31519b5cb02a7fd89b906d88731cd4d6a7bbf88d",
    licence="mit",
    gpu="A10G",
    role="per-view edits from the instruction (Instruct-NeRF2NeRF setup)",
)

# Preview (4.3): SDXL img2img with depth ControlNet. Depth comes from the gsplat
# render of the current view, so no depth estimator is needed.
PREVIEW_BASE = ModelSpec(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    revision="462165984030d82259a11f4367a4eed129e94a7b",
    licence="openrail++",
    gpu="L4",
    role="restyle the rendered view from the prompt",
)
PREVIEW_DEPTH = ModelSpec(
    repo_id="diffusers/controlnet-depth-sdxl-1.0",
    revision="17bb97973f29801224cd66f192c5ffacf82648b4",
    licence="openrail++",
    gpu="L4",
    role="keep the scene layout via the rendered depth map",
)

ALL = (
    TEXT_TO_IMAGE,
    IMAGE_TO_3D,
    TEXT_TO_3D_FALLBACK,
    IMAGE_EDIT,
    PREVIEW_BASE,
    PREVIEW_DEPTH,
)
