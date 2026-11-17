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
# Licences allowed only for research use, and only for an editor that is not the
# default (the user picks it per scene). Recorded in openspec/project.md.
RESEARCH_ONLY_LICENCES = frozenset({"flux-1-dev-non-commercial-license"})


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
    gpu="L40S",
    role="object reference image from the prompt (4 steps; 24 GB transformer needs 48 GB)",
)
IMAGE_TO_3D = ModelSpec(
    repo_id="microsoft/TRELLIS-image-large",
    revision="25e0d31ffbebe4b5a97464dd851910efc3002d96",
    licence="mit",
    gpu="L40S",
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

# Futures editor (instruction edits): FLUX.1 Kontext dev changes what the prompt names
# ("remove the chairs", "paint the chairs red") and keeps the rest of the view. Gated on
# the Hub: the HF token's account must accept the licence. Non-commercial licence, so
# research use only (RESEARCH_ONLY_LICENCES). Read on 2026-11-17.
KONTEXT = ModelSpec(
    repo_id="black-forest-labs/FLUX.1-Kontext-dev",
    revision="24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d",
    licence="flux-1-dev-non-commercial-license",
    gpu="L40S",
    role="instruction edits of a view or future (12 B transformer, bf16, about 34 GB)",
)

# Object masks for Kontext futures (add-object-masking): an open-vocabulary detector finds
# boxes for the object the user names, SAM turns them into a mask; Kontext's result is
# kept inside it only. Both run in the Kontext container. Read on 2026-11-17.
MASK_DETECTOR = ModelSpec(
    repo_id="IDEA-Research/grounding-dino-base",
    revision="12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
    licence="apache-2.0",
    gpu="L40S",
    role="boxes of the object named in 'Object to change' (open-vocabulary detection)",
)
MASK_SEGMENTER = ModelSpec(
    repo_id="facebook/sam-vit-large",
    revision="6851e0441005b0fb96f2cc4dfac472f3d1b14af1",
    licence="apache-2.0",
    gpu="L40S",
    role="mask of the detected boxes (Segment Anything, box prompts)",
)


@dataclass(frozen=True)
class EditorSpec:
    """A restyle editor for previews and futures (add-viewpoint-futures).

    `key` is stored with every future, so keys never change once used. `modal_class` is
    the Modal class that runs it (each editor has its own GPU and image).
    """

    key: str
    name: str  # short, for the editor picker
    models: tuple[ModelSpec, ...]
    uses_depth: bool
    modal_class: str
    steps: int  # denoising steps for Quality (and the Preview tool)
    fast_steps: int  # denoising steps for Fast futures
    # Detector and segmenter for "Object to change" (add-object-masking); empty: the
    # editor cannot mask. Not part of `label`, so stored model labels stay the same.
    mask_models: tuple[ModelSpec, ...] = ()

    @property
    def label(self) -> str:
        """Model label as previews always stored it, e.g. 'sdxl@4621659 + controlnet@17bb979'."""
        return " + ".join(m.label for m in self.models)

    @property
    def licence(self) -> str:
        return " + ".join(sorted({m.licence for m in self.models}))

    @property
    def can_mask(self) -> bool:
        return bool(self.mask_models)

    @property
    def research_only(self) -> bool:
        return any(m.licence in RESEARCH_ONLY_LICENCES for m in self.models)


EDITORS = {
    "sdxl-depth": EditorSpec(
        key="sdxl-depth",
        name="SDXL + depth ControlNet",
        models=(PREVIEW_BASE, PREVIEW_DEPTH),
        uses_depth=True,
        modal_class="PreviewRenderer",
        steps=30,
        fast_steps=20,
    ),
    "flux-kontext": EditorSpec(
        key="flux-kontext",
        name="FLUX.1 Kontext dev (instruction edits, research use only)",
        models=(KONTEXT,),
        uses_depth=False,
        modal_class="KontextRenderer",
        # 30 steps took 14.7 s warm on an H100 at 1408 x 768 (2026-11-17). Quality uses
        # Kontext's own default (28); Fast (20) is about a third faster.
        steps=28,
        fast_steps=20,
        mask_models=(MASK_DETECTOR, MASK_SEGMENTER),
    ),
}
DEFAULT_EDITOR = "sdxl-depth"


def editor_spec(key: str | None) -> EditorSpec:
    """The registered editor for `key` (None: the default); KeyError names the known keys."""
    key = key or DEFAULT_EDITOR
    if key not in EDITORS:
        raise KeyError(f"unknown editor {key!r}; registered: {', '.join(EDITORS)}")
    return EDITORS[key]


ALL = (
    TEXT_TO_IMAGE,
    IMAGE_TO_3D,
    TEXT_TO_3D_FALLBACK,
    IMAGE_EDIT,
    PREVIEW_BASE,
    PREVIEW_DEPTH,
    KONTEXT,
    MASK_DETECTOR,
    MASK_SEGMENTER,
)
