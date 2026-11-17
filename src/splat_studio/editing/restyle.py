"""Restyle editors (add-viewpoint-futures): one interface for the 2D restyle step.

A preview or future turns a rendered view (colour, plus a depth control image) into a
restyled image from a prompt. The model that does this sits behind `Editor`, so a
stronger layout-keeping editor can be added later without changing stored futures or
the UI. Which editors exist is listed in `models.EDITORS`; `make_editor` builds one.

This module imports no GPU libraries at import time, so the Space and the tests can use
it; `SdxlDepthEditor.load` and `restyle` import torch and diffusers when called.
"""

from __future__ import annotations

from typing import Any, Protocol

from splat_studio import models

NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, oversaturated, text, watermark"


class Editor(Protocol):
    key: str  # registry key, stored with every future, e.g. "sdxl-depth"
    label: str  # shown in the prompt log and gallery
    licence: str  # every model's licence; checked against models.ALLOWED_LICENCES
    uses_depth: bool  # True: gets the viewpoint's depth control image

    def load(self) -> None: ...

    def restyle(self, image: Any, depth: Any, prompt: str, seed: int, **options: Any) -> Any: ...


class SdxlDepthEditor:
    """SDXL img2img with a depth ControlNet (the Preview tool's settings since task 4.3)."""

    def __init__(self, spec: models.EditorSpec = models.EDITORS["sdxl-depth"]) -> None:
        self.spec = spec
        self.key, self.label = spec.key, spec.label
        self.licence, self.uses_depth = spec.licence, spec.uses_depth
        self.pipe = None

    def load(self) -> None:
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

        base, depth = self.spec.models
        controlnet = ControlNetModel.from_pretrained(
            depth.repo_id,
            revision=depth.revision,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            base.repo_id,
            revision=base.revision,
            controlnet=controlnet,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")

    def restyle(
        self,
        image: Any,
        depth: Any,
        prompt: str,
        seed: int,
        *,
        strength: float = 0.7,
        control_scale: float = 0.6,
        steps: int = 30,
    ) -> Any:
        """`image` and `depth` are PIL images of the same size; returns a PIL image."""
        import torch

        if self.pipe is None:
            self.load()
        width, height = image.size
        return self.pipe(
            prompt=prompt.strip(),
            negative_prompt=NEGATIVE_PROMPT,
            image=image,
            control_image=depth,
            strength=float(strength),
            controlnet_conditioning_scale=float(control_scale),
            num_inference_steps=int(steps),
            guidance_scale=7.0,
            width=width,
            height=height,
            generator=torch.Generator("cpu").manual_seed(int(seed)),
        ).images[0]


_CLASSES = {"sdxl-depth": SdxlDepthEditor}


def make_editor(key: str = models.DEFAULT_EDITOR) -> Editor:
    """A (not yet loaded) editor for a registry key; raises KeyError for unknown keys."""
    spec = models.editor_spec(key)
    return _CLASSES[spec.key](spec)
