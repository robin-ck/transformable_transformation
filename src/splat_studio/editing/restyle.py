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
from splat_studio.editing.instructions import DEFAULTS

NEGATIVE_PROMPT = DEFAULTS["negative"]  # a future may bring its own (`negative_prompt`)


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
        negative_prompt: str | None = None,
    ) -> Any:
        """`image` and `depth` are PIL images of the same size; returns a PIL image.
        `negative_prompt` None uses `NEGATIVE_PROMPT`; "" uses none."""
        import torch

        if self.pipe is None:
            self.load()
        width, height = image.size
        return self.pipe(
            prompt=prompt.strip(),
            negative_prompt=NEGATIVE_PROMPT if negative_prompt is None else negative_prompt,
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


def allow_sdpa_without_gqa(functional: Any) -> None:
    """Let diffusers 0.35 run on torch 2.4 (the gsplat image's torch).

    diffusers' native attention always passes `enable_gqa` to
    `scaled_dot_product_attention`, which only torch 2.5+ accepts. `functional` is
    `torch.nn.functional`; the wrapper drops `enable_gqa=False` and refuses True.
    Does nothing when the installed torch already accepts it, or when already wrapped.
    """
    import inspect

    sdpa = functional.scaled_dot_product_attention
    if getattr(sdpa, "_drops_gqa", False):
        return
    try:
        if "enable_gqa" in inspect.signature(sdpa).parameters:
            return
    except (TypeError, ValueError):  # a builtin without a signature: check the docstring
        if "enable_gqa" in (sdpa.__doc__ or ""):
            return

    def wrapped(*args: Any, enable_gqa: bool = False, **kwargs: Any) -> Any:
        if enable_gqa:
            raise NotImplementedError("enable_gqa=True needs torch 2.5 or newer")
        return sdpa(*args, **kwargs)

    wrapped._drops_gqa = True  # type: ignore[attr-defined]
    functional.scaled_dot_product_attention = wrapped


class FluxKontextEditor:
    """FLUX.1 Kontext dev: edits the image as the prompt instructs and keeps the rest.

    Unlike SDXL img2img it needs no strength: an instruction such as "remove the chairs"
    changes the chairs only. The depth image is not used; `strength`, `control_scale`
    and `negative_prompt` are accepted and ignored, so callers can pass the same options
    to every editor.
    """

    def __init__(self, spec: models.EditorSpec) -> None:
        self.spec = spec
        self.key, self.label = spec.key, spec.label
        self.licence, self.uses_depth = spec.licence, spec.uses_depth
        self.pipe = None

    def load(self) -> None:
        import torch
        from diffusers import FluxKontextPipeline

        allow_sdpa_without_gqa(torch.nn.functional)
        (model,) = self.spec.models
        self.pipe = FluxKontextPipeline.from_pretrained(
            model.repo_id, revision=model.revision, torch_dtype=torch.bfloat16
        ).to("cuda")

    def restyle(
        self,
        image: Any,
        depth: Any,
        prompt: str,
        seed: int,
        *,
        steps: int = 28,
        guidance: float = 2.5,
        **_: Any,
    ) -> Any:
        """Returns a PIL image the size of `image` (Kontext works near one megapixel)."""
        import torch
        from PIL import Image

        if self.pipe is None:
            self.load()
        out = self.pipe(
            image=image,
            prompt=prompt.strip(),
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            generator=torch.Generator("cpu").manual_seed(int(seed)),
        ).images[0]
        if out.size != image.size:
            out = out.resize(image.size, Image.LANCZOS)
        return out


# --- add-futures-editors ------------------------------------------------------------


def canny_control(image: Any, low: int = 100, high: int = 200) -> Any:
    """Canny edges of `image` as a white-on-black RGB PIL image of the same size (the
    union ControlNet's "canny" mode). Needs opencv; imported here so the Space need not."""
    import cv2
    import numpy as np
    from PIL import Image

    grey = np.asarray(image.convert("L"))
    edges = cv2.Canny(grey, int(low), int(high))
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def lightning_adapter(steps: int) -> str:
    """Which distilled LoRA to use for `steps`: the 4-step one up to 4 steps, else 8-step."""
    return "4step" if int(steps) <= 4 else "8step"


# Lightning LoRA files per adapter name (in the pinned repos of models.py).
SDXL_LIGHTNING_LORAS = {
    "4step": "sdxl_lightning_4step_lora.safetensors",
    "8step": "sdxl_lightning_8step_lora.safetensors",
}
QWEN_LIGHTNING_LORAS = {
    "4step": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "8step": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
}
# FlowMatch scheduler the Qwen Lightning LoRAs were distilled with (shift 3, exponential
# time shift), from ModelTC/Qwen-Image-Lightning's generate_with_diffusers.py.
QWEN_LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256,
    "base_shift": 1.0986122886681098,  # log(3)
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 1.0986122886681098,  # log(3)
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
UNION_MODES = {"depth": 1, "canny": 3}  # xinsir/controlnet-union-sdxl-1.0 control types


def multiple_of(size: tuple[int, int], m: int = 16) -> tuple[int, int]:
    """`size` rounded down to multiples of `m` (transformer editors need it), at least m."""
    w, h = size
    return max(m, w // m * m), max(m, h // m * m)


class _Base:
    """Shared fields of every editor; `last_info` holds what the last `restyle` returned
    besides the image (e.g. Step1X's rewritten prompt), for the worker's run info."""

    def __init__(self, spec: models.EditorSpec) -> None:
        self.spec = spec
        self.key, self.label = spec.key, spec.label
        self.licence, self.uses_depth = spec.licence, spec.uses_depth
        self.pipe = None
        self.last_info: dict = {}

    @staticmethod
    def _seeded(seed: int):
        import torch

        return torch.Generator("cpu").manual_seed(int(seed))

    @staticmethod
    def _back_to(out: Any, image: Any) -> Any:
        from PIL import Image

        return out if out.size == image.size else out.resize(image.size, Image.LANCZOS)


class SdxlUnionEditor(_Base):
    """SDXL img2img with the union ControlNet (depth + canny of the render) and the
    SDXL-Lightning LoRAs: the default editor's job, with a tighter layout and 4-8 steps.
    Lightning runs without guidance, so the negative prompt has no effect."""

    def load(self) -> None:
        import torch
        from diffusers import (
            ControlNetUnionModel,
            EulerDiscreteScheduler,
            StableDiffusionXLControlNetUnionImg2ImgPipeline,
        )

        base, union, lightning = self.spec.models
        controlnet = ControlNetUnionModel.from_pretrained(
            union.repo_id, revision=union.revision, torch_dtype=torch.float16
        )
        pipe = StableDiffusionXLControlNetUnionImg2ImgPipeline.from_pretrained(
            base.repo_id,
            revision=base.revision,
            controlnet=controlnet,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        for name, weight in SDXL_LIGHTNING_LORAS.items():
            pipe.load_lora_weights(
                lightning.repo_id, revision=lightning.revision, weight_name=weight,
                adapter_name=name,
            )  # fmt: skip
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe = pipe

    def restyle(
        self,
        image: Any,
        depth: Any,
        prompt: str,
        seed: int,
        *,
        strength: float = 0.7,
        control_scale: float = 0.6,
        steps: int = 8,
        **_: Any,
    ) -> Any:
        if self.pipe is None:
            self.load()
        self.pipe.set_adapters([lightning_adapter(steps)])
        width, height = image.size
        self.last_info = {"control_modes": list(UNION_MODES)}
        return self.pipe(
            prompt=prompt.strip(),
            image=image,
            control_image=[depth, canny_control(image)],
            control_mode=[UNION_MODES["depth"], UNION_MODES["canny"]],
            strength=float(strength),
            controlnet_conditioning_scale=float(control_scale),
            num_inference_steps=int(steps),
            guidance_scale=0.0,
            width=width,
            height=height,
            generator=self._seeded(seed),
        ).images[0]


class QwenEditEditor(_Base):
    """Qwen-Image-Edit-2511 with the Lightning LoRAs (4 or 8 steps, no true CFG)."""

    def load(self) -> None:
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

        allow_sdpa_without_gqa(torch.nn.functional)
        model, lightning = self.spec.models
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model.repo_id,
            revision=model.revision,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(QWEN_LIGHTNING_SCHEDULER),
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        for name, weight in QWEN_LIGHTNING_LORAS.items():
            pipe.load_lora_weights(
                lightning.repo_id, revision=lightning.revision, weight_name=weight,
                adapter_name=name,
            )  # fmt: skip
        self.pipe = pipe

    def restyle(self, image: Any, depth: Any, prompt: str, seed: int, *, steps: int = 8,
                **_: Any) -> Any:  # fmt: skip
        if self.pipe is None:
            self.load()
        self.pipe.set_adapters([lightning_adapter(steps)])
        self.last_info = {}
        out = self.pipe(
            image=[image],
            prompt=prompt.strip(),
            negative_prompt=" ",
            true_cfg_scale=1.0,
            num_inference_steps=int(steps),
            generator=self._seeded(seed),
        ).images[0]
        return self._back_to(out, image)


class Flux2KleinEditor(_Base):
    """FLUX.2 klein 4B: step-distilled generation and editing in one small model."""

    def load(self) -> None:
        import torch
        from diffusers import Flux2KleinPipeline

        allow_sdpa_without_gqa(torch.nn.functional)
        (model,) = self.spec.models
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model.repo_id, revision=model.revision, torch_dtype=torch.bfloat16
        ).to("cuda")

    def restyle(self, image: Any, depth: Any, prompt: str, seed: int, *, steps: int = 4,
                **_: Any) -> Any:  # fmt: skip
        if self.pipe is None:
            self.load()
        width, height = multiple_of(image.size)
        self.last_info = {}
        out = self.pipe(
            image=[image],
            prompt=prompt.strip(),
            width=width,
            height=height,
            guidance_scale=1.0,
            num_inference_steps=int(steps),
            generator=self._seeded(seed),
        ).images[0]
        return self._back_to(out, image)


class Step1XEditor(_Base):
    """Step1X-Edit v1.2: its MLLM rewrites the instruction first ("thinking"); the
    rewritten prompt is kept in `last_info`. Reflection (several edits) stays off."""

    def load(self) -> None:
        import torch
        from diffusers import Step1XEditPipelineV1P2

        allow_sdpa_without_gqa(torch.nn.functional)
        (model,) = self.spec.models
        self.pipe = Step1XEditPipelineV1P2.from_pretrained(
            model.repo_id, revision=model.revision, torch_dtype=torch.bfloat16
        ).to("cuda")

    def restyle(self, image: Any, depth: Any, prompt: str, seed: int, *, steps: int = 28,
                guidance: float = 6.0, **_: Any) -> Any:  # fmt: skip
        if self.pipe is None:
            self.load()
        out = self.pipe(
            image=image,
            prompt=prompt.strip(),
            num_inference_steps=int(steps),
            true_cfg_scale=float(guidance),
            generator=self._seeded(seed),
            enable_thinking_mode=True,
            enable_reflection_mode=False,
        )
        rewritten = getattr(out, "reformat_prompt", None)
        if isinstance(rewritten, list | tuple):
            rewritten = rewritten[0] if rewritten else None
        self.last_info = {"editor_prompt_rewritten": rewritten} if rewritten else {}
        images = getattr(out, "final_images", None) or out.images
        return self._back_to(images[0], image)


_CLASSES = {
    "sdxl-depth": SdxlDepthEditor,
    "flux-kontext": FluxKontextEditor,
    "sdxl-union": SdxlUnionEditor,
    "qwen-edit": QwenEditEditor,
    "flux2-klein": Flux2KleinEditor,
    "step1x-edit": Step1XEditor,
}


def make_editor(key: str = models.DEFAULT_EDITOR) -> Editor:
    """A (not yet loaded) editor for a registry key; raises KeyError for unknown keys."""
    spec = models.editor_spec(key)
    return _CLASSES[spec.key](spec)
