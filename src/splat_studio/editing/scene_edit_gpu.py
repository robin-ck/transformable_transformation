"""Edit scene (task 4.4), GPU part: render views, edit them, fine-tune the Gaussians.

Runs inside the Modal container only (needs CUDA, torch, gsplat, diffusers); the plan,
cameras and bookkeeping live in `scene_edit.py`. See that module for the approach.

Rendering uses only the view-independent colour (spherical harmonics degree 0), the
same as the fine-tuning learns, so before and after renders compare like with like.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from splat_studio import models

from . import scene_edit as se
from .preview import intrinsics, viewmat

DEVICE = "cuda"
BACKGROUND = (31 / 255, 36 / 255, 48 / 255)  # the viewer's #1f2430
# Learning rates as gsplat's simple_trainer (means scale with the scene).
LR = {"means": 1.6e-4, "log_scales": 5e-3, "quats": 1e-3, "opacity_logits": 5e-2, "sh0": 2.5e-3}
SSIM_WEIGHT = 0.2


def _stack(v: np.ndarray, names: list[str]) -> np.ndarray:
    return np.stack([v[n] for n in names], axis=1).astype(np.float32)


def load_params(v: np.ndarray) -> dict[str, torch.Tensor]:
    """Trainable tensors from PLY fields (log scales, raw quaternions, opacity logits)."""
    arrays = {
        "means": _stack(v, ["x", "y", "z"]),
        "log_scales": _stack(v, ["scale_0", "scale_1", "scale_2"]),
        "quats": _stack(v, ["rot_0", "rot_1", "rot_2", "rot_3"]),
        "opacity_logits": np.asarray(v["opacity"], dtype=np.float32),
        "sh0": _stack(v, ["f_dc_0", "f_dc_1", "f_dc_2"]),
    }
    return {k: torch.tensor(a, device=DEVICE, requires_grad=True) for k, a in arrays.items()}


class Views:
    """Camera matrices for gsplat, one row per view."""

    def __init__(self, cams, size: int) -> None:
        self.size = size
        self.viewmats = torch.tensor(
            np.stack([viewmat(c) for c in cams]), dtype=torch.float32, device=DEVICE
        )
        self.Ks = torch.tensor(
            np.stack([intrinsics(c, size, size) for c in cams]), dtype=torch.float32, device=DEVICE
        )

    def __len__(self) -> int:
        return len(self.viewmats)


def render(p: dict[str, torch.Tensor], views: Views, i: int) -> torch.Tensor:
    """View `i` as an (H, W, 3) image in [0, 1], over the viewer's background colour."""
    from gsplat import rasterization

    colors, _alphas, _ = rasterization(
        means=p["means"],
        quats=p["quats"],
        scales=torch.exp(p["log_scales"]),
        opacities=torch.sigmoid(p["opacity_logits"]),
        colors=p["sh0"][:, None, :],
        viewmats=views.viewmats[i : i + 1],
        Ks=views.Ks[i : i + 1],
        width=views.size,
        height=views.size,
        sh_degree=0,
        backgrounds=torch.tensor([BACKGROUND], device=DEVICE),
    )
    return colors[0].clamp(0, 1)


class Editor:
    """InstructPix2Pix with the Instruct-NeRF2NeRF update: the edit starts from the
    (noised) current render and is conditioned on the original render."""

    def __init__(self, pipe=None, device: str = DEVICE, dtype=torch.float16) -> None:
        """Loads the pinned InstructPix2Pix; `pipe` (with `device`, `dtype`) is for tests."""
        from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline

        if pipe is None:
            spec = models.IMAGE_EDIT
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                spec.repo_id, revision=spec.revision, torch_dtype=dtype, safety_checker=None
            ).to(device)
        self.pipe, self.device, self.dtype = pipe, device, dtype
        self.pipe.set_progress_bar_config(disable=True)
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    @torch.no_grad()
    def embed(self, prompt: str) -> torch.Tensor:
        """Text embeddings in the order the guidance needs: (instruction, none, none)."""
        tok = self.pipe.tokenizer(
            [prompt, ""],
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        emb = self.pipe.text_encoder(tok.input_ids.to(self.device))[0]
        return torch.cat([emb[:1], emb[1:], emb[1:]])

    @torch.no_grad()
    def edit(
        self,
        current: torch.Tensor,
        original: torch.Tensor,
        emb: torch.Tensor,
        strength: float,
        seed: int,
    ) -> torch.Tensor:
        vae, unet, sched = self.pipe.vae, self.pipe.unet, self.scheduler

        def to_vae(img: torch.Tensor) -> torch.Tensor:
            return (img.permute(2, 0, 1)[None] * 2 - 1).to(self.device, self.dtype)

        # InstructPix2Pix conditions on the unscaled latent mode of the image.
        cond = vae.encode(to_vae(original)).latent_dist.mode()
        start = vae.encode(to_vae(current)).latent_dist.mode() * vae.config.scaling_factor
        sched.set_timesteps(se.EDIT_STEPS, device=self.device)
        timesteps = sched.timesteps[int(len(sched.timesteps) * (1 - strength)) :]
        gen = torch.Generator(self.device).manual_seed(int(seed))
        noise = torch.randn(start.shape, generator=gen, device=self.device, dtype=start.dtype)
        latents = sched.add_noise(start, noise, timesteps[:1])
        cond3 = torch.cat([cond, cond, torch.zeros_like(cond)])
        for t in timesteps:
            x = sched.scale_model_input(torch.cat([latents] * 3), t)
            pred = unet(torch.cat([x, cond3], dim=1), t, encoder_hidden_states=emb).sample
            text, image, uncond = pred.chunk(3)
            latents = sched.step(se.combine_guidance(text, image, uncond), t, latents).prev_sample
        img = vae.decode(latents / vae.config.scaling_factor).sample
        return ((img[0].permute(1, 2, 0).float() + 1) / 2).clamp(0, 1)


def _ssim():
    try:
        from fused_ssim import fused_ssim

        return lambda a, b: fused_ssim(a, b, padding="valid")
    except Exception:  # not built in this image: L1 alone still trains
        return None


def finetune(
    p: dict[str, torch.Tensor],
    mask: torch.Tensor,
    views: Views,
    targets: list[torch.Tensor],
    steps: int,
    scene_scale: float,
    rng: np.random.Generator,
) -> float:
    """Fit the masked Gaussians to the edited views; returns the last loss."""
    opt = torch.optim.Adam(
        [{"params": [p[k]], "lr": LR[k] * (scene_scale if k == "means" else 1.0)} for k in LR],
        eps=1e-15,
    )
    ssim = _ssim()
    loss = torch.zeros(())
    for _ in range(steps):
        i = int(rng.integers(len(views)))
        img, target = render(p, views, i), targets[i]
        loss = F.l1_loss(img, target)
        if ssim is not None:
            a, b = img.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
            loss = (1 - SSIM_WEIGHT) * loss + SSIM_WEIGHT * (1 - ssim(a, b))
        opt.zero_grad(set_to_none=False)
        loss.backward()
        for t in p.values():  # Gaussians outside the region keep their values
            if t.grad is not None:
                t.grad[~mask] = 0
        opt.step()
    return float(loss.detach())


def to_uint8(img: torch.Tensor) -> np.ndarray:
    return (img.detach().cpu().numpy() * 255).round().astype(np.uint8)


def run_edit(
    v: np.ndarray,
    prompt: str,
    region: se.Region | None = None,
    seed: int = 0,
    on_stage: Callable[[str, float], None] | None = None,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """Edit the scene `v` (PLY fields); returns the new fields, stats and sample images."""
    stage_t = [time.time()]

    def stage(name: str) -> None:
        if on_stage:
            on_stage(name, time.time() - stage_t[0])
        stage_t[0] = time.time()

    rng = np.random.default_rng(seed)
    positions = _stack(v, ["x", "y", "z"])
    mask_np = se.check_region(positions, region)
    cams = se.orbit_cameras(positions, region)
    _centre, scene_radius = se.scene_centre_radius(positions)
    scale = region.radius if region else scene_radius

    p = load_params(v)
    mask = torch.tensor(mask_np, device=DEVICE)
    views = Views(cams, se.VIEW_SIZE)
    with torch.no_grad():
        originals = [render(p, views, i) for i in range(len(views))]
    stage("render")

    editor = Editor()
    emb = editor.embed(prompt.strip())
    stage("load_model")

    targets = list(originals)
    losses = []
    for r in range(se.ROUNDS):
        with torch.no_grad():
            current = originals if r == 0 else [render(p, views, i) for i in range(len(views))]
        for i in range(len(views)):
            strength, view_seed = se.strength_for(r, rng), seed * 7919 + r * 1000 + i
            targets[i] = editor.edit(current[i], originals[i], emb, strength, view_seed)
        stage(f"edit_round_{r + 1}")
        losses.append(finetune(p, mask, views, targets, se.STEPS_PER_ROUND, scale, rng))
        stage(f"train_round_{r + 1}")

    with torch.no_grad():
        after = render(p, views, 0)
    images = {
        "before.png": to_uint8(originals[0]),
        "edited_view.png": to_uint8(targets[0]),
        "after.png": to_uint8(after),
    }
    out = se.write_params(
        v,
        p["means"].detach().cpu().numpy(),
        p["log_scales"].detach().cpu().numpy(),
        p["quats"].detach().cpu().numpy(),
        p["opacity_logits"].detach().cpu().numpy(),
        p["sh0"].detach().cpu().numpy(),
        mask_np,
    )
    stats = {
        **se.plan_summary(len(views)),
        "edited_gaussians": int(mask_np.sum()),
        "losses": [round(x, 4) for x in losses],
    }
    return out, stats, images
