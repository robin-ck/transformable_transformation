"""Edit scene on the GPU (task 4.4): iterative dataset update with InstructPix2Pix.

Runs inside the Modal container (needs torch, gsplat and diffusers); the Space never
imports it. The steps follow Instruct-NeRF2NeRF (Haque et al., 2023):

1. Render `n_views` views around the user's view (`scene_edit.orbit_cameras`). These
   renders are the "originals" that InstructPix2Pix is conditioned on.
2. Edit every view once from its original, so training starts from an edited set.
3. Fine-tune the Gaussians on the edited views. Every `edit_every` steps, render one
   view from the current scene, add noise, and let InstructPix2Pix denoise it again
   (conditioned on that view's original); the result replaces that view's target.
   Views drift towards one edit that is consistent in 3D.

With a region, only the selected Gaussians get gradients, and each edited target
keeps the original render outside the region's footprint in that view.
No densification: the number of Gaussians stays the same.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from splat_studio.reconstruction.export import read_ply, to_splat_bytes, write_ply

from . import scene_edit
from .preview import ViewCamera, intrinsics, output_size, viewmat

EDIT_PIXELS = 512 * 512  # InstructPix2Pix (SD 1.5) works at about 512 x 512


@dataclass
class EditSettings:
    n_views: int = 40
    steps: int = 1500
    edit_every: int = 10
    denoise_steps: int = 20
    guidance: float = 7.5  # text
    image_guidance: float = 1.5  # how close to the original view
    noise_range: tuple[float, float] = (0.70, 0.98)  # IN2N's re-edit noise levels
    yaw_deg: float = 30.0
    pitch_deg: float = 10.0
    seed: int = 0


@dataclass
class EditResult:
    ply: Path
    splat: Path
    before: Path
    after: Path
    example: Path
    num_gaussians: int
    num_selected: int | None
    size: tuple[int, int]
    stats: dict = field(default_factory=dict)


def _t(a, device="cuda") -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)


class Splats:
    """Trainable Gaussians and a renderer for `ViewCamera`s."""

    def __init__(self, arrays: dict[str, np.ndarray], width: int, height: int) -> None:
        self.p = {k: torch.nn.Parameter(_t(v)) for k, v in arrays.items()}
        self.width, self.height = width, height
        self.degree = scene_edit.sh_degree(1 + arrays["shN"].shape[1])

    def render(self, cam: ViewCamera, mode: str = "RGB", colors=None):
        from gsplat import rasterization

        p = self.p
        sh = colors is None
        out, alpha, _ = rasterization(
            means=p["means"],
            quats=p["quats"],
            scales=torch.exp(p["scales"]),
            opacities=torch.sigmoid(p["opacities"]),
            colors=torch.cat([p["sh0"], p["shN"]], dim=1) if sh else colors,
            sh_degree=self.degree if sh else None,
            viewmats=_t(viewmat(cam))[None],
            Ks=_t(intrinsics(cam, self.width, self.height))[None],
            width=self.width,
            height=self.height,
            render_mode=mode,
        )
        return out[0], alpha[0, ..., 0]

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.p.items()}


class Editor:
    """InstructPix2Pix with Instruct-NeRF2NeRF's partial re-noising of the current render."""

    def __init__(self, repo_id: str, revision: str, prompt: str, settings: EditSettings) -> None:
        from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            repo_id,
            revision=revision,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        ).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        self.unet, self.vae = pipe.unet, pipe.vae
        self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.s = settings
        tok = pipe.tokenizer

        def encode_text(text: str) -> torch.Tensor:
            ids = tok(text, padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids.to("cuda")  # fmt: skip
            return pipe.text_encoder(ids)[0]

        with torch.no_grad():
            text, empty = encode_text(prompt), encode_text("")
        # Order as in the IP2P pipeline: (text + image), (image only), (neither).
        self.embeds = torch.cat([text, empty, empty])
        self.sf = self.vae.config.scaling_factor

    @torch.no_grad()
    def __call__(
        self, current: torch.Tensor, original: torch.Tensor, noise: float, gen
    ) -> torch.Tensor:
        """Edit `current` (H,W,3 in [0,1]); conditioned on `original`; returns H,W,3."""

        def encode(img: torch.Tensor, sample: bool) -> torch.Tensor:
            x = img.permute(2, 0, 1)[None].half() * 2 - 1
            d = self.vae.encode(x).latent_dist
            return d.sample(generator=gen) if sample else d.mode()

        lat = encode(current, True) * self.sf
        cond = encode(original, False)  # IP2P conditions on unscaled image latents
        self.scheduler.set_timesteps(self.s.denoise_steps, device="cuda")
        ts = self.scheduler.timesteps
        ts = ts[max(0, int(round(len(ts) * (1 - noise)))) :]
        eps0 = torch.randn(lat.shape, generator=gen, device="cuda", dtype=lat.dtype)
        lat = self.scheduler.add_noise(lat, eps0, ts[:1])
        image_cond = torch.cat([cond, cond, torch.zeros_like(cond)])
        for t in ts:
            x = torch.cat([lat] * 3)
            x = torch.cat([self.scheduler.scale_model_input(x, t), image_cond], dim=1)
            e_full, e_img, e_none = self.unet(x, t, encoder_hidden_states=self.embeds).sample.chunk(
                3
            )
            eps = (
                e_none
                + self.s.guidance * (e_full - e_img)
                + self.s.image_guidance * (e_img - e_none)
            )
            lat = self.scheduler.step(eps, t, lat).prev_sample
        img = self.vae.decode(lat / self.sf).sample[0]
        return ((img.float() + 1) / 2).clamp(0, 1).permute(1, 2, 0)


def _region_mask(splats: Splats, cam: ViewCamera, selected: torch.Tensor) -> torch.Tensor:
    """Where the selected Gaussians make up the view, as a soft 0..1 mask (H, W, 1).

    The render of the selection flag is opacity-weighted, so divide by the coverage to
    get the selected share of each pixel; shares from 0.2 to 0.6 ramp from 0 to 1.
    """
    share, alpha = splats.render(cam, "RGB", colors=selected[:, None])
    share = share[..., :1] / alpha[..., None].clamp_min(1e-6)
    return ((share - 0.2) / 0.4).clamp(0, 1)


def _ssim_loss():
    try:
        from fused_ssim import fused_ssim

        def loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            x = a.permute(2, 0, 1)[None].contiguous()
            y = b.permute(2, 0, 1)[None].contiguous()
            return 1 - fused_ssim(x, y, padding="valid")

        return loss
    except Exception:  # fall back to L1 only
        return None


def _save(img: torch.Tensor, path: Path) -> Path:
    from PIL import Image

    Image.fromarray((img.detach().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(
        path
    )
    return path


def run_edit(
    ply_path: Path,
    anchor: ViewCamera,
    prompt: str,
    out_dir: Path,
    *,
    model: tuple[str, str],
    region: scene_edit.Region | None = None,
    settings: EditSettings | None = None,
    on_stage: Callable[[str, float], None] | None = None,
) -> EditResult:
    """Edit the scene in `ply_path` around the `anchor` view; writes results to `out_dir`."""
    from splat_studio.reconstruction import errors

    s = settings or EditSettings()
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(s.seed)
    torch.manual_seed(s.seed)
    gen = torch.Generator("cuda").manual_seed(s.seed)
    clock = [time.time()]

    def stage(name: str) -> None:
        if on_stage:
            on_stage(name, time.time() - clock[0])
        clock[0] = time.time()

    v = read_ply(ply_path)
    width, height = output_size(anchor.aspect, EDIT_PIXELS)
    splats = Splats(scene_edit.split_ply(v), width, height)
    stage("load_scene")

    # Where the user is looking: the median depth of the Gaussian centres in view
    # (a depth render would be pulled forward by haze; see scene_edit.select_region).
    means = splats.p["means"].detach().cpu().numpy()
    distance = scene_edit.view_depth(means, anchor, width, height)
    with torch.no_grad():
        _, alpha = splats.render(anchor)
    if distance is None or (alpha.cpu().numpy() >= 0.5).mean() < 0.05:
        raise errors.edit_empty()
    pivot = scene_edit.pivot_point(anchor, distance)
    cams = scene_edit.orbit_cameras(anchor, pivot, s.n_views, s.yaw_deg, s.pitch_deg, seed=s.seed)

    selected = None
    if region is not None:
        sel = scene_edit.select_region(means, anchor, width, height, region)
        if sel.sum() < 50:
            raise errors.region_empty()
        selected = _t(sel.astype(np.float32))
        for p in splats.p.values():  # only the selected Gaussians may change
            mask = selected.view((-1,) + (1,) * (p.dim() - 1))
            p.register_hook(lambda g, m=mask: g * m)

    with torch.no_grad():
        originals = [splats.render(c)[0].clamp(0, 1) for c in cams]
        masks = None
        if selected is not None:
            masks = [_region_mask(splats, c, selected) for c in cams]
    stage("render_views")

    editor = Editor(*model, prompt, s)
    stage("load_editor")

    def target_for(i: int, current: torch.Tensor, noise: float) -> torch.Tensor:
        edited = editor(current, originals[i], noise, gen)
        if edited.shape[:2] != current.shape[:2]:
            edited = F.interpolate(edited.permute(2, 0, 1)[None], size=current.shape[:2],
                                   mode="bilinear")[0].permute(1, 2, 0)  # fmt: skip
        if masks is not None:
            edited = masks[i] * edited + (1 - masks[i]) * originals[i]
        return edited

    targets = [target_for(i, originals[i], s.noise_range[1]) for i in range(len(cams))]
    _save(targets[0], out_dir / "edit_example.png")
    stage("first_edits")

    scale = max(distance, 1e-3)
    p = splats.p
    opt = torch.optim.Adam(
        [
            {"params": [p["means"]], "lr": 1.6e-4 * scale},
            {"params": [p["scales"]], "lr": 5e-3},
            {"params": [p["quats"]], "lr": 1e-3},
            {"params": [p["opacities"]], "lr": 5e-2},
            {"params": [p["sh0"]], "lr": 2.5e-3},
            {"params": [p["shN"]], "lr": 2.5e-3 / 20},
        ],
        eps=1e-15,
    )
    ssim = _ssim_loss()
    order = list(range(len(cams)))
    next_edit = 0
    edits = 0
    for step in range(s.steps):
        if step and step % s.edit_every == 0:
            j = order[next_edit % len(order)]
            next_edit += 1
            with torch.no_grad():
                current = splats.render(cams[j])[0].clamp(0, 1)
            targets[j] = target_for(j, current, random.uniform(*s.noise_range))
            edits += 1
        i = random.randrange(len(cams))
        img = splats.render(cams[i])[0]
        loss = F.l1_loss(img, targets[i])
        if ssim is not None:
            loss = 0.8 * loss + 0.2 * ssim(img, targets[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    stage("train")

    arrays = splats.arrays()
    arrays["quats"] /= np.linalg.norm(arrays["quats"], axis=1, keepdims=True) + 1e-12
    edited = scene_edit.join_ply(v, arrays)
    ply, splat = out_dir / "scene.ply", out_dir / "scene.splat"
    write_ply(ply, edited)
    splat.write_bytes(to_splat_bytes(edited))
    with torch.no_grad():
        after = splats.render(anchor)[0]
    before = _save(originals[0], out_dir / "before.png")
    after_path = _save(after, out_dir / "after.png")
    stage("export")
    return EditResult(
        ply=ply,
        splat=splat,
        before=before,
        after=after_path,
        example=out_dir / "edit_example.png",
        num_gaussians=len(v),
        num_selected=int(selected.sum().item()) if selected is not None else None,
        size=(width, height),
        stats={
            "views": len(cams),
            "steps": s.steps,
            "re_edits": edits,
            "pivot_distance": round(distance, 3),
            "mask_coverage": round(float(masks[0].mean()), 3) if masks else None,
            "loss": round(float(loss), 4),
        },  # fmt: skip
    )
