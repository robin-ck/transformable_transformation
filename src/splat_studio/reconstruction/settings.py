"""Settings for a new scene (scene-settings): training steps, video frames, image size.

The user picks a preset or moves the sliders on the New scene tab. The Space checks the
values before it uploads anything, and `pipeline.run` checks them again on the GPU
worker before any work, so a bad or too costly setting never reaches the trainer.
"""

from __future__ import annotations

from .cost import estimate_cost

# The settings the user may change, with their allowed range.
LIMITS: dict[str, tuple[int, int]] = {
    "steps": (1000, 30000),  # gsplat training steps (gsplat's own default is 30,000)
    "frames": (60, 300),  # about this many frames are taken from a video
    "max_side": (800, 1600),  # long side of the training images, in pixels
}
# "Standard" is what every scene used before this change (MVP task 2.4). "Fast" and
# "High" are not measured yet; their times below are estimates.
PRESETS: dict[str, dict[str, int]] = {
    "Fast": {"steps": 3000, "frames": 120, "max_side": 1200},
    "Standard": {"steps": 7000, "frames": 180, "max_side": 1600},
    "High": {"steps": 15000, "frames": 240, "max_side": 1600},
}
DEFAULT_PRESET = "Standard"


def check_settings(settings: dict | None) -> dict[str, int]:
    """The full settings: the default preset, overridden by `settings`. Raises ValueError."""
    out = dict(PRESETS[DEFAULT_PRESET])
    for key, value in (settings or {}).items():
        if key not in LIMITS:
            raise ValueError(f"unknown scene setting {key!r}")
        lo, hi = LIMITS[key]
        whole = isinstance(value, int) or (isinstance(value, float) and value.is_integer())
        if isinstance(value, bool) or not whole:
            raise ValueError(f"{key} must be a whole number, got {value!r}")
        n = int(value)
        if not lo <= n <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}, got {n}")
        out[key] = n
    return out


def preset_of(settings: dict) -> str | None:
    """The preset name whose values `settings` matches, else None."""
    full = check_settings(settings)
    return next((name for name, p in PRESETS.items() if p == full), None)


# --- time and cost estimate ---------------------------------------------------------

GPU, CPU_CORES = "A10G", 4.0  # keep in sync with TRAIN_GPU, CPU_CORES in modal_app/app.py
# The estimate is an upper bound: the slowest stage times of the three Standard runs so
# far (videos of 56 to 84 s, 180 frames at 1,600 px, 7,000 steps, A10G), which took 589,
# 629 and 866 s in total:
#   20261113-132210-3749ae  frames 10 s, poses 503 s, training 337 s, 1.56 M Gaussians
#   20261115-184011-7e524c  frames  9 s, poses 317 s, training 237 s, 0.54 M Gaussians
#   20261115-213245-7a7de6  frames 13 s, poses 248 s, training 359 s, 2.38 M Gaussians
# Training time follows the number of Gaussians the scene grows, so it depends on the scene.
# sharpest-frames takes 3x the candidates; locally a 2 min 1080p video went from 3.4 to 5.1 s.
FRAMES_S = 25.0  # was 15 s (a 2 min video took about 18 s)
POSES_S, POSES_FRAMES = 503.0, 180
# COLMAP's mapper grows faster than linearly with the number of images; the power is a
# guess. Photo sets use exhaustive matching (every pair), so they may take longer.
POSES_POWER = 1.5
TRAIN_STEP_S = 359.0 / 7000  # one step at 1,600 px while the scene still grows
GROWN_AFTER = 7000  # steps after this run on more Gaussians (gsplat adds them until 15k)
GROWN_FACTOR = 1.5  # how much slower those steps are; a guess, not measured
SAVING_S = 30.0  # export, upload and downloading the inputs
MAX_S = 45 * 60  # refuse longer estimates; the job's hard timeout is 60 min


def estimate_breakdown(settings: dict | None, photos: int | None = None) -> dict[str, float]:
    """Rough GPU seconds per stage. `photos`: the number of photos, or None for a video."""
    s = check_settings(settings)
    images = photos or s["frames"]
    pixels = (s["max_side"] / LIMITS["max_side"][1]) ** 2
    early = min(s["steps"], GROWN_AFTER)
    late = s["steps"] - early
    return {
        "frames": 0.0 if photos else FRAMES_S,
        "poses": POSES_S * (images / POSES_FRAMES) ** POSES_POWER,
        "training": TRAIN_STEP_S * pixels * (early + GROWN_FACTOR * late),
        "saving": SAVING_S,
    }


def estimate_seconds(settings: dict | None, photos: int | None = None) -> float:
    return sum(estimate_breakdown(settings, photos).values())


def estimate_usd(seconds: float) -> float:
    return estimate_cost(GPU, seconds, CPU_CORES)


def check_time(settings: dict | None, photos: int | None = None) -> float:
    """The estimated seconds; raises ValueError if they exceed `MAX_S`."""
    seconds = estimate_seconds(settings, photos)
    if seconds > MAX_S:
        fewer = "upload fewer photos" if photos else "lower the frames"
        raise ValueError(
            f"These settings would take about {seconds / 60:.0f} min on the GPU; the limit "
            f"for one scene is {MAX_S // 60} min. Lower the training steps or {fewer}."
        )
    return seconds
