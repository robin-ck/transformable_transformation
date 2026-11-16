"""Thin client the Space uses to start and check Modal GPU jobs.

`modal` is imported lazily so the rest of the package (and tests) work without it.
"""

from __future__ import annotations

from typing import Any

APP_NAME = "splat-studio"


def _modal() -> Any:
    import modal

    return modal


def spawn(function_name: str, *args: Any, **kwargs: Any) -> str:
    """Start a deployed Modal function in the background; returns its call id."""
    fn = _modal().Function.from_name(APP_NAME, function_name)
    return fn.spawn(*args, **kwargs).object_id


def spawn_method(class_name: str, method: str, *args: Any, **kwargs: Any) -> str:
    """Start a method of a deployed Modal class in the background; returns its call id."""
    cls = _modal().Cls.from_name(APP_NAME, class_name)
    return getattr(cls(), method).spawn(*args, **kwargs).object_id


def poll(call_id: str) -> tuple[str, dict | None]:
    """Return ("running", None) or ("finished", result) without blocking.

    Errors from the job itself (including a hit runtime cap, FunctionTimeoutError) are raised.
    """
    modal = _modal()
    call = modal.FunctionCall.from_id(call_id)
    # FunctionTimeoutError subclasses modal's TimeoutError, so it must be re-raised first.
    function_timeout = getattr(modal.exception, "FunctionTimeoutError", ())
    try:
        return "finished", call.get(timeout=0)
    except function_timeout:
        raise
    except (TimeoutError, modal.exception.TimeoutError):
        return "running", None


def submit_reconstruction(job_id: str, settings: dict | None = None) -> str:
    """`settings` override the default preset (`reconstruction.settings.LIMITS`)."""
    return spawn("reconstruct", job_id, settings=settings)


def submit_object(
    job_id: str, prompt: str, seed: int = 0, reference_path: str | None = None
) -> str:
    """`reference_path` (a dataset path) makes the worker skip text-to-image."""
    kwargs = {"reference_path": reference_path} if reference_path else {}
    return spawn_method("ObjectGenerator", "generate", job_id, prompt, seed=seed, **kwargs)


def submit_edit(
    job_id: str,
    scene_id: str,
    number: int,
    camera: dict,
    prompt: str,
    region: list[float] | None = None,
    seed: int = 0,
    budget: dict | None = None,
) -> str:
    """`budget` overrides the edit's work settings (`editing.scene_edit.BUDGET_LIMITS`)."""
    kwargs = {"budget": budget} if budget else {}
    return spawn_method(
        "SceneEditor", "edit", job_id, scene_id, number, camera, prompt,
        region=region, seed=seed, **kwargs,
    )  # fmt: skip


def submit_preview(
    job_id: str, scene_path: str, camera: dict, prompt: str, seed: int = 0, strength: float = 0.7
) -> str:
    return spawn_method(
        "PreviewRenderer",
        "render",
        job_id,
        scene_path,
        camera,
        prompt,
        seed=seed,
        strength=strength,
    )


def submit_image(
    job_id: str,
    prompt: str,
    seed: int = 0,
    source_path: str | None = None,
    strength: float = 0.7,
    aspect: float = 1.0,
) -> str:
    """Text-to-image, or image-to-image when `source_path` (a dataset path) is given."""
    return spawn_method(
        "ImageStudio",
        "generate",
        job_id,
        prompt,
        seed=seed,
        source_path=source_path,
        strength=strength,
        aspect=aspect,
    )
