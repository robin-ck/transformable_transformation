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


def submit_reconstruction(job_id: str, steps: int = 7000) -> str:
    return spawn("reconstruct", job_id, steps=steps)


def submit_object(job_id: str, prompt: str, seed: int = 0) -> str:
    return spawn("generate_object", job_id, prompt, seed=seed)
