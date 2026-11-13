"""Readable failure reasons for reconstruction jobs."""

from __future__ import annotations

from dataclasses import dataclass

DETAIL_CHARS = 2000

FILMING_TIP = (
    "Tip: move slowly around the subject, keep it in view, overlap each shot by about "
    "two thirds with the previous one, avoid motion blur, and avoid shiny or blank surfaces."
)


@dataclass
class ReconstructionError(Exception):
    """A failure the user can understand and act on."""

    code: str
    message: str
    tip: str = ""
    detail: str = ""  # technical log tail for debugging; kept in run.json, not shown in the UI

    def __str__(self) -> str:
        return f"{self.message} {self.tip}".strip()

    def to_dict(self) -> dict[str, str]:
        out = {"code": self.code, "message": self.message, "tip": self.tip}
        if self.detail:
            out["detail"] = self.detail
        return out


def too_few_frames(n: int, minimum: int) -> ReconstructionError:
    return ReconstructionError(
        "too_few_frames",
        f"Only {n} usable frames were extracted; at least {minimum} are needed.",
        "Tip: upload a longer video (at least 10 seconds) or more photos.",
    )


def no_poses(detail: str = "") -> ReconstructionError:
    return ReconstructionError(
        "pose_failed",
        "Camera positions could not be recovered from your footage.",
        FILMING_TIP,
        detail[-DETAIL_CHARS:],
    )


def pose_tool_failed(step: str, detail: str) -> ReconstructionError:
    """COLMAP itself crashed or was misconfigured: our fault, not the footage."""
    return ReconstructionError(
        "pose_tool_failed",
        f"The camera pose step (COLMAP {step}) failed with an internal error. "
        "This is a problem on our side, not with your footage.",
        "Please try again later or report the job ID.",
        detail[-DETAIL_CHARS:],
    )


def low_registration(registered: int, total: int, min_ratio: float) -> ReconstructionError:
    return ReconstructionError(
        "pose_partial",
        f"Camera positions were found for only {registered} of {total} frames "
        f"(at least {min_ratio:.0%} needed), so the scene would be incomplete.",
        FILMING_TIP,
    )


def training_failed(detail: str) -> ReconstructionError:
    return ReconstructionError(
        "training_failed",
        "Building the splat scene failed on the GPU. Please try again later.",
        f"Technical detail: {detail[-300:]}" if detail else "",
    )


def timeout(limit_s: int) -> ReconstructionError:
    return ReconstructionError(
        "timeout",
        f"The job exceeded its runtime limit of {limit_s // 60} minutes and was stopped.",
        "Tip: use a shorter video or fewer photos.",
    )


def object_failed(detail: str) -> ReconstructionError:
    return ReconstructionError(
        "object_failed",
        "Generating the object failed on the GPU. Please try again, or reword the prompt.",
        f"Technical detail: {detail[-300:]}" if detail else "",
    )


def model_access(repo_id: str, detail: str = "") -> ReconstructionError:
    """A gated model could not be downloaded with the configured HF token."""
    return ReconstructionError(
        "model_access",
        f"The model {repo_id} could not be downloaded.",
        f"Accept its terms on huggingface.co/{repo_id} with the account of the HF token "
        "in the Modal secret, then try again.",
        detail[-DETAIL_CHARS:],
    )
