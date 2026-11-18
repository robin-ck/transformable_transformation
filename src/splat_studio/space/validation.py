"""Check an upload against the input limits before anything is sent to the GPU."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from splat_studio.reconstruction import frames
from splat_studio.reconstruction.pipeline import PHOTO_EXT, VIDEO_EXT

MAX_VIDEO_SECONDS = 300
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024  # a 5-min 4K phone clip is ~1.75-2.25 GB; 1080p is far smaller
MIN_PHOTOS = 20
MAX_PHOTOS = 300

HELP = "Upload one video (MP4/MOV, up to 5 minutes and 2 GB) or 20 to 300 photos (JPG/PNG)."


class UploadError(ValueError):
    """An upload that breaks the input rules; the message is shown to the user."""


@dataclass
class ValidUpload:
    kind: str  # "video" or "photos"
    files: list[Path]
    summary: str


def _mb(n: int) -> str:
    return f"{n / 1024 / 1024:.0f} MB"


def validate_upload(
    paths: Sequence[str | Path] | None,
    probe: Callable[[Path], tuple[float, float]] = frames.probe,
) -> ValidUpload:
    """Return the upload if it meets the limits, else raise UploadError with a clear reason."""
    files = [Path(p) for p in (paths or [])]
    if not files:
        raise UploadError(f"Nothing uploaded. {HELP}")

    bad = [p.name for p in files if p.suffix.lower() not in VIDEO_EXT | PHOTO_EXT]
    if bad:
        shown = ", ".join(bad[:3]) + (f" and {len(bad) - 3} more" if len(bad) > 3 else "")
        raise UploadError(f"Unsupported file type: {shown}. {HELP}")

    videos = [p for p in files if p.suffix.lower() in VIDEO_EXT]
    photos = [p for p in files if p.suffix.lower() in PHOTO_EXT]
    if videos and photos:
        raise UploadError(f"Upload either one video or a set of photos, not both. {HELP}")

    if videos:
        if len(videos) > 1:
            raise UploadError(f"Upload only one video at a time (you uploaded {len(videos)}).")
        video = videos[0]
        size = video.stat().st_size
        if size > MAX_VIDEO_BYTES:
            raise UploadError(
                f"The video is {_mb(size)}; the limit is {_mb(MAX_VIDEO_BYTES)}. "
                "Trim it or export it at a lower bitrate."
            )
        try:
            duration, _fps = probe(video)
        except Exception as e:
            raise UploadError(
                "The video could not be read. Try exporting it again as MP4 (H.264)."
            ) from e
        if duration > MAX_VIDEO_SECONDS:
            raise UploadError(
                f"The video is {duration:.0f} s long; the limit is {MAX_VIDEO_SECONDS} s "
                "(5 minutes). Trim it and upload again."
            )
        return ValidUpload("video", [video], f"Video {video.name}: {duration:.0f} s, {_mb(size)}.")

    n = len(photos)
    if n < MIN_PHOTOS:
        raise UploadError(
            f"At least {MIN_PHOTOS} photos are needed (you uploaded {n}). "
            "Take more photos around the subject with plenty of overlap."
        )
    if n > MAX_PHOTOS:
        raise UploadError(f"At most {MAX_PHOTOS} photos are allowed (you uploaded {n}).")
    return ValidUpload("photos", sorted(photos), f"{n} photos.")


OBJECT_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
MAX_OBJECT_IMAGE_BYTES = 20 * 1024 * 1024
OBJECT_IMAGE_HELP = "Use one PNG, JPG or WebP photo of the object, up to 20 MB."


def _validate_image(path: str | Path | None, help_text: str) -> Path:
    if not path:
        raise UploadError(f"No image uploaded. {help_text}")
    image = Path(path)
    if image.suffix.lower() not in OBJECT_IMAGE_EXT:
        raise UploadError(f"Unsupported image type: {image.name}. {help_text}")
    if image.stat().st_size > MAX_OBJECT_IMAGE_BYTES:
        raise UploadError(
            f"The image is {_mb(image.stat().st_size)}; the limit is "
            f"{_mb(MAX_OBJECT_IMAGE_BYTES)}. {help_text}"
        )
    return image


def validate_object_image(path: str | Path | None) -> Path:
    """Return the reference image for Add object if it is usable, else raise UploadError."""
    return _validate_image(path, OBJECT_IMAGE_HELP)


SOURCE_IMAGE_HELP = "Use one PNG, JPG or WebP image, up to 20 MB."


def validate_source_image(path: str | Path | None) -> Path:
    """Return the input image for the Image editing tab if usable, else raise UploadError."""
    return _validate_image(path, SOURCE_IMAGE_HELP)
