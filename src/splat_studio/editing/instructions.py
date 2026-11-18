"""Editor instructions (edit-editor-instructions): the text a future adds to the user's prompt.

Before the editor model runs, a future's prompt can get fixed text added to it:
  negative        SDXL only: what the image should not look like (its negative prompt)
  object_change   Kontext with "Object to change": added after the prompt
  object_removal  the same, when the prompt removes something ("remove the chairs")
  area_change     Kontext with a drawn "Area to change": added after the prompt
  area_removal    the same, when the prompt removes something
  removal_words   words that make a prompt a removal (comma separated, whole words)
  pronouns        words replaced by the named object ("make it green" -> "make the
                  chair green"; comma separated, whole words)
  preview_negative  the Preview tool's negative prompt (SDXL)
In the object texts `{target}` stands for the named object ("the chairs"). Emptying a
word list turns the rule off.

Each scene keeps its own texts in `viewpoints/<scene_id>/setup.json` (`instructions`,
only the ones changed from `DEFAULTS`); the Space shows and edits them, and every future
records the full set it was made with, so the worker uses exactly what the user saw.
Pure logic, no GPU.
"""

from __future__ import annotations

import re

MAX_TEXT = 400  # characters per instruction

DEFAULTS: dict[str, str] = {
    "negative": "blurry, low quality, distorted, deformed, oversaturated, text, watermark",
    "object_change": "Keep the shape, material, texture, shading and lighting of {target}. "
    "Keep everything else unchanged.",
    "object_removal": "Fill the space where {target} stood with the floor, wall and objects "
    "that are behind them, matching the lighting. Keep everything else unchanged.",
    "area_change": "Make the change in the middle of the image, matching the perspective, "
    "scale, lighting and style of the scene. Keep everything else unchanged.",
    "area_removal": "Fill the space in the middle of the image with what is behind it, "
    "matching the surroundings and the lighting. Keep everything else unchanged.",
    "removal_words": "remove, delete, erase, take away, get rid of, without",
    "pronouns": "it, them",
    "preview_negative": "blurry, low quality, distorted, deformed, oversaturated, text, watermark",
}
WORD_LISTS = ("removal_words", "pronouns")
FUTURE_KEYS = tuple(k for k in DEFAULTS if k != "preview_negative")  # Futures settings

# Shown next to each text in the Space: (label, when it is used).
LABELS: dict[str, tuple[str, str]] = {
    "negative": ("Negative prompt (SDXL)", "What the image should not look like."),
    "object_change": (
        "Object to change: added after your prompt",
        "Kontext, when an object is named. {target} is the named object.",
    ),
    "object_removal": (
        "Object to remove: added after your prompt",
        'Kontext, when an object is named and the prompt removes it ("remove", "delete"…). '
        "{target} is the named object.",
    ),
    "area_change": (
        "Drawn area: added after your prompt",
        "Kontext, when an area is painted.",
    ),
    "area_removal": (
        "Drawn area, removal: added after your prompt",
        'Kontext, when an area is painted and the prompt removes something ("remove"…).',
    ),
    "removal_words": (
        "Removal words (comma separated)",
        "A prompt with one of these whole words is a removal: it gets the removal text "
        "above and a wider mask margin. Empty: no prompt is a removal.",
    ),
    "pronouns": (
        "Words replaced by the named object (comma separated)",
        'With an object named, "make it green" becomes "make the chair green". '
        "Empty: nothing is replaced.",
    ),
    "preview_negative": (
        "Negative prompt",
        "What the image should not look like (SDXL). Saved for this scene when you click "
        "Preview this view.",
    ),
}


class InstructionError(ValueError):
    """An instruction text cannot be used; the message says why."""


def resolve(custom: dict | None = None) -> dict[str, str]:
    """The full set: `DEFAULTS` with the texts in `custom` in their place."""
    out = dict(DEFAULTS)
    for key, text in (custom or {}).items():
        if key in DEFAULTS and text is not None:
            out[key] = " ".join(str(text).split())
    return out


def clean(texts: dict) -> dict[str, str]:
    """The texts that differ from `DEFAULTS` (what setup.json stores), checked.

    An empty text means "add nothing". Raises InstructionError for an unknown key, a text
    that is too long, or a `{...}` placeholder other than `{target}`."""
    out = {}
    for key, text in texts.items():
        if key not in DEFAULTS:
            raise InstructionError(f"Unknown instruction {key!r}.")
        text = " ".join((text or "").split())
        label = LABELS[key][0]
        if len(text) > MAX_TEXT:
            raise InstructionError(
                f"'{label}' is {len(text)} characters long; the limit is {MAX_TEXT}."
            )
        if key in WORD_LISTS:
            text = ", ".join(words({key: text}, key))
        try:
            text.format(target="x")
        except (KeyError, IndexError, ValueError) as e:
            raise InstructionError(
                f"'{label}' may only use {{target}} in braces ({e!s} is not known)."
            ) from e
        if text != DEFAULTS[key]:
            out[key] = text
    return out


def words(texts: dict | None, key: str) -> list[str]:
    """A word list ("removal_words", "pronouns"), lowercase, empty items dropped."""
    items = (" ".join(w.split()).lower() for w in resolve(texts)[key].split(","))
    return [w for w in items if w]


def word_pattern(texts: dict | None, key: str) -> re.Pattern | None:
    """A case-insensitive whole-word pattern for a word list; None when it is empty."""
    items = words(texts, key)
    if not items:
        return None
    alts = "|".join(r"\s+".join(map(re.escape, w.split())) for w in items)
    return re.compile(rf"\b({alts})\b", re.IGNORECASE)


def tail(texts: dict | None, key: str, target: str = "") -> str:
    """One instruction, with `{target}` filled in ("" when the user emptied it)."""
    text = resolve(texts)[key]
    return text.replace("{target}", target) if text else ""


def editor_prompt(
    prompt: str, object_query: str | None = None, has_area: bool = False,
    texts: dict | None = None,
) -> str:  # fmt: skip
    """What the editor gets for a future: the prompt as is, or with the object or area
    instruction added (the same rule the worker applies)."""
    from . import area, masking

    if has_area:
        return area.area_prompt(prompt, texts)
    if masking.object_phrases(object_query):
        return masking.object_prompt(prompt, object_query, texts)
    return prompt
