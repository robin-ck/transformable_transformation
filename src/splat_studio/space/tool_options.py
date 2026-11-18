"""Which Scene tab tools are shown (add-tool-visibility-options).

One global choice for the whole Space, stored in the dataset; the page's Settings dialog
changes it and a page-load handler sends it to the browser.
"""

from __future__ import annotations

import json
from typing import Protocol

# Tool sub-tabs of the Scene tab: id (used in elem_ids "tool-<id>") and label, in tab order.
TOOLS: dict[str, str] = {
    "history": "History",
    "objects": "Objects",
    "edit": "Edit scene",
    "preview": "Preview",
    "futures": "Futures",
}
PATH = "settings/space.json"


class JsonStore(Protocol):
    def write_json(self, path: str, data: dict) -> None: ...
    def read_json(self, path: str) -> dict | None: ...


def clean(hidden: object) -> list[str]:
    """Known tool ids from `hidden`, in tab order, without duplicates; anything else is []."""
    if not isinstance(hidden, list | tuple):
        return []
    return [t for t in TOOLS if t in hidden]


def parse(text: str | None) -> list[str]:
    """Hidden tools from the page's JSON list; malformed text means none hidden."""
    try:
        return clean(json.loads(text or "[]"))
    except (TypeError, ValueError):
        return []


def load(store: JsonStore) -> list[str]:
    """The saved hidden tools; none when nothing is saved or the file is unreadable."""
    try:
        data = store.read_json(PATH)
    except Exception:  # a broken settings file must not break the Space
        return []
    return clean(data.get("hidden_tools")) if isinstance(data, dict) else []


def save(store: JsonStore, hidden: list[str]) -> list[str]:
    """Save `hidden` (cleaned), keeping other keys in the settings file; returns what was saved."""
    hidden = clean(hidden)
    try:
        data = store.read_json(PATH) or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    store.write_json(PATH, {**data, "hidden_tools": hidden})
    return hidden
