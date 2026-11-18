"""Saved viewpoints and starter prompts of a scene (add-viewpoint-futures).

Stored in the dataset, one file per scene plus one folder per viewpoint:
  viewpoints/<scene_id>/setup.json            viewpoints, starters, editor and the
                                              changed editor instructions (this module)
  viewpoints/<scene_id>/<vp_id>/render.png    plain colour render (written by the worker)
  viewpoints/<scene_id>/<vp_id>/depth.png     depth control image
  viewpoints/<scene_id>/<vp_id>/view.json     camera, size, source version, render time

A viewpoint is pinned to the scene version shown when it was saved. Its id is short and
random, so renaming keeps the folder. Removing a viewpoint only drops it from
`setup.json`; its folder and futures stay. Pure logic on a `jobs.Store`, no GPU.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np

from splat_studio import models

from .jobs import Store

MAX_NAME = 40
MAX_LABEL = 30
MAX_STARTER = 200
# Starters a scene begins with; the user can change or remove them.
DEFAULT_STARTERS = (
    ("nature first", "a natural river landscape with wide reed beds, willows and wildflower "
     "meadows on the banks"),
    ("flood safe", "a river landscape designed for floods, with a wide floodplain, low "
     "green dikes and room for the water"),
    ("recreation", "a river park with footpaths, a wooden footbridge, benches and people "
     "walking along the water"),
    ("housing", "new low-rise housing along the river with green roofs and a public "
     "riverside promenade"),
)  # fmt: skip


class ViewpointError(ValueError):
    """A viewpoint or starter change cannot be made; the message says why."""


@dataclass
class Viewpoint:
    id: str
    name: str
    version: int  # scene version it shows
    camera: dict  # editing.preview.ViewCamera.to_dict()
    order: int
    created_at: float
    target: list[float] | None = None  # orbit centre in the viewer, to move back there


@dataclass
class Starter:
    id: str
    label: str
    text: str


@dataclass
class Setup:
    viewpoints: list[Viewpoint] = field(default_factory=list)
    starters: list[Starter] = field(default_factory=list)
    editor: str = models.DEFAULT_EDITOR
    # Editor instruction texts changed from `editing.instructions.DEFAULTS` (key -> text).
    instructions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "viewpoints": [asdict(v) for v in self.ordered()],
            "starters": [asdict(s) for s in self.starters],
            "editor": self.editor,
            "instructions": dict(self.instructions),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Setup:
        vps = [Viewpoint(**{k: v[k] for k in v if k in Viewpoint.__dataclass_fields__})
               for v in d.get("viewpoints", [])]  # fmt: skip
        starters = [Starter(s["id"], s["label"], s["text"]) for s in d.get("starters", [])]
        return cls(vps, starters, d.get("editor") or models.DEFAULT_EDITOR,
                   dict(d.get("instructions") or {}))  # fmt: skip

    def ordered(self) -> list[Viewpoint]:
        return sorted(self.viewpoints, key=lambda v: (v.order, v.created_at))

    def get(self, vp_id: str | None) -> Viewpoint | None:
        return next((v for v in self.viewpoints if v.id == vp_id), None)

    def starter(self, starter_id: str | None) -> Starter | None:
        return next((s for s in self.starters if s.id == starter_id), None)


def setup_path(scene_id: str) -> str:
    return f"viewpoints/{scene_id}/setup.json"


def view_prefix(scene_id: str, vp_id: str) -> str:
    """Dataset folder of a viewpoint's render, depth and view.json."""
    return f"viewpoints/{scene_id}/{vp_id}"


def default_setup() -> Setup:
    return Setup(starters=[Starter(_slug(label), label, text) for label, text in DEFAULT_STARTERS])


def load_setup(store: Store, scene_id: str) -> Setup:
    """The scene's setup; a scene without one gets no viewpoints and the default starters."""
    data = store.read_json(setup_path(scene_id))
    return Setup.from_dict(data) if data else default_setup()


def save_setup(store: Store, scene_id: str, setup: Setup) -> None:
    store.write_json(setup_path(scene_id), setup.to_dict())


def _clean(text: str | None, limit: int, what: str) -> str:
    value = " ".join((text or "").split())
    if len(value) > limit:
        raise ViewpointError(f"The {what} is {len(value)} characters long; the limit is {limit}.")
    return value


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "starter"


# --- viewpoints ------------------------------------------------------------------------


def add_viewpoint(
    setup: Setup,
    name: str | None,
    version: int,
    camera: dict,
    target: list[float] | None = None,
    now: float | None = None,
) -> Viewpoint:
    """Add a viewpoint at the end of the list; an empty name becomes "view <n>"."""
    name = _clean(name, MAX_NAME, "name") or f"view {len(setup.viewpoints) + 1}"
    order = max((v.order for v in setup.viewpoints), default=-1) + 1
    created = time.time() if now is None else now
    target = list(target) if target else None
    vp = Viewpoint(uuid.uuid4().hex[:8], name, int(version), dict(camera), order, created, target)
    setup.viewpoints.append(vp)
    return vp


def _need(setup: Setup, vp_id: str | None) -> Viewpoint:
    vp = setup.get(vp_id)
    if vp is None:
        raise ViewpointError("Pick a saved viewpoint first.")
    return vp


def rename_viewpoint(setup: Setup, vp_id: str | None, name: str | None) -> Viewpoint:
    vp = _need(setup, vp_id)
    new = _clean(name, MAX_NAME, "name")
    if not new:
        raise ViewpointError("Type the new name first.")
    vp.name = new
    return vp


def move_viewpoint(setup: Setup, vp_id: str | None, step: int) -> None:
    """Move a viewpoint `step` places up (negative) or down in the list."""
    vp = _need(setup, vp_id)
    ordered = setup.ordered()
    i = ordered.index(vp)
    j = min(max(i + step, 0), len(ordered) - 1)
    ordered.insert(j, ordered.pop(i))
    for n, v in enumerate(ordered):
        v.order = n


def remove_viewpoint(setup: Setup, vp_id: str | None) -> Viewpoint:
    """Drop a viewpoint from the list; its stored images and futures are kept."""
    vp = _need(setup, vp_id)
    setup.viewpoints.remove(vp)
    return vp


def same_view(a: dict | None, b: dict | None, tol: float = 1e-3) -> bool:
    """Whether two `ViewCamera` dicts look from the same place in the same direction.

    Field of view and aspect are ignored: they follow the browser window, and a future
    starts from the viewpoint's stored render anyway.
    """
    if not a or not b:
        return False
    ma, mb = np.asarray(a["c2w"], dtype=float), np.asarray(b["c2w"], dtype=float)
    scale = 1.0 + max(np.linalg.norm(ma[:3, 3]), np.linalg.norm(mb[:3, 3]))
    return bool(
        np.abs(ma[:3, :3] - mb[:3, :3]).max() < tol
        and np.linalg.norm(ma[:3, 3] - mb[:3, 3]) < tol * scale
    )


def viewer_target(vp: Viewpoint) -> list[float]:
    """Orbit centre to put the viewer back at the viewpoint: the stored one, else a point
    straight ahead of the camera."""
    if vp.target:
        return list(vp.target)
    m = np.asarray(vp.camera["c2w"], dtype=float)
    ahead = m[:3, 3] - m[:3, 2]  # Three.js cameras look down -z
    return [round(float(x), 6) for x in ahead]


# --- starters ----------------------------------------------------------------------------


def parse_starters(text: str | None, old: list[Starter] = ()) -> list[Starter]:
    """Starters from one "label: prompt text" per line; empty lines are skipped.

    A label that existed before keeps its id, so a changed text stays the same starter.
    """
    known = {s.label.lower(): s.id for s in old}
    out: list[Starter] = []
    for n, line in enumerate((text or "").splitlines(), 1):
        if not line.strip():
            continue
        if ":" not in line:
            raise ViewpointError(f'Line {n} needs "label: prompt text".')
        label, body = line.split(":", 1)
        label = _clean(label, MAX_LABEL, f"label on line {n}")
        body = _clean(body, MAX_STARTER, f"text on line {n}")
        if not label or not body:
            raise ViewpointError(f"Line {n} needs both a label and a prompt text.")
        sid = known.get(label.lower()) or _slug(label)
        if any(s.id == sid or s.label.lower() == label.lower() for s in out):
            raise ViewpointError(f'The label "{label}" is used twice.')
        out.append(Starter(sid, label, body))
    return out


def starters_text(starters: list[Starter]) -> str:
    return "\n".join(f"{s.label}: {s.text}" for s in starters)


def set_editor(setup: Setup, key: str) -> Setup:
    """Choose the scene's editor among the registered ones; raises ViewpointError."""
    try:
        spec = models.editor_spec(key)
    except KeyError as e:
        raise ViewpointError(str(e)) from e
    setup.editor = spec.key
    return setup
