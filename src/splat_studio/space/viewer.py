"""Browser splat viewer for the Space (task 3.3).

The viewer is a small HTML page placed in an `<iframe srcdoc>` inside a `gr.HTML`
block (Gradio does not run scripts in `gr.HTML` itself). It loads
`@mkkellogg/gaussian-splats-3d` (MIT, Three.js-based) from a CDN, and fetches the
scene's `.splat` file through Gradio's file route. The srcdoc document shares the
Space's origin, so the relative `/gradio_api/file=` URL works and no CORS setup is needed.

Navigation comes from Three.js OrbitControls: left drag orbits, right drag (or
two-finger drag) pans, wheel (or pinch) zooms. "Reset view" (or the R key) returns
to the start camera.

Placement mode (task 4.2): a second `.splat` (a generated object) is loaded next to
the scene with its own position, turn and size. The page listens for
`{type: "placement", x, y, z, turn, size}` messages from the Space and moves the
object live, so dragging a slider does not reload the scene.

Preview (task 4.3): on `{type: "get_camera", id}` the page replies to the sender with
`{type: "camera", id, matrix, fov, aspect, target}` (Three.js camera-to-world,
column-major; `target` is the orbit centre), so the Modal job can render exactly the
view on screen, and a reloaded viewer can start where the user was looking.

Click and drag (redesign-space-gui): during placement a click on the scene puts the
object's base on the clicked splat (the library's raycast), or on the floor plane
(`floor`, see `add_object.floor_height`) if no splat is hit. Dragging the object moves
it in the horizontal plane at its current height; Shift-drag turns it. Orbiting is
paused while dragging. After a click or drag the page posts
`{type: "placement_changed", x, y, z, turn, size}` to the Space, answers
`{type: "get_placement", id}` with `{type: "placement", id, ...}`, and shows or hides
the object on `{type: "object_visible", visible}`.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Pinned CDN builds; keep three within the viewer's peer range (>=0.160).
VIEWER_JS = "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.js"
THREE_JS = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js"

SPLAT_RECORD_BYTES = 32  # see reconstruction/export.py
# gsplat's simple_trainer normalises the world so that +z points up (it averages the
# training cameras' up vectors), so scenes from our pipeline are z-up.
DEFAULT_UP = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Camera:
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up: tuple[float, float, float] = DEFAULT_UP


def splat_positions(path: Path, max_points: int = 200_000) -> np.ndarray:
    """Gaussian centres from a `.splat` file, evenly subsampled to at most `max_points`."""
    n = path.stat().st_size // SPLAT_RECORD_BYTES
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    data = np.memmap(path, dtype=np.float32, mode="r", shape=(n, SPLAT_RECORD_BYTES // 4))
    step = max(1, n // max_points)
    return np.array(data[::step, :3])


def camera_for(positions: np.ndarray, up: tuple[float, float, float] = DEFAULT_UP) -> Camera:
    """A start camera that looks at the scene's centre from outside, slightly above.

    Uses the median and a percentile radius so stray floaters far from the scene
    do not push the camera away.
    """
    finite = positions[np.isfinite(positions).all(axis=1)] if len(positions) else positions
    if len(finite) == 0:
        return Camera((0.0, -3.0, 1.5), (0.0, 0.0, 0.0), up)
    centre = np.median(finite, axis=0)
    radius = float(np.percentile(np.linalg.norm(finite - centre, axis=1), 80))
    radius = max(radius, 1e-3)
    up_v = np.asarray(up, dtype=np.float64)
    up_v /= np.linalg.norm(up_v)
    # Any horizontal direction will do; pick the axis least aligned with up.
    side = np.eye(3)[np.argmin(np.abs(up_v))]
    back = side - up_v * (side @ up_v)
    back /= np.linalg.norm(back)
    position = centre - 1.8 * radius * back + 0.8 * radius * up_v
    return Camera(_vec(position), _vec(centre), _vec(up_v))


def _vec(v: np.ndarray) -> tuple[float, float, float]:
    x, y, z = (round(float(c), 4) for c in v)
    return (x, y, z)


def file_url(path: Path) -> str:
    """URL of a local file served by Gradio (the folder must be in `allowed_paths`)."""
    return "/gradio_api/file=" + Path(path).resolve().as_posix()


def camera_from_json(text: str | None, up: tuple[float, float, float] = DEFAULT_UP):
    """The viewer's camera from a `get_camera` reply (JSON), or None if unusable.

    Used to reopen the viewer at the same view after a reload (e.g. hiding an object).
    """
    try:
        d = json.loads(text or "")
        m = [float(v) for v in d["matrix"]]
        target = [float(v) for v in d["target"]]
    except (ValueError, TypeError, KeyError):
        return None
    if len(m) != 16 or len(target) != 3:
        return None
    position = np.array(m[12:15])
    if not (np.isfinite(position).all() and np.isfinite(target).all()):
        return None
    if np.linalg.norm(position - np.array(target)) < 1e-6:
        return None
    return Camera(_vec(position), _vec(np.array(target)), up)


def viewer_document(scene_url: str, camera: Camera, obj: dict | None = None) -> str:
    """Self-contained HTML page that shows one `.splat` scene.

    `obj` adds a movable object: {"url": ..., "position": [...], "quaternion": [x, y, z, w],
    "scale": [...], "turn": degrees, "floor": z, "visible": bool} (see
    `editing.objects.Placement.to_viewer`).
    """
    config = json.dumps(
        {
            "url": scene_url,
            "position": camera.position,
            "lookAt": camera.look_at,
            "up": camera.up,
            "object": obj,
        }
    ).replace("</", "<\\/")
    return (
        _PAGE.replace("__CONFIG__", config)
        .replace("__THREE__", THREE_JS)
        .replace("__VIEWER__", VIEWER_JS)
    )


def viewer_iframe(
    scene_url: str, camera: Camera, height: int = 560, obj: dict | None = None
) -> str:
    """The viewer as an iframe for a `gr.HTML` block."""
    doc = html.escape(viewer_document(scene_url, camera, obj), quote=True)
    return (
        f'<iframe class="splat-viewer" title="3D scene viewer" srcdoc="{doc}" '
        f'style="width:100%;height:{int(height)}px;border:0;border-radius:8px;" '
        'allow="fullscreen" allowfullscreen></iframe>'
    )


def placeholder(message: str, height: int = 560) -> str:
    """Grey box with a message, shown where the viewer would be."""
    return (
        f'<div style="height:{int(height)}px;display:flex;align-items:center;'
        "justify-content:center;border-radius:8px;background:#1f2430;color:#c9cfdb;"
        f'font:15px system-ui,sans-serif;text-align:center;padding:16px">'
        f"{html.escape(message)}</div>"
    )


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html, body { margin: 0; height: 100%; background: #1f2430; overflow: hidden;
    font: 14px system-ui, sans-serif; color: #e6e9ef; }
  #root { position: absolute; inset: 0; touch-action: none; }
  #bar { position: absolute; top: 10px; right: 10px; display: flex; gap: 6px; z-index: 10; }
  #bar button { background: rgba(20,24,32,.8); color: #e6e9ef; border: 1px solid #4a5263;
    border-radius: 6px; padding: 6px 10px; cursor: pointer; font: inherit; }
  #bar button:hover { background: rgba(40,46,60,.9); }
  #help { position: absolute; left: 10px; bottom: 8px; z-index: 10; opacity: .7;
    font-size: 12px; pointer-events: none; }
  #placehelp { position: absolute; left: 10px; top: 10px; z-index: 10; display: none;
    font-size: 12px; background: rgba(20,24,32,.8); padding: 4px 8px; border-radius: 6px; }
  #msg { position: absolute; inset: 0; display: flex; align-items: center;
    justify-content: center; text-align: center; padding: 16px; z-index: 5; }
</style>
<script type="importmap">{"imports": {"three": "__THREE__"}}</script>
</head><body>
<div id="root"></div>
<div id="msg">Loading scene…</div>
<div id="bar"><button id="reset" title="Reset view (R)">Reset view</button></div>
<div id="help">Drag: orbit · Right-drag / two fingers: pan · Wheel / pinch: zoom · R: reset</div>
<div id="placehelp">Click the scene: put the object there · Drag it: move · Shift-drag: turn</div>
<script type="module">
const cfg = __CONFIG__;
const msg = document.getElementById("msg");
const root = document.getElementById("root");

// Placement (task 4.2, click and drag from redesign-space-gui).
async function setupPlacement(viewer, o) {
  const THREE = await import("three");
  const obj = viewer.getSplatScene(1);
  const p = { x: o.position[0], y: o.position[1], z: o.position[2],
              turn: Number(o.turn) || 0, size: o.scale[0] };
  const floor = Number.isFinite(o.floor) ? o.floor : p.z;
  obj.visible = o.visible !== false;
  const round = (v, n = 1000) => Math.round(v * n) / n;
  const apply = () => {
    const half = p.turn * Math.PI / 360;
    obj.position.set(p.x, p.y, p.z);
    obj.quaternion.set(0, 0, Math.sin(half), Math.cos(half));  // turn about z (up)
    const size = Math.max(p.size, 1e-4);
    obj.scale.set(size, size, size);
  };
  const values = () => ({ x: round(p.x), y: round(p.y), z: round(p.z),
                          turn: Math.round(p.turn), size: round(p.size, 10000) });
  const tell = () => {
    window.parent.postMessage({ type: "placement_changed", ...values() }, "*");
  };

  window.addEventListener("message", (e) => {
    const d = e.data;
    if (!d) return;
    if (d.type === "placement") {
      p.x = Number(d.x); p.y = Number(d.y); p.z = Number(d.z);
      p.turn = Number(d.turn) || 0; p.size = Number(d.size) || p.size;
      apply();
    } else if (d.type === "get_placement" && e.source) {
      e.source.postMessage({ type: "placement", id: d.id, ...values() }, "*");
    } else if (d.type === "object_visible") {
      obj.visible = !!d.visible;
    }
  });

  // Rays from a pointer event.
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const rayAt = (e) => {
    const r = root.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, viewer.camera);
    return ray.ray;
  };
  const onPlane = (r, z) => {
    if (Math.abs(r.direction.z) < 1e-6) return null;
    const t = (z - r.origin.z) / r.direction.z;
    return t > 0 ? r.origin.clone().addScaledVector(r.direction, t) : null;
  };
  // The object is normalised: base centre at the origin, largest side 1.
  const box = new THREE.Box3(new THREE.Vector3(-0.5, -0.5, 0), new THREE.Vector3(0.5, 0.5, 1));
  const inverse = new THREE.Matrix4();
  const hitsObject = (r) => {
    if (!obj.visible) return false;
    obj.updateMatrixWorld(true);
    inverse.copy(obj.matrixWorld).invert();
    return r.clone().applyMatrix4(inverse).intersectsBox(box);
  };

  // Click: the viewer's own click handler (which would move the orbit centre) is
  // replaced, so a click puts the object's base on the clicked point instead.
  viewer.onMouseClick = (mouse) => {
    const r = rayAt(mouse);
    let point = null;
    try {
      const hits = [], dims = new THREE.Vector2(root.offsetWidth, root.offsetHeight);
      const wasVisible = obj.visible;
      obj.visible = false;  // raycast the scene, not the object itself
      viewer.raycaster.setFromCameraAndScreenPosition(
        viewer.camera, new THREE.Vector2(mouse.offsetX, mouse.offsetY), dims);
      viewer.raycaster.intersectSplatMesh(viewer.splatMesh, hits);
      obj.visible = wasVisible;
      if (hits.length && hits[0].origin && Number.isFinite(hits[0].origin.x)) {
        point = hits[0].origin.clone();
      }
    } catch (err) { console.warn("splat raycast failed, using the floor plane", err); }
    if (!point) point = onPlane(r, floor);
    if (!point) return;
    p.x = point.x; p.y = point.y; p.z = point.z;
    apply(); tell();
  };

  // Drag (capture phase, so orbiting never sees a drag that starts on the object).
  let drag = null;
  root.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    const r = rayAt(e);
    if (!hitsObject(r)) return;
    const start = onPlane(r, p.z);
    drag = { id: e.pointerId, shift: e.shiftKey, x0: e.clientX, turn0: p.turn,
             offset: start ? new THREE.Vector3(p.x - start.x, p.y - start.y, 0) : null };
    if (viewer.controls) viewer.controls.enabled = false;
    root.setPointerCapture(e.pointerId);
    root.style.cursor = "grabbing";
    e.stopPropagation(); e.preventDefault();
  }, true);
  root.addEventListener("pointermove", (e) => {
    if (!drag || e.pointerId !== drag.id) {
      if (!drag && e.buttons === 0) root.style.cursor = hitsObject(rayAt(e)) ? "grab" : "";
      return;
    }
    if (drag.shift || e.shiftKey) {
      let t = drag.turn0 + (e.clientX - drag.x0) * 0.5;
      p.turn = ((t + 180) % 360 + 360) % 360 - 180;
    } else if (drag.offset) {
      const q = onPlane(rayAt(e), p.z);
      if (q) { p.x = q.x + drag.offset.x; p.y = q.y + drag.offset.y; }
    }
    apply();
    e.stopPropagation();
  }, true);
  const end = (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    drag = null;
    if (viewer.controls) viewer.controls.enabled = true;
    root.style.cursor = "";
    e.stopPropagation();
    tell();
  };
  root.addEventListener("pointerup", end, true);
  root.addEventListener("pointercancel", end, true);
}
try {
  const GS = await import("__VIEWER__");
  const viewer = new GS.Viewer({
    rootElement: document.getElementById("root"),
    cameraUp: cfg.up,
    initialCameraPosition: cfg.position,
    initialCameraLookAt: cfg.lookAt,
    sharedMemoryForWorkers: false,
    useBuiltInControls: true,
    dynamicScene: !!cfg.object,  // the object moves after loading
  });
  window.splatViewer = viewer;  // for debugging from the browser console
  msg.style.display = "none";  // the viewer shows its own loading progress
  const reset = () => {
    const c = viewer.camera, o = viewer.controls;
    if (!c || !o) return;
    c.up.set(...cfg.up);
    c.position.set(...cfg.position);
    o.target.set(...cfg.lookAt);
    c.lookAt(...cfg.lookAt);
    o.update();
  };
  document.getElementById("reset").addEventListener("click", reset);
  window.addEventListener("keydown", (e) => { if (e.key === "r" || e.key === "R") reset(); });
  // Preview (task 4.3): the Space asks for the current camera to render the same view.
  window.addEventListener("message", (e) => {
    const d = e.data, c = viewer.camera;
    if (!d || d.type !== "get_camera" || !c || !e.source) return;
    c.updateMatrixWorld();
    const t = viewer.controls ? viewer.controls.target : null;
    e.source.postMessage({ type: "camera", id: d.id, matrix: Array.from(c.matrixWorld.elements),
      fov: c.fov, aspect: c.aspect, target: t ? [t.x, t.y, t.z] : null }, "*");
  });
  if (cfg.object) {
    const o = cfg.object;
    await viewer.addSplatScenes([
      { path: cfg.url, format: GS.SceneFormat.Splat },
      { path: o.url, format: GS.SceneFormat.Splat,
        position: o.position, rotation: o.quaternion, scale: o.scale },
    ], true);
    await setupPlacement(viewer, o);
    document.getElementById("placehelp").style.display = "block";
  } else {
    await viewer.addSplatScene(cfg.url, {
      format: GS.SceneFormat.Splat, showLoadingUI: true, progressiveLoad: true,
    });
  }
  viewer.start();
  window.splatViewerReady = true;
} catch (err) {
  console.error(err);
  msg.style.display = "flex";
  msg.textContent = "The scene could not be shown: " + (err && err.message ? err.message : err);
}
</script>
</body></html>
"""


def scene_html(
    store,
    scene_path: str | None,
    height: int = 560,
    object_path: str | None = None,
    placement: dict | None = None,
    splat: Path | None = None,
    camera: Camera | None = None,
) -> str:
    """Viewer (or a message) for the scene version stored under `scene_path`.

    `store` is a `jobs.Store`; the `.splat` file is fetched to local disk first, unless
    `splat` gives a local file to show instead (e.g. with hidden objects removed).
    With `object_path` (an object job folder) and `placement` (`Placement.to_viewer()`,
    plus optional "floor" and "visible"), the generated object is shown too and can be
    moved live. `camera` keeps the user's view across a reload.
    """
    if not scene_path:
        return placeholder("Pick a finished scene to view it here.", height)
    try:
        if splat is None:
            splat = store.download(f"{scene_path}/scene.splat")
    except Exception as e:  # network or auth trouble: say so instead of crashing the UI
        return placeholder(f"The scene file could not be fetched ({type(e).__name__}).", height)
    if splat is None:
        return placeholder(f"No scene file found in {scene_path}.", height)
    obj = None
    if object_path:
        try:
            obj_splat = store.download(f"{object_path}/object.splat")
        except Exception as e:
            return placeholder(
                f"The object file could not be fetched ({type(e).__name__}).", height
            )
        if obj_splat is None:
            return placeholder(f"No object file found in {object_path}.", height)
        obj = {"url": file_url(obj_splat), **(placement or {})}
    camera = camera or camera_for(splat_positions(splat))
    return viewer_iframe(file_url(splat), camera, height, obj)
