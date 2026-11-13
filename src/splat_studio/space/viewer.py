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
`{type: "camera", id, matrix, fov, aspect}` (Three.js camera-to-world, column-major),
so the Modal job can render exactly the view on screen.
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


def viewer_document(scene_url: str, camera: Camera, obj: dict | None = None) -> str:
    """Self-contained HTML page that shows one `.splat` scene.

    `obj` adds a movable object: {"url": ..., "position": [...], "quaternion": [x, y, z, w],
    "scale": [...]} (see `editing.objects.Placement.to_viewer`).
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
  #msg { position: absolute; inset: 0; display: flex; align-items: center;
    justify-content: center; text-align: center; padding: 16px; z-index: 5; }
</style>
<script type="importmap">{"imports": {"three": "__THREE__"}}</script>
</head><body>
<div id="root"></div>
<div id="msg">Loading scene…</div>
<div id="bar"><button id="reset" title="Reset view (R)">Reset view</button></div>
<div id="help">Drag: orbit · Right-drag / two fingers: pan · Wheel / pinch: zoom · R: reset</div>
<script type="module">
const cfg = __CONFIG__;
const msg = document.getElementById("msg");
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
    e.source.postMessage({ type: "camera", id: d.id, matrix: Array.from(c.matrixWorld.elements),
      fov: c.fov, aspect: c.aspect }, "*");
  });
  if (cfg.object) {
    const o = cfg.object;
    await viewer.addSplatScenes([
      { path: cfg.url, format: GS.SceneFormat.Splat },
      { path: o.url, format: GS.SceneFormat.Splat,
        position: o.position, rotation: o.quaternion, scale: o.scale },
    ], true);
    window.addEventListener("message", (e) => {
      const d = e.data;
      if (!d || d.type !== "placement") return;
      const obj = viewer.getSplatScene(1);
      const half = (Number(d.turn) || 0) * Math.PI / 360;
      obj.position.set(Number(d.x), Number(d.y), Number(d.z));
      obj.quaternion.set(0, 0, Math.sin(half), Math.cos(half));  // turn about z (up)
      const size = Math.max(Number(d.size) || 0, 1e-4);
      obj.scale.set(size, size, size);
    });
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
) -> str:
    """Viewer (or a message) for the scene version stored under `scene_path`.

    `store` is a `jobs.Store`; the `.splat` file is fetched to local disk first.
    With `object_path` (an object job folder) and `placement` (`Placement.to_viewer()`),
    the generated object is shown too and can be moved live.
    """
    if not scene_path:
        return placeholder("Pick a finished scene to view it here.", height)
    try:
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
    return viewer_iframe(file_url(splat), camera_for(splat_positions(splat)), height, obj)
