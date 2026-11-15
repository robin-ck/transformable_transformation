"""Print timings of recent object and preview jobs (warm-gpu-workers, task 0.1 and 3.1).

Run from the repo root with HF_TOKEN and SPLAT_STUDIO_DATASET set or in `.env`:
    python scripts/job_timings.py [N]
start_wait_s: click to function start (queue + cold start); run_s and stages: inside the function.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splat_studio.space.jobs import HubStore  # noqa: E402


def load_env(path: Path) -> None:
    for line in path.read_text().splitlines() if path.exists() else []:
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def main(n: int = 15) -> None:
    load_env(Path(".env"))
    store = HubStore(os.environ["SPLAT_STUDIO_DATASET"], os.environ.get("HF_TOKEN"))
    for jid in sorted(store.list_dirs("jobs"))[-n:]:
        j = store.read_json(f"jobs/{jid}/job.json") or {}
        if j.get("kind") not in ("object", "preview"):
            continue
        run = j.get("run") or {}
        wait = round(run["started_at"] - j["created_at"], 1) if run.get("started_at") else None
        print(jid, j.get("kind"), j.get("status"), "start_wait_s=", wait,
              "run_s=", run.get("seconds"), run.get("stages"))  # fmt: skip


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
