"""Print timings of recent object and preview jobs (warm-gpu-workers, task 0.1 and 3.1).

Run from the repo root with HF_TOKEN and SPLAT_STUDIO_DATASET set or in `.env`:
    python scripts/job_timings.py [N]
Reads `objects/<id>/` and `previews/<id>/` (the last N of each).
start_wait_s: click to function start (queue + cold start); only for jobs sent from the Space,
which write job.json. total_s: click to finish. run_s and stages: inside the function.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splat_studio.space.jobs import HubStore  # noqa: E402

KINDS = {"objects": "object", "previews": "preview"}


def load_env(path: Path) -> None:
    for line in path.read_text().splitlines() if path.exists() else []:
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def main(n: int = 15) -> None:
    load_env(Path(".env"))
    store = HubStore(os.environ["SPLAT_STUDIO_DATASET"], os.environ.get("HF_TOKEN"))
    for prefix, kind in KINDS.items():
        for jid in sorted(store.list_dirs(prefix))[-n:]:
            job = store.read_json(f"{prefix}/{jid}/job.json") or {}
            run = job.get("run") or store.read_json(f"{prefix}/{jid}/run.json") or {}
            wait = total = None
            if job.get("created_at") and run.get("started_at"):
                wait = round(run["started_at"] - job["created_at"], 1)
            if job.get("created_at") and job.get("finished_at"):
                total = round(job["finished_at"] - job["created_at"], 1)
            source = "space" if job else "direct"
            print(jid, kind, source, run.get("status"), "start_wait_s=", wait,
                  "total_s=", total, "run_s=", run.get("seconds"), run.get("stages"))  # fmt: skip


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
