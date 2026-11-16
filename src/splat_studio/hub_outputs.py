"""Write a worker job's outputs to the HF Hub dataset in one commit.

Each Hub commit takes seconds (20 to 30 s when the Hub is slow), whatever its size, so a
job commits its output files together with its final `run.json` instead of one commit
per file. `run.json` then turns "done" in the same commit that adds the files, so a
reader who sees it done also finds the files.
"""

from __future__ import annotations

import json
import time
from typing import Any

from splat_studio.reconstruction.cost import RunLog


def commit_files(api: Any, repo: str, files: dict[str, Any], message: str) -> None:
    """One commit to dataset `repo`. `files` maps a repo path to a local file (str or
    Path), bytes, or a dict that is written as JSON."""
    from huggingface_hub import CommitOperationAdd

    def content(value: Any) -> str | bytes:
        if isinstance(value, dict):
            return json.dumps(value, indent=2).encode()
        return value if isinstance(value, bytes) else str(value)

    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=content(v)) for p, v in files.items()]
    api.create_commit(repo_id=repo, operations=ops, commit_message=message, repo_type="dataset")


def finish_with_outputs(
    api: Any,
    repo: str,
    prefix: str,
    log: RunLog,
    files: dict[str, Any],
    info: dict,
    message: str,
) -> dict:
    """Finish `log` as done and commit `files` plus `<prefix>/run.json` as one commit.

    Returns the job result: the run log, then `info`, with the commit time as the
    `upload` stage. The committed `run.json` cannot contain its own commit time, so it
    has no `upload` stage and its `seconds` stop before the commit; the returned result
    (what the Space stores in `job.json`) has both.
    """
    t = time.time()
    log.finish("done")
    commit_files(api, repo, {**files, f"{prefix}/run.json": {**log.to_dict(), **info}}, message)
    log.stage("upload", time.time() - t)
    log.finish("done")
    return {**log.to_dict(), **info}
