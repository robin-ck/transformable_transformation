"""Runtime and cost log for GPU jobs."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

# Approximate Modal GPU prices in USD per second. Check modal.com/pricing and update.
GPU_USD_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10G": 0.000306,
    "L40S": 0.000542,
    "A100": 0.000583,
    "H100": 0.001097,
}
CPU_CORE_USD_PER_SECOND = 0.0000131


def estimate_cost(gpu: str, seconds: float, cpu_cores: float = 0.0) -> float:
    if gpu not in GPU_USD_PER_SECOND:
        raise KeyError(f"unknown GPU type: {gpu}")
    return round(seconds * (GPU_USD_PER_SECOND[gpu] + cpu_cores * CPU_CORE_USD_PER_SECOND), 4)


def gpu_from_device_name(name: str, default: str) -> str:
    """Price key for a CUDA device name such as "NVIDIA L40S"; `default` if unknown.

    For functions with a list of GPU types, where the container may get any of them.
    """
    upper = name.upper()
    for key in sorted(GPU_USD_PER_SECOND, key=len, reverse=True):  # L40S before L4
        if key in upper:
            return key
    return default


@dataclass
class RunLog:
    """Per-job record: stage timings, total runtime, estimated cost and outcome."""

    job_id: str
    gpu: str
    cpu_cores: float = 0.0
    started_at: float = field(default_factory=time.time)
    stages: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    est_cost_usd: float = 0.0
    status: str = "running"
    error: dict[str, str] | None = None
    warm: bool | None = None  # None: a plain function; else whether the models were loaded

    def stage(self, name: str, seconds: float) -> None:
        self.stages[name] = round(seconds, 1)

    def finish(self, status: str, error: dict[str, str] | None = None) -> RunLog:
        self.seconds = round(time.time() - self.started_at, 1)
        self.est_cost_usd = estimate_cost(self.gpu, self.seconds, self.cpu_cores)
        self.status, self.error = status, error
        return self

    def to_dict(self) -> dict:
        return asdict(self)


class LoadTimes:
    """Model load stages of a warm container (`@modal.enter`), reported once.

    The first job after a load gets the load stages and `warm=False`; later jobs in the same
    container get no load stages and `warm=True`.
    """

    def __init__(self) -> None:
        self.pending: dict[str, float] = {}

    def record(self, name: str, seconds: float) -> None:
        self.pending[name] = round(seconds, 1)

    def apply(self, log: RunLog) -> None:
        log.warm = not self.pending
        log.stages = {**self.pending, **log.stages}
        self.pending = {}
