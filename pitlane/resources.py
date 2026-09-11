"""Host and GPU sampling for the duration of a cell, and the abort that needs it.

`preflight` checks RAM once, before anything starts. That says the machine
*could* host the run; it says nothing about the run itself, and the things that
grow during one are exactly the things preflight cannot see -- LMCache's L1 pool
climbing toward `--l1-size-gb`, a workflow holding every response in memory, a
second job someone started on the same box an hour in.

Without a ceiling the ending is the kernel's OOM killer, which picks its own
victim: as likely vLLM or pitlane as the process that grew. Aborting first means
the run ends where it can be read -- a row in `resources.csv` with the sample
that crossed, and a cell marked aborted -- rather than as a process that
vanished.

Sampling is cheap and the interval is long, so the monitor's own cost does not
show up in what it is measuring.
"""

from __future__ import annotations

import csv
import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("pitlane.resources")

_FIELDS = [
    "ts", "arm", "question_id", "rep",
    "ram_free_gb", "ram_total_gb",
    "gpu_util_pct", "vram_used_mib", "vram_total_mib",
    "disk_free_gb",
]


@dataclass
class Sample:
    ts: float
    ram_free_gb: float | None = None
    ram_total_gb: float | None = None
    gpu_util_pct: float | None = None
    vram_used_mib: float | None = None
    vram_total_mib: float | None = None
    disk_free_gb: float | None = None


def _meminfo() -> tuple[float | None, float | None]:
    """`(available, total)` in GB, from /proc/meminfo.

    `MemAvailable`, not `MemFree`: page cache is reclaimable, so `MemFree`
    reports a machine as nearly out of memory while it is nothing of the kind,
    and an abort on that would fire on every healthy run.
    """
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in ("MemAvailable:", "MemTotal:"):
                fields[parts[0]] = int(parts[1]) / 1024 / 1024
        return fields.get("MemAvailable:"), fields.get("MemTotal:")
    except (OSError, ValueError):
        return None, None


def _gpu(index: int) -> tuple[float | None, float | None, float | None]:
    """`(util %, used MiB, total MiB)` for one GPU, or Nones without nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None, None, None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        util, used, total = (float(x) for x in out.split(",")[:3])
        return util, used, total
    except (ValueError, subprocess.SubprocessError):
        return None, None, None


class Monitor:
    """Samples into `path` on a background thread until stopped.

    `aborted` is set when a sample crosses a threshold. It is an `Event` rather
    than an exception because the thing that has to react is a subprocess in
    another module: `workflow.run` waits on it alongside the child, and the
    runner reads `reason` afterwards to explain the cell.
    """

    def __init__(
        self,
        path: Path,
        *,
        scope: dict[str, object],
        gpu: int = 0,
        disk_path: Path | None = None,
        interval_s: float = 5.0,
        ram_floor_gb: float = 0.0,
        vram_ceiling_frac: float = 0.0,
    ) -> None:
        self.path = path
        self.scope = scope
        self.gpu = gpu
        self.disk_path = disk_path
        self.interval_s = interval_s
        self.ram_floor_gb = ram_floor_gb
        self.vram_ceiling_frac = vram_ceiling_frac

        self.aborted = threading.Event()
        self.reason: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned_blind = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "Monitor":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._loop, name="pitlane-resources", daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Joined, not abandoned: the last row has to be on disk before the
            # runner reads the file back.
            self._thread.join(timeout=self.interval_s + 5)

    def __enter__(self) -> "Monitor":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    # -- sampling ----------------------------------------------------------
    def sample(self) -> Sample:
        ram_free, ram_total = _meminfo()
        util, used, total = _gpu(self.gpu)
        disk_free = None
        if self.disk_path is not None:
            try:
                disk_free = shutil.disk_usage(self.disk_path).free / 1024**3
            except OSError:
                disk_free = None
        return Sample(time.time(), ram_free, ram_total, util, used, total, disk_free)

    def _check(self, sample: Sample) -> str | None:
        if self.ram_floor_gb > 0 and sample.ram_free_gb is None:
            # A floor that cannot be read is not a floor. Said once, loudly:
            # silently never firing is the failure mode that looks like safety.
            if not self._warned_blind:
                self._warned_blind = True
                logger.warning(
                    "RAM abort floor of %.0f GB is set but memory is "
                    "unreadable (/proc/meminfo missing); the run has no RAM "
                    "ceiling", self.ram_floor_gb,
                )
        if (
            self.ram_floor_gb > 0
            and sample.ram_free_gb is not None
            and sample.ram_free_gb < self.ram_floor_gb
        ):
            return (
                f"free RAM {sample.ram_free_gb:.0f} GB below the abort floor of "
                f"{self.ram_floor_gb:.0f} GB"
            )
        # Off by default, and deliberately. vLLM claims its share of VRAM up
        # front -- `--gpu-memory-utilization` is typically 0.9 -- so a card that
        # looks nearly full is a card that is working correctly, and a ceiling
        # on it would fire on every healthy run. Only useful on a shared card,
        # where the question is whether someone else is growing into yours.
        if (
            self.vram_ceiling_frac > 0
            and sample.vram_used_mib is not None
            and sample.vram_total_mib
        ):
            frac = sample.vram_used_mib / sample.vram_total_mib
            if frac > self.vram_ceiling_frac:
                return (
                    f"VRAM {frac:.0%} above the abort ceiling of "
                    f"{self.vram_ceiling_frac:.0%}"
                )
        return None

    def _loop(self) -> None:
        new = not self.path.exists()
        try:
            with self.path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=_FIELDS)
                if new:
                    writer.writeheader()
                while True:
                    sample = self.sample()
                    row = {**self.scope, "ts": sample.ts}
                    row.update({
                        "ram_free_gb": _round(sample.ram_free_gb),
                        "ram_total_gb": _round(sample.ram_total_gb),
                        "gpu_util_pct": _round(sample.gpu_util_pct),
                        "vram_used_mib": _round(sample.vram_used_mib),
                        "vram_total_mib": _round(sample.vram_total_mib),
                        "disk_free_gb": _round(sample.disk_free_gb),
                    })
                    writer.writerow({k: row.get(k) for k in _FIELDS})
                    # Flushed every sample: the row that matters most is the one
                    # written just before the run was torn down.
                    handle.flush()

                    reason = self._check(sample)
                    if reason and not self.aborted.is_set():
                        self.reason = reason
                        logger.error("aborting: %s", reason)
                        self.aborted.set()

                    if self._stop.wait(self.interval_s):
                        return
        except Exception:  # noqa: BLE001 - monitoring must not end the run
            logger.exception("resource monitor stopped")


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)
