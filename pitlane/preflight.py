"""Preflight checks -- refuse to start a run that will hang the machine.

Every check returns a Check rather than raising, so one command reports every
problem at once instead of making the user rediscover them one restart at a
time.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pitlane.config import Config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True

    def __str__(self) -> str:
        mark = "ok  " if self.ok else ("FAIL" if self.fatal else "warn")
        return f"[{mark}] {self.name}: {self.detail}"


def _gpu_free(gpu: int) -> Check:
    if shutil.which("nvidia-smi") is None:
        return Check("gpu", False, "nvidia-smi not found", fatal=False)
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    procs = [l for l in query.stdout.splitlines() if l.strip()]
    mem = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    if procs:
        return Check("gpu", False, f"GPU busy: {len(procs)} compute process(es); mem {mem} MiB")
    return Check("gpu", True, f"GPU {gpu} idle, mem {mem} MiB")


def _ram_free(min_gb: float) -> Check:
    try:
        meminfo = Path("/proc/meminfo").read_text()
        available_kb = next(
            int(line.split()[1]) for line in meminfo.splitlines()
            if line.startswith("MemAvailable:")
        )
        free_gb = available_kb / 1024 / 1024
    except (OSError, StopIteration):
        return Check("ram", False, "/proc/meminfo unreadable (not Linux?)", fatal=False)
    ok = free_gb >= min_gb
    return Check("ram", ok, f"{free_gb:.0f} GB available, need {min_gb:.0f} GB")


def _port_free(port: int) -> Check:
    with socket.socket() as sock:
        busy = sock.connect_ex(("127.0.0.1", port)) == 0
    return Check(f"port {port}", not busy, "in use -- kill the old server" if busy else "free")


def _disk_free(path: Path, min_gb: float = 50.0) -> Check:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    free_gb = usage.free / 1024**3
    return Check(f"disk {path}", free_gb >= min_gb, f"{free_gb:.0f} GB free")


def _redis(url: str) -> Check:
    if shutil.which("redis-cli") is None:
        return Check("redis", False, "redis-cli not found", fatal=False)
    out = subprocess.run(["redis-cli", "-u", url, "ping"], capture_output=True, text=True)
    ok = out.stdout.strip().upper() == "PONG"
    return Check("redis", ok, out.stdout.strip() or out.stderr.strip() or "no response")


def _paths(config: Config) -> list[Check]:
    checks = [
        Check("workflow repo", config.paths.workflow_repo.is_dir(), str(config.paths.workflow_repo)),
    ]
    for arm, repo in config.paths.vllm_repos.items():
        checks.append(Check(f"vllm repo [{arm}]", repo.is_dir(), str(repo)))
    return checks


def run(config: Config, arm: str | None = None) -> list[Check]:
    """All checks for `arm` (or the arm-independent ones when arm is None)."""
    checks: list[Check] = [
        Check("tmux", shutil.which("tmux") is not None, shutil.which("tmux") or "not found"),
        _gpu_free(config.gpu),
        _ram_free(config.min_free_ram_gb),
        _port_free(config.port),
        _disk_free(config.paths.bench_root),
        *_paths(config),
    ]
    if arm == "ours":
        checks.append(_redis(config.redis_url))
    if arm in {"baseline", "ours"}:
        checks.append(_port_free(config.lmcache_port))
    return checks


def blockers(checks: list[Check]) -> list[Check]:
    return [c for c in checks if not c.ok and c.fatal]
