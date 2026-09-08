"""Bringing a serving stack up and down: LMCache, then vLLM, then the gate.

The ordering rules here are the ones that were easy to get wrong by hand:
wiping the LMCache L2 dir is inseparable from restarting its server, and a
server is never "up" because a sleep expired -- only because it answered.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from pitlane import tmux
from pitlane.arms import Arm
from pitlane.config import Config

logger = logging.getLogger(__name__)

LMCACHE_SESSION = "lmcache"
VLLM_SESSION = "vllm"

# Lines that mean the boot is dead; polling for readiness past one of these
# just burns the timeout.
_FATAL_LOG_PATTERNS = (
    "Traceback (most recent call last)",
    "torch.OutOfMemoryError",
    "CUDA out of memory",
    "Address already in use",
    "ValueError: No available memory for the cache blocks",
    "EngineCore failed to start",
)


class StackError(RuntimeError):
    pass


# -- LMCache ---------------------------------------------------------------
def restart_lmcache(config: Config, *, l1_size_gb: int = 200) -> None:
    """Stop, wipe, start -- in that order, always together.

    The L1 index is in memory and the L2 store is on disk. Wiping the disk
    under a live server leaves it serving keys whose backing files are gone;
    restarting without wiping carries the previous cell's cache into this one.
    """
    tmux.kill(LMCACHE_SESSION)
    _wait_port_free(config.lmcache_port, timeout_s=60)

    target = config.paths.lmcache_l2_dir
    if target.exists():
        shutil.rmtree(target)
        logger.info("wiped LMCache L2 dir %s", target)
    target.mkdir(parents=True, exist_ok=True)

    adapter = json.dumps({"type": "fs", "base_path": str(target)})
    command = (
        "LMCACHE_LOG_KV_HASH=1 lmcache server"
        f" --l1-size-gb {l1_size_gb}"
        " --eviction-policy LRU"
        " --chunk-size 16"
        " --host 0.0.0.0"
        f" --port {config.lmcache_port}"
        f" --l2-adapter {shlex.quote(adapter)}"
    )
    tmux.start(LMCACHE_SESSION, command)
    _wait_port_open(config.lmcache_port, timeout_s=120)
    logger.info("LMCache up on port %s", config.lmcache_port)


def stop_lmcache() -> None:
    tmux.kill(LMCACHE_SESSION)


# -- vLLM ------------------------------------------------------------------
def start_server(config: Config, arm: Arm, cell: Path) -> Path:
    """Launch the arm's vLLM and block until it serves /v1/models."""
    repo = config.paths.vllm_repos[arm.repo]
    log_path = cell / arm.log_name
    stats_dir = cell / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    env = arm.resolved_env("server", repo=repo, cell=cell)
    env["VLLM_REQUEST_STATS_DIR"] = str(stats_dir)
    env = {k: v for k, v in env.items() if v != ""}

    args = " ".join(
        shlex.quote(a) for a in arm.resolved_server_args(repo=repo, cell=cell)
    )
    command = (
        f"vllm serve {shlex.quote(config.model_name)} --port {config.port} "
        f"{args} > {shlex.quote(str(log_path))} 2>&1"
    )
    tmux.kill(VLLM_SESSION)
    _wait_port_free(config.port, timeout_s=120)
    tmux.start(VLLM_SESSION, command, cwd=str(repo), env=env)
    wait_ready(config, log_path)
    return log_path


def stop_server(config: Config) -> None:
    tmux.kill(VLLM_SESSION)
    # VRAM is released asynchronously; the next arm's boot fails confusingly if
    # it starts while the old process is still tearing down.
    _wait_port_free(config.port, timeout_s=300)


def wait_ready(config: Config, log_path: Path) -> None:
    deadline = time.time() + config.server_ready_timeout_s
    url = config.base_url("/v1/models")
    while time.time() < deadline:
        if _http_ok(url):
            logger.info("server ready at %s", url)
            return
        fatal = _fatal_line(log_path)
        if fatal:
            raise StackError(f"server boot failed: {fatal}")
        time.sleep(5)
    raise StackError(
        f"server not ready within {config.server_ready_timeout_s:.0f}s; see {log_path}"
    )


def reset_kv_metrics(config: Config) -> bool:
    """Start a fresh metric epoch. False when the server does not support it."""
    request = urllib.request.Request(config.base_url("/v1/kv_metrics/reset"), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read() or b"{}")
        if body.get("ok") is False:
            logger.warning("kv_metrics reset declined: %s", body.get("reason"))
            return False
        return True
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("kv_metrics reset failed: %s", exc)
        return False


# -- helpers ---------------------------------------------------------------
def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError):
        return False


def _fatal_line(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    tail = log_path.read_text(errors="replace")[-20000:]
    for pattern in _FATAL_LOG_PATTERNS:
        if pattern in tail:
            for line in tail.splitlines():
                if pattern in line:
                    return line.strip()[:300]
    return None


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_port_open(port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _port_open(port):
            return
        time.sleep(1)
    raise StackError(f"nothing listening on port {port} after {timeout_s:.0f}s")


def _wait_port_free(port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _port_open(port):
            return
        time.sleep(2)
    raise StackError(f"port {port} still in use after {timeout_s:.0f}s")
