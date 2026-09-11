"""Bringing a serving stack up and down: LMCache, then vLLM, then the gate.

The ordering rules here are the ones that were easy to get wrong by hand:
wiping the LMCache L2 dir is inseparable from restarting its server, and a
server is never "up" because a sleep expired -- only because it answered.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from pitlane import tmux
from pitlane.arms import Arm
from pitlane import config as config_mod
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
def lmcache_venv(config: Config, arm: Arm | None = None) -> Path | None:
    """Where `lmcache` is installed: its own venv, else the arm's own.

    Falling back to the arm's venv is right more often than PATH is -- vLLM
    imports LMCache for the connector, so a build's environment already has a
    compatible server -- but `LMCACHE_VENV` exists because "usually" is not
    "always" and a shared server may be installed once elsewhere.
    """
    if config.paths.lmcache_venv is not None:
        return config.paths.lmcache_venv
    if arm is not None:
        return config.paths.venvs.get(arm.venv or arm.repo)
    return None


def restart_lmcache(config: Config, arm: Arm | None = None, *,
                    l1_size_gb: int = 200) -> None:
    """Stop, wipe, start -- in that order, always together.

    The L1 index is in memory and the L2 store is on disk. Wiping the disk
    under a live server leaves it serving keys whose backing files are gone;
    restarting without wiping carries the previous cell's cache into this one.

    With `LMCACHE_L2_DIR` unset there is no disk tier: `--l2-adapter` is
    omitted, LMCache keeps its whole store in L1, and the restart alone is the
    wipe. That is a smaller change to what is measured than it sounds -- L1 is
    the 200 GB of CPU memory that serves `external_hit_tokens`, and L2 only
    holds what spills past it, which a question of a few tens of thousands of
    prompt tokens never reaches. It does mean nothing survives a restart, so
    an arm that wants a warm cache across cells needs the disk tier.
    """
    tmux.kill(LMCACHE_SESSION)
    _free_port(config.lmcache_port)

    venv = lmcache_venv(config, arm)
    target = config.paths.lmcache_l2_dir
    adapter_arg = ""
    if target is not None:
        if target.exists():
            shutil.rmtree(target)
            logger.info("wiped LMCache L2 dir %s", target)
        target.mkdir(parents=True, exist_ok=True)
        adapter = json.dumps({"type": "fs", "base_path": str(target)})
        adapter_arg = f" --l2-adapter {shlex.quote(adapter)}"
    else:
        logger.info("LMCACHE_L2_DIR unset; running LMCache with L1 only")

    command = (
        f"LMCACHE_LOG_KV_HASH=1 {shlex.quote(config_mod.venv_bin(venv, 'lmcache'))}"
        " server"
        f" --l1-size-gb {l1_size_gb}"
        " --eviction-policy LRU"
        " --chunk-size 16"
        " --host 0.0.0.0"
        f" --port {config.lmcache_port}"
        f"{adapter_arg}"
    )
    tmux.start(LMCACHE_SESSION, command, env=config_mod.venv_env(venv))
    _wait_port_open(config.lmcache_port, timeout_s=120)
    logger.info("LMCache up on port %s", config.lmcache_port)


def stop_lmcache(config: Config | None = None) -> None:
    """Kill the session and wait for the port, like `stop_server` already did.

    Killing the tmux session returns immediately; the process still has to run
    down and release 10903. The next arm's preflight starts within
    milliseconds, sees the port bound and refuses to run -- so a matrix would
    complete its first arm and fail every one after it, reporting a stale
    server as the operator's fault.
    """
    tmux.kill(LMCACHE_SESSION)
    if config is not None:
        _free_port(config.lmcache_port)


# -- vLLM ------------------------------------------------------------------
def start_server(config: Config, arm: Arm, cell: Path) -> Path:
    """Launch the arm's vLLM and block until it serves /v1/models."""
    repo = config.paths.vllm_repos[arm.repo]
    venv = config.paths.venvs.get(arm.venv)
    log_path = cell / arm.log_name
    stats_dir = cell / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    env = arm.resolved_env("server", repo=repo, cell=cell)
    env["VLLM_REQUEST_STATS_DIR"] = str(stats_dir)
    # Each build lives in its own virtualenv; `vllm` on PATH is whichever one
    # the shell activated, which for a multi-arm run is the wrong one twice out
    # of three times.
    env = config_mod.venv_env(venv, env)
    env = {k: v for k, v in env.items() if v != ""}

    args = " ".join(
        shlex.quote(a) for a in arm.resolved_server_args(repo=repo, cell=cell)
    )
    command = (
        f"{shlex.quote(config_mod.venv_bin(venv, 'vllm'))} "
        f"serve {shlex.quote(config.model_name)} --port {config.port} "
        f"{args} > {shlex.quote(str(log_path))} 2>&1"
    )
    tmux.kill(VLLM_SESSION)
    # Reclaim rather than merely wait: a previous run killed by Ctrl-C leaves an
    # orphan that will never exit, and these ports are pitlane's own.
    _free_port(config.port, grace_s=30, kill_s=30)
    tmux.start(VLLM_SESSION, command, cwd=str(repo), env=env)
    wait_ready(config, log_path)
    return log_path


def stop_server(config: Config) -> None:
    """Kill the session and make sure the server is actually gone.

    VRAM is released asynchronously, so the next arm's boot fails confusingly if
    it starts while the old process is still tearing down -- but waiting is only
    correct while something is still tearing down. An orphaned server never
    exits, and the old 300 s wait sat there watching it.
    """
    tmux.kill(VLLM_SESSION)
    _free_port(config.port, grace_s=60, kill_s=30)


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


_SS_PID = re.compile(r"pid=(\d+)")


def _listeners(port: int) -> list[int]:
    """PIDs listening on `port`, via whichever of lsof/ss is installed."""
    if shutil.which("lsof"):
        out = subprocess.run(
            ["lsof", "-t", "-i", f":{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True,
        )
        pids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        if pids:
            return pids
    if shutil.which("ss"):
        out = subprocess.run(
            ["ss", "-lptnH", f"sport = :{port}"], capture_output=True, text=True,
        )
        return sorted({int(m) for m in _SS_PID.findall(out.stdout)})
    return []


def _signal_listeners(port: int, sig: int) -> int:
    """Signal the *process group* of everything listening on `port`.

    The group, not the process. `tmux.start` sends its command with `send-keys`,
    so the pane holds a shell and the server is its child -- `kill-session`
    reaps the shell and orphans the server, which goes on holding the port. The
    server's own children are worse: vLLM's EngineCore is a separate process
    that holds no port at all, so signalling only the listener leaves it alive
    with the GPU memory still mapped, and the next arm boots onto a card that
    is not free. Job control puts the whole tree in one group, so the group is
    the unit that actually corresponds to "this server".
    """
    own_group = os.getpgid(0)
    signalled = 0
    for pid in _listeners(port):
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if group == own_group:
            # Would take pitlane down with it. Only reachable if a server was
            # started outside tmux from this very shell.
            logger.warning("pid %d on port %d shares our process group; "
                           "not signalling", pid, port)
            continue
        try:
            os.killpg(group, sig)
            signalled += 1
        except (ProcessLookupError, PermissionError) as exc:
            logger.warning("could not signal group %d on port %d: %s",
                           group, port, exc)
    return signalled


def _free_port(port: int, *, grace_s: float = 20.0, kill_s: float = 20.0) -> None:
    """Get `port` released, escalating only as far as it has to.

    Waiting alone is what the stop paths used to do, and it hangs for the full
    timeout against an orphan that is never going to exit -- which is a worse
    failure than not stopping at all, because it looks like the shutdown is
    progressing.
    """
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not _port_open(port):
            return
        time.sleep(1)

    if _signal_listeners(port, signal.SIGTERM):
        logger.info("port %d still held; sent SIGTERM", port)
    deadline = time.time() + kill_s
    while time.time() < deadline:
        if not _port_open(port):
            return
        time.sleep(1)

    if _signal_listeners(port, signal.SIGKILL):
        logger.warning("port %d still held; sent SIGKILL", port)
    deadline = time.time() + 10
    while time.time() < deadline:
        if not _port_open(port):
            return
        time.sleep(1)
    raise StackError(
        f"port {port} still in use after SIGKILL; something is holding it that "
        f"pitlane did not start"
    )


def down(config: Config) -> None:
    """Tear down everything pitlane starts, and confirm the ports are free.

    For the state a Ctrl-C leaves behind. Killing the tmux sessions is not
    enough on its own -- that is the whole reason the orphans exist -- so this
    goes through the same escalation the stop paths use, which also releases
    the VRAM an orphaned EngineCore is still holding.
    """
    tmux.kill(VLLM_SESSION)
    tmux.kill(LMCACHE_SESSION)
    for port, label in ((config.port, "vllm"), (config.lmcache_port, "lmcache")):
        if _port_open(port):
            logger.info("reclaiming port %d (%s)", port, label)
            _free_port(port, grace_s=5, kill_s=15)
        else:
            logger.info("port %d (%s) already free", port, label)
