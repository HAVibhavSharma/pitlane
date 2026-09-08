"""tmux session management.

Long-lived processes (LMCache, vLLM) go in detached tmux sessions rather than
as children of this process: they must outlive a dropped ssh connection, and a
human needs to be able to attach and watch one mid-run. Nothing here needs
sudo -- tmux only wants a writable socket dir.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time

logger = logging.getLogger(__name__)


class TmuxError(RuntimeError):
    pass


def _tmux(*args: str, check: bool = True) -> str:
    result = subprocess.run(("tmux", *args), capture_output=True, text=True)
    if check and result.returncode != 0:
        raise TmuxError(f"tmux {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def available() -> bool:
    try:
        _tmux("-V")
        return True
    except (TmuxError, FileNotFoundError):
        return False


def has_session(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name], capture_output=True
    ).returncode == 0


def kill(name: str) -> None:
    if has_session(name):
        _tmux("kill-session", "-t", name, check=False)


def start(name: str, command: str, cwd: str | None = None, env: dict[str, str] | None = None) -> None:
    """Create a detached session and run one command in it.

    The command is sent with send-keys rather than passed to new-session so the
    session survives the command exiting -- a crashed server leaves its output
    on screen instead of vanishing, which is most of the diagnosis.
    """
    kill(name)
    args = ["new-session", "-d", "-s", name]
    if cwd:
        args += ["-c", cwd]
    _tmux(*args)
    for key, value in (env or {}).items():
        _tmux("send-keys", "-t", name, f"export {key}={shlex.quote(value)}", "Enter")
    _tmux("send-keys", "-t", name, command, "Enter")
    logger.info("tmux[%s]: %s", name, command.split("\n")[0][:120])


def capture(name: str, lines: int = 200) -> str:
    """Last `lines` of the session's pane, wrapped lines joined."""
    if not has_session(name):
        return ""
    return _tmux("capture-pane", "-pJ", "-S", f"-{lines}", "-t", name, check=False)


def run_to_completion(
    name: str,
    command: str,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    poll_s: float = 5.0,
    timeout_s: float | None = None,
) -> int:
    """Run a command in a session and block until it exits; return its status.

    send-keys is asynchronous, so completion is detected by echoing the exit
    code after the command with a sentinel -- the pane is the only channel a
    detached session offers.
    """
    sentinel = f"__pitlane_done_{int(time.time() * 1000)}__"
    start(name, f"{command}; echo {sentinel}$?", cwd=cwd, env=env)
    deadline = time.time() + timeout_s if timeout_s else None
    while True:
        pane = capture(name, lines=400)
        for line in pane.splitlines():
            stripped = line.strip()
            # The typed command line contains the sentinel too; only the echoed
            # one has a status code glued to it.
            if stripped.startswith(sentinel) and stripped[len(sentinel):].isdigit():
                return int(stripped[len(sentinel):])
        if deadline and time.time() > deadline:
            raise TmuxError(f"session {name}: command did not finish within {timeout_s}s")
        time.sleep(poll_s)
