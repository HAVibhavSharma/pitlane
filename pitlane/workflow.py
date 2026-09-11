"""Workflow adapters -- how to ask a repo to run exactly one question.

ODR and swe-agent expose the same two scripts under tests/ but spell their
flags differently, and that is the whole of the difference: one counts
"queries", the other "instances".
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from pitlane.arms import Arm
from pitlane import config as config_mod
from pitlane import trace as trace_mod
from pitlane.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Adapter:
    name: str
    count_flag: str          # how many questions/instances to run
    reps_flag: str           # completions per question/instance
    select_flag: str | None  # pick specific ids, when supported

    def selection_args(self, count: int, question_id: str | None) -> list[str]:
        args = [self.count_flag, str(count), self.reps_flag, "1"]
        if question_id and self.select_flag:
            args += [self.select_flag, question_id]
        return args


ODR = Adapter("odr", "--max-queries", "--completions-per-query", None)
SWE = Adapter("swe-agent", "--max-instances", "--completions-per-instance", "--instance-ids")


def detect(repo: Path) -> Adapter:
    """Pick the adapter by the flags the repo's script actually accepts.

    Sniffing the source beats a config switch: the repo is the thing that
    changed, so it should be the thing that decides.
    """
    script = repo / "tests" / "run_evaluate.py"
    if not script.exists():
        raise FileNotFoundError(f"no tests/run_evaluate.py under {repo}")
    text = script.read_text()
    if "--max-instances" in text:
        return SWE
    if "--max-queries" in text:
        return ODR
    raise ValueError(f"cannot tell which workflow {script} is: no known count flag")


@dataclass
class WorkflowResult:
    exit_code: int
    started_ts: float
    finished_ts: float
    log_path: Path
    aborted: bool = False


def run(
    config: Config,
    arm: Arm,
    cell: Path,
    *,
    question_id: str | None,
    count: int = 1,
    trace_mode: str = "pinned",
    abort: "threading.Event | None" = None,
    extra_env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> WorkflowResult:
    """Run one question and return when it is done.

    Unlike the servers, the workflow runs as a child process: its exit code is
    the run's verdict, and a pane sentinel is a poor way to learn it. Run
    pitlane itself inside tmux if the ssh connection is unreliable.
    """
    repo = config.paths.workflow_repo
    adapter = detect(repo)
    cell.mkdir(parents=True, exist_ok=True)

    env = dict(config.env)
    env.update(arm.resolved_env("workflow", repo=repo, cell=cell))
    env.update({
        "ODR_TRACE_MODE": trace_mode,
        "ODR_TRACE_PATH": str(config.trace_path),
        "ODR_TRACE_REPORT": str(cell / "divergence.jsonl"),
        "OPENAI_BASE_URL": config.base_url("/v1"),
    })
    if trace_mode == "pinned":
        # A miss means the trajectory left the recording; the cell's numbers
        # would not be comparable, so fail rather than quietly go live.
        env["ODR_TRACE_MODE"] = "pinned"
        env["ODR_TRACE_ON_MISS"] = "strict"

        # Pin the date to the recording's, not to the operator's env. Several
        # prompts interpolate `get_today_str()`, so replaying on a different
        # day changes every prompt prefix and misses on the first request --
        # which reads as a diverged trajectory rather than as a stale variable.
        # The trace knows its own date, so nothing has to be kept in step by
        # hand.
        recorded = trace_mod.recorded_date(config.trace_path)
        if recorded:
            existing = env.get("ODR_FROZEN_DATE", "").strip()
            if existing and existing != recorded:
                logger.warning(
                    "ODR_FROZEN_DATE=%s disagrees with the trace (%s); using "
                    "the trace's", existing, recorded,
                )
            env["ODR_FROZEN_DATE"] = recorded
        else:
            logger.warning(
                "could not read a date from %s; prompts will use today's, "
                "which misses unless the trace was recorded today",
                config.trace_path,
            )

        jobs = trace_mod.recorded_jobs(config.trace_path)
        if jobs and len(jobs) != count:
            # Not fatal here -- `--count` may legitimately be probing -- but it
            # is the single most likely reason a replay misses every request,
            # and it is invisible in the workflow's own output.
            logger.warning(
                "trace holds %d question(s) but this cell asks for %d; the "
                "workflow samples N questions, so a different N is a different "
                "set and every request will miss",
                len(jobs), count,
            )
    elif trace_mode == "record" and not env.get("ODR_FROZEN_DATE", "").strip():
        # Pin the recording's own date too. Unpinned, a record run that crosses
        # midnight writes two different prompt prefixes into one trace, and no
        # later replay can satisfy both.
        env["ODR_FROZEN_DATE"] = datetime.now().strftime("%Y-%m-%d")
        logger.info("pinned ODR_FROZEN_DATE=%s for this recording",
                    env["ODR_FROZEN_DATE"])
    env.update(extra_env or {})
    for key in arm.unset_keys("workflow"):
        env.pop(key, None)
    env = {k: v for k, v in env.items() if v != ""}

    command = [
        config_mod.venv_bin(config.paths.workflow_venv, "python"), arm.workflow_script,
        *adapter.selection_args(count, question_id),
        *arm.workflow_args,
    ]
    log_path = cell / "workflow.log"
    (cell / "command.txt").write_text(" ".join(shlex.quote(c) for c in command) + "\n")

    if dry_run:
        logger.info("[dry-run] %s (cwd=%s)", " ".join(command), repo)
        return WorkflowResult(0, time.time(), time.time(), log_path)

    started = time.time()
    # The runner's own stamp for TTFT: the question is dispatched now, and
    # nothing downstream records that instant.
    (cell / "question_started_ts").write_text(f"{started!r}\n")
    child_env = config_mod.venv_env(
        config.paths.workflow_venv, {**dict(_os_environ()), **env}
    )
    was_aborted = False
    with log_path.open("w") as log:
        # Popen rather than `subprocess.run`, so the wait is interruptible. The
        # child gets its own session, which makes the whole workflow tree one
        # signalling unit -- ODR spawns researchers, and terminating only the
        # parent would leave them running against a server about to be killed.
        process = subprocess.Popen(
            command, cwd=repo, env=child_env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        while True:
            try:
                process.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                pass
            if abort is not None and abort.is_set() and not was_aborted:
                was_aborted = True
                logger.error("resource abort; terminating the workflow")
                _terminate(process)

    finished = time.time()
    logger.info("workflow exited %s after %.1fs", process.returncode, finished - started)
    return WorkflowResult(
        process.returncode, started, finished, log_path, aborted=was_aborted,
    )


def _terminate(process: "subprocess.Popen", grace_s: float = 20.0) -> None:
    """SIGTERM the workflow's process group, then SIGKILL what is left."""
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(group, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(group, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _os_environ() -> dict[str, str]:
    import os
    return dict(os.environ)
