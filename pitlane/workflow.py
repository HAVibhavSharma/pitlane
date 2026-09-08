"""Workflow adapters -- how to ask a repo to run exactly one question.

ODR and swe-agent expose the same two scripts under tests/ but spell their
flags differently, and that is the whole of the difference: one counts
"queries", the other "instances".
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from pitlane.arms import Arm
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


def run(
    config: Config,
    arm: Arm,
    cell: Path,
    *,
    question_id: str | None,
    count: int = 1,
    trace_mode: str = "pinned",
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
    env.update(extra_env or {})
    for key in arm.unset_keys("workflow"):
        env.pop(key, None)
    env = {k: v for k, v in env.items() if v != ""}

    command = [
        "python", arm.workflow_script,
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
    with log_path.open("w") as log:
        process = subprocess.run(
            command, cwd=repo, env={**dict(_os_environ()), **env},
            stdout=log, stderr=subprocess.STDOUT,
        )
    finished = time.time()
    logger.info("workflow exited %s after %.1fs", process.returncode, finished - started)
    return WorkflowResult(process.returncode, started, finished, log_path)


def _os_environ() -> dict[str, str]:
    import os
    return dict(os.environ)
