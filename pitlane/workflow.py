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


def _accepts_completed_log(repo: Path, script: str) -> bool:
    """Whether this arm's script takes `--completed-log`.

    Sniffed from the script the arm actually runs, for the same reason the
    adapter is: the repo is the thing that changed. An older checkout, or one
    of the two evaluation scripts patched and not the other, then runs exactly
    as before instead of dying on an unknown flag.
    """
    try:
        return "--completed-log" in (repo / script).read_text()
    except OSError:
        return False


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
    # The workflow was asked to stop at a question boundary and did. The cell
    # is intact but incomplete: the questions it answered are recorded, the
    # rest are not, and a resume picks up from there.
    drained: bool = False


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
    completed_log: Path | None = None,
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
    env.update(arm.resolved_env("workflow", repo=repo, cell=cell,
                                model=config.model_name))
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

        # The date has to match the recording's prompts, which interpolate
        # `get_today_str()` -- replay on another date and every prompt prefix
        # differs, so the first request misses and `strict` ends the run.
        #
        # An explicit `ODR_FROZEN_DATE` wins. It is the operator saying which
        # date the recording was made under, and they can be right when the
        # trace cannot answer: a recording frozen to a date carries that date
        # in its prompts while its wall stamps say when it actually ran.
        recorded, source = trace_mod.recorded_date(config.trace_path)
        explicit = env.get("ODR_FROZEN_DATE", "").strip()
        if explicit:
            if recorded and recorded != explicit:
                # Worth saying either way round: if the trace's prompts really
                # do read `recorded`, this run will miss on the first request.
                logger.warning(
                    "ODR_FROZEN_DATE=%s but the trace's %s says %s; using the "
                    "explicit value", explicit, source, recorded,
                )
            env["ODR_FROZEN_DATE"] = explicit
            logger.info("frozen date %s (explicit)", explicit)
        elif recorded:
            env["ODR_FROZEN_DATE"] = recorded
            logger.info("frozen date %s (from the trace's %s)", recorded, source)
        else:
            logger.warning(
                "no ODR_FROZEN_DATE and none readable from %s; prompts will "
                "use today's date, which misses unless the recording was made "
                "today and unfrozen", config.trace_path,
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

    if env.get("LANGGRAPH_VLLM_AGENT_ENABLE", "").strip() not in ("", "0", "false"):
        # `BackgroundVLLMAgentWorker.enabled` is `base_url and model and
        # enabled`, and it is a property with no error path: without a model the
        # worker reports itself disabled and the predictor stops silently. No
        # prefetches from langgraph, no `min_lead` markers, and nothing in the
        # log to say why -- which is exactly how this went unnoticed.
        if not (env.get("LANGGRAPH_VLLM_AGENT_MODEL", "").strip()
                or env.get("OPENAI_MODEL", "").strip()):
            logger.warning(
                "LANGGRAPH_VLLM_AGENT_ENABLE is set but no agent model is; "
                "langgraph's predictor will report itself disabled and issue "
                "nothing",
            )

    command = [
        config_mod.venv_bin(config.paths.workflow_venv, "python"), arm.workflow_script,
        *adapter.selection_args(count, question_id),
        *arm.workflow_args,
    ]
    # Per-question resume inside a batch cell. A 50-question cell is one
    # workflow process, so without this an interrupt at question 40 costs all
    # 40: the cell is the unit pitlane can skip, and the questions inside it
    # are only the workflow's to track. Passed only where the workflow
    # advertises the flag, so an older checkout still runs.
    if completed_log is not None and _accepts_completed_log(repo, arm.workflow_script):
        command += ["--completed-log", str(completed_log)]
    log_path = cell / "workflow.log"
    (cell / "command.txt").write_text(" ".join(shlex.quote(c) for c in command) + "\n")

    if dry_run:
        logger.info("[dry-run] %s (cwd=%s)", " ".join(command), repo)
        return WorkflowResult(0, time.time(), time.time(), log_path)

    started = time.time()
    # The runner's own stamp for TTFT: the question is dispatched now, and
    # nothing downstream records that instant.
    #
    # Kept from the first attempt when there is one. The collector windows the
    # cell as [this stamp, finish], and a resumed cell's stats directory holds
    # the rows of every attempt -- restamping would put the earlier questions
    # before the window and silently drop them, which is the same shape as a
    # run that answered fewer questions.
    launched = started
    stamp = cell / "question_started_ts"
    if not stamp.exists():
        stamp.write_text(f"{started!r}\n")
    else:
        try:
            started = float(stamp.read_text().strip())
        except ValueError:
            stamp.write_text(f"{started!r}\n")
    child_env = config_mod.venv_env(
        config.paths.workflow_venv, {**dict(_os_environ()), **env}
    )
    was_aborted = False
    draining = False
    # Appended for the same reason as the server log: a resumed cell is one
    # cell, and the questions the first attempt answered are part of it.
    with log_path.open("a") as log:
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
            except KeyboardInterrupt:
                # The child has its own session, so a terminal's Ctrl-C reaches
                # pitlane and not the workflow -- which is what makes a clean
                # stop possible at all. Forwarded, so the workflow can finish
                # the question it is on and record it; killing here would throw
                # away a question that is minutes from done and leave the batch
                # with work it paid for and cannot report.
                #
                # The second interrupt is not caught: it propagates, teardown
                # runs, and the question in flight is lost -- which is what
                # pressing it twice asks for.
                if not draining:
                    draining = True
                    logger.warning(
                        "interrupt: asking the workflow to finish the question "
                        "in flight and stop. Ctrl-C again to kill it now."
                    )
                    _signal_group(process, signal.SIGINT)
                continue
            if abort is not None and abort.is_set() and not was_aborted:
                was_aborted = True
                logger.error("resource abort; terminating the workflow")
                _terminate(process)

    finished = time.time()
    # This attempt's own duration. `started` may belong to an earlier attempt
    # of a resumed cell, which is right for the metrics window and wrong for
    # saying how long this process ran.
    logger.info("workflow exited %s after %.1fs",
                process.returncode, finished - launched)
    return WorkflowResult(
        process.returncode, started, finished, log_path, aborted=was_aborted,
        drained=draining,
    )


def _signal_group(process: "subprocess.Popen", signum: int) -> None:
    """Send one signal to the workflow's whole process group.

    The group, not the process: ODR runs its researchers as children, and a
    signal to the parent alone would leave them working against a server that
    is about to go away.
    """
    try:
        os.killpg(os.getpgid(process.pid), signum)
    except (ProcessLookupError, PermissionError) as exc:
        logger.warning("could not signal the workflow: %s", exc)


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
