"""The lap itself: setup, run one question, collect, tear down.

The ordering is the contract -- LMCache restarted before the server, metrics
epoch reset after the server answers and before the workflow starts, artifacts
collected before anything is killed.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from pitlane import metrics as metrics_mod
from pitlane import report, resources, stack, timeline, workflow
from pitlane.arms import Arm, Registry
from pitlane.config import Config

logger = logging.getLogger(__name__)

# Metrics grows columns; a stored file may predate some of them, and an
# unknown key would make the whole resume fall back to re-running.
_METRIC_FIELDS = {f.name for f in fields(metrics_mod.Metrics)}


@dataclass
class CellResult:
    arm: str
    question_id: str
    rep: int
    cell: Path
    metrics: metrics_mod.Metrics
    exit_code: int
    # Stopped at a question boundary on an interrupt: complete as far as it
    # got, and not a cell a resume should skip.
    drained: bool = False


def run_cell(
    config: Config,
    arm: Arm,
    question_id: str,
    rep: int,
    *,
    trace_mode: str = "pinned",
    count: int = 1,
    cache_state: str = "cold",
    keep_stack: bool = False,
    reuse_stack: bool = False,
    dry_run: bool = False,
) -> CellResult:
    cell = config.cell_dir(arm.name, question_id, rep)
    cell.mkdir(parents=True, exist_ok=True)
    logger.info("=== %s / %s / rep%s -> %s", arm.name, question_id, rep, cell)

    # A dry run touches nothing outside the cell directory: no tmux sessions,
    # no ports, no waiting on a server that was never asked to start.
    if not dry_run and not reuse_stack:
        if arm.lmcache_server:
            stack.restart_lmcache(config, arm)
        else:
            # Continuum drives LMCache in-process; a stray server on 10903
            # would be a second, invisible cache tier.
            stack.stop_lmcache(config)
        stack.start_server(config, arm, cell)

    if not dry_run:
        stack.reset_kv_metrics(config)

    monitor = resources.Monitor(
        config.run_dir / "resources.csv",
        scope={"arm": arm.name, "question_id": question_id, "rep": rep},
        gpu=config.gpu,
        disk_path=config.paths.bench_root,
        interval_s=config.resource_interval_s,
        ram_floor_gb=config.abort_free_ram_gb,
        vram_ceiling_frac=config.abort_vram_frac,
    )
    with monitor:
        result = workflow.run(
            config, arm, cell,
            question_id=question_id, count=count, trace_mode=trace_mode,
            dry_run=dry_run, abort=monitor.aborted,
            completed_log=_questions_log(config.run_dir, arm.name,
                                         question_id, rep),
        )

    collected = metrics_mod.collect(
        cell, arm=arm.name, question_id=question_id, rep=rep,
        t0=result.started_ts, t1=result.finished_ts, cache_state=cache_state,
        lead_min_s=config.prefetch_lead_min_s,
    )
    if result.aborted:
        # Ahead of the exit-code warning: "killed" explains the non-zero code,
        # and a cell that ran out of memory is not a cell with a bad number in
        # it, it is a cell with no number in it.
        collected.warnings.append(
            f"aborted on resources: {monitor.reason or 'threshold crossed'}"
        )
    if result.exit_code != 0:
        collected.warnings.append(f"workflow exited {result.exit_code}")
    if result.drained:
        collected.warnings.append(
            "stopped on interrupt after finishing the question in flight; "
            "the remaining questions of this cell have not run"
        )
    if arm.name == "ours" and collected.total_prefetches == 0:
        collected.warnings.append(
            "zero prefetches: registry likely unseeded (only /v1/agents/* registers prefixes)"
        )

    report.write_metrics(cell, collected)
    report.append_row(config.run_dir / "results.csv", collected)
    report.append_request_rows(config.run_dir / "requests.csv", collected)
    report.append_prefetch_rows(config.run_dir / "prefetches.csv", collected)
    report.append_tool_rows(config.run_dir / "tools.csv", collected)
    # After the CSVs, since both are read back out of them.
    report.write_by_question(config.run_dir)
    timeline.write_run(config.run_dir)
    report.write_summary(config.run_dir)

    if not keep_stack and not reuse_stack and not dry_run:
        stack.stop_server(config)
        if arm.lmcache_server:
            stack.stop_lmcache(config)

    for warning in collected.warnings:
        logger.warning("%s/%s: %s", arm.name, question_id, warning)
    # Last, and only on the way out: the entry a resume trusts has to mean the
    # cell got all the way here. Written after the artifacts it vouches for.
    # A drained cell exits cleanly and is still unfinished: it answered some of
    # its questions and stopped. Recorded as not-complete so a resume comes
    # back to it, which the per-question log then makes cheap.
    _record_cell(config.run_dir, arm.name, question_id, rep,
                 exit_code=result.exit_code, aborted=result.aborted,
                 drained=result.drained)
    return CellResult(arm.name, question_id, rep, cell, collected,
                      result.exit_code, drained=result.drained)


# The resume ledger. One hidden file at the top of the run, not a marker in
# every cell: a cell directory is a result, and a result should carry what was
# measured and nothing about how the run was driven. Anyone reading the tree --
# or copying one cell out of it -- sees exactly what they saw before resume
# existed.
_LEDGER = ".pitlane-progress.json"
_PROGRESS_DIR = ".pitlane"


def _questions_log(run_dir: Path, arm: str, question_id: str, rep: int) -> Path:
    """Where the workflow records the questions of this cell it has finished.

    Beside the ledger and hidden for the same reason: which questions a
    previous attempt got through is how the run was driven, not something it
    measured, and a cell directory should read the same as one produced
    without resume.
    """
    return run_dir / _PROGRESS_DIR / f"{arm}__{question_id}__rep{rep}.log"


def _cell_key(arm: str, question_id: str, rep: int) -> str:
    return f"{arm}/{question_id}/rep{rep}"


def _read_ledger(run_dir: Path) -> dict[str, dict[str, Any]]:
    try:
        body = json.loads((run_dir / _LEDGER).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return body.get("cells", {}) if isinstance(body, dict) else {}


def _record_cell(run_dir: Path, arm: str, question_id: str, rep: int,
                 *, exit_code: int, aborted: bool, drained: bool = False) -> None:
    """Note that this cell reached the end, atomically.

    Written through a temporary file and renamed, because the thing this
    records is survival of an interrupt -- a ledger torn in half by the kill it
    is meant to outlive would take every earlier cell with it.
    """
    cells = _read_ledger(run_dir)
    cells[_cell_key(arm, question_id, rep)] = {
        "exit_code": exit_code,
        "aborted": aborted,
        "drained": drained,
        "finished_at": time.time(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / f"{_LEDGER}.tmp"
    tmp.write_text(json.dumps({"cells": cells}, indent=2) + "\n")
    tmp.replace(run_dir / _LEDGER)


def completed(run_dir: Path, arm: str, question_id: str, rep: int,
              cell: Path) -> bool:
    """Whether this cell finished cleanly enough to skip on a resume.

    Deliberately strict. A cell is worth keeping only if the workflow exited 0,
    it was not aborted on resources, and the collection it produced is still on
    disk -- anything else is re-run, because an eight-minute boot is cheaper
    than a number nobody can account for.

    A cell that was *mid-flight* when the run died has no ledger entry, which
    is the case this exists to catch: its directory looks complete and its
    numbers are half a question.
    """
    entry = _read_ledger(run_dir).get(_cell_key(arm, question_id, rep))
    if entry is None or not (cell / "metrics.json").exists():
        return False
    return (entry.get("exit_code") == 0
            and not entry.get("aborted")
            and not entry.get("drained"))


def _resumed(config: Config, arm: Arm, question_id: str, rep: int,
             cell: Path) -> CellResult | None:
    """The stored result for a finished cell, or None if it must be re-run.

    A cell that has to be re-run first has its earlier rows removed from the
    run-level CSVs. They were appended by the attempt that failed, and every
    reader downstream would otherwise count the cell twice.
    """
    if completed(config.run_dir, arm.name, question_id, rep, cell):
        try:
            stored = metrics_mod.Metrics(**{
                key: value
                for key, value in json.loads((cell / "metrics.json").read_text()).items()
                if key in _METRIC_FIELDS
            })
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        logger.info("=== %s / %s / rep%s -- done, skipping",
                    arm.name, question_id, rep)
        return CellResult(arm.name, question_id, rep, cell, stored, 0)

    for name in ("results.csv", "requests.csv", "prefetches.csv", "tools.csv"):
        dropped = report.drop_cell_rows(
            config.run_dir / name, arm.name, question_id, rep)
        if dropped:
            logger.info("resume: dropped %d stale row(s) from %s", dropped, name)
    return None


_BATCH_ID = re.compile(r"^batch(\d+)$")


def question_count(question_id: str, override: int | None = None) -> int:
    """How many questions the workflow should run for this cell.

    The workflow cannot be asked for a *specific* question -- ODR selects with
    `random.Random(0).sample(examples, N)` -- so N is the only handle there is,
    and it has to match the N the trace was recorded at. The sample for N=1 is
    not a subset of the sample for N=2, so a mismatch is not a smaller run: it
    is a different question, and every request misses the trace.

    `batch<N>` carries its own N, which is the convention the runbook already
    uses. Anything else is one question unless `--count` says otherwise.
    """
    if override is not None:
        return override
    match = _BATCH_ID.match(question_id)
    return int(match.group(1)) if match else 1


def _cache_state(count: int, arm: Arm) -> str:
    """`cold` unless later questions in this cell inherit an earlier one's HBM.

    The flag alone is not enough to claim it. `--hbm-flush-between-queries` is
    gated in the workflow on `--no-kv-metrics-reset`, because the flush is a
    parameter of the reset endpoint -- so an arm that disables resets passes the
    flag and flushes nothing, and trusting the flag would report those cells as
    isolated when they are not.
    """
    if count == 1:
        return "cold"
    args = arm.workflow_args
    flushes = ("--hbm-flush-between-queries" in args
               and "--no-kv-metrics-reset" not in args)
    return "cold" if flushes else "warm"


def run_matrix(
    config: Config,
    registry: Registry,
    *,
    arms: list[str],
    questions: list[str],
    reps: int = 1,
    count: int | None = None,
    dry_run: bool = False,
    keep_stack: bool = False,
    resume: bool = False,
) -> list[CellResult]:
    """Interleave arms within a question: A, B, C, A, B, C, ...

    Blocking by arm would load any drift in machine state onto whichever arm
    ran last, which is exactly the bias the study is trying to avoid.
    """
    results: list[CellResult] = []
    planned = len(questions) * reps * len(arms)
    for question in questions:
        cell_count = question_count(question, count)
        for rep in range(1, reps + 1):
            for arm_name in arms:
                arm = registry[arm_name]
                if resume:
                    cell = config.cell_dir(arm.name, question, rep)
                    done = _resumed(config, arm, question, rep, cell)
                    if done is not None:
                        results.append(done)
                        continue
                else:
                    # Not a resume: this cell starts from nothing. A questions
                    # log left by an earlier run under the same id would
                    # otherwise make the workflow skip questions it was asked
                    # to run, and the cell would report a batch it never
                    # executed.
                    _questions_log(config.run_dir, arm.name, question,
                                   rep).unlink(missing_ok=True)
                started = time.time()
                results.append(
                    run_cell(
                        config, registry[arm_name], question, rep,
                        count=cell_count,
                        # A batch cell's later questions are only warm if the
                        # workflow does not flush between them. With the flush
                        # on every question starts cold, and saying otherwise
                        # would make the reporter refuse to average cells that
                        # are in fact comparable with isolated ones.
                        cache_state=_cache_state(cell_count, registry[arm_name]),
                        dry_run=dry_run, keep_stack=keep_stack,
                    )
                )
                logger.info("cell took %.0fs", time.time() - started)
                if results[-1].drained:
                    logger.warning(
                        "stopped after %d of %d cell(s); the question in "
                        "flight finished and was recorded",
                        len(results), planned,
                    )
                    return results
    return results
