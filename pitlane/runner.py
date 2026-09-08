"""The lap itself: setup, run one question, collect, tear down.

The ordering is the contract -- LMCache restarted before the server, metrics
epoch reset after the server answers and before the workflow starts, artifacts
collected before anything is killed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from pitlane import metrics as metrics_mod
from pitlane import report, stack, workflow
from pitlane.arms import Arm, Registry
from pitlane.config import Config

logger = logging.getLogger(__name__)


@dataclass
class CellResult:
    arm: str
    question_id: str
    rep: int
    cell: Path
    metrics: metrics_mod.Metrics
    exit_code: int


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
            stack.restart_lmcache(config)
        else:
            # Continuum drives LMCache in-process; a stray server on 10903
            # would be a second, invisible cache tier.
            stack.stop_lmcache()
        stack.start_server(config, arm, cell)

    if not dry_run:
        stack.reset_kv_metrics(config)

    result = workflow.run(
        config, arm, cell,
        question_id=question_id, count=count, trace_mode=trace_mode, dry_run=dry_run,
    )

    collected = metrics_mod.collect(
        cell, arm=arm.name, question_id=question_id, rep=rep,
        t0=result.started_ts, t1=result.finished_ts, cache_state=cache_state,
    )
    if result.exit_code != 0:
        collected.warnings.append(f"workflow exited {result.exit_code}")
    if arm.name == "ours" and collected.total_prefetches == 0:
        collected.warnings.append(
            "zero prefetches: registry likely unseeded (only /v1/agents/* registers prefixes)"
        )

    report.write_metrics(cell, collected)
    report.append_row(config.run_dir / "results.csv", collected)
    report.write_summary(config.run_dir)

    if not keep_stack and not reuse_stack and not dry_run:
        stack.stop_server(config)
        if arm.lmcache_server:
            stack.stop_lmcache()

    for warning in collected.warnings:
        logger.warning("%s/%s: %s", arm.name, question_id, warning)
    return CellResult(arm.name, question_id, rep, cell, collected, result.exit_code)


def run_matrix(
    config: Config,
    registry: Registry,
    *,
    arms: list[str],
    questions: list[str],
    reps: int = 1,
    dry_run: bool = False,
    keep_stack: bool = False,
) -> list[CellResult]:
    """Interleave arms within a question: A, B, C, A, B, C, ...

    Blocking by arm would load any drift in machine state onto whichever arm
    ran last, which is exactly the bias the study is trying to avoid.
    """
    results: list[CellResult] = []
    for question in questions:
        for rep in range(1, reps + 1):
            for arm_name in arms:
                started = time.time()
                results.append(
                    run_cell(
                        config, registry[arm_name], question, rep,
                        dry_run=dry_run, keep_stack=keep_stack,
                    )
                )
                logger.info("cell took %.0fs", time.time() - started)
    return results
