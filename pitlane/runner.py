"""The lap itself: setup, run one question, collect, tear down.

The ordering is the contract -- LMCache restarted before the server, metrics
epoch reset after the server answers and before the workflow starts, artifacts
collected before anything is killed.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from pitlane import metrics as metrics_mod
from pitlane import report, resources, stack, timeline, workflow
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
    return CellResult(arm.name, question_id, rep, cell, collected, result.exit_code)


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
) -> list[CellResult]:
    """Interleave arms within a question: A, B, C, A, B, C, ...

    Blocking by arm would load any drift in machine state onto whichever arm
    ran last, which is exactly the bias the study is trying to avoid.
    """
    results: list[CellResult] = []
    for question in questions:
        cell_count = question_count(question, count)
        for rep in range(1, reps + 1):
            for arm_name in arms:
                started = time.time()
                results.append(
                    run_cell(
                        config, registry[arm_name], question, rep,
                        count=cell_count,
                        cache_state="warm" if cell_count > 1 else "cold",
                        dry_run=dry_run, keep_stack=keep_stack,
                    )
                )
                logger.info("cell took %.0fs", time.time() - started)
    return results
