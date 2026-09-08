"""Turn one cell's raw artifacts into the numbers the study reports.

Sources, all wall-clock epoch seconds and therefore joinable:
  stats/finished_requests_engine*.jsonl  per-request stats from vLLM
  stats/scheduler_engine*.jsonl          queue depth over time
  divergence.jsonl                       pinned-replay fidelity
  agent_prefetch.jsonl                   client-side prefetch issue log
  question_started_ts                    the runner's dispatch stamp
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# The graph node whose response is the final output. ODR names it; a workflow
# that does not is handled by falling back to the last request of the job.
FINAL_NODE_CANDIDATES = ("final_report_generation", "final_report", "generate_patch")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn last line while the server is still writing
    return rows


def _read_glob(directory: Path, pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob(pattern)):
        rows.extend(_read_jsonl(path))
    return rows


def first_token_ts(row: dict[str, Any]) -> float | None:
    """When this request produced its first token, on the wall clock.

    vLLM records the arrival on the wall clock and everything after it as
    monotonic durations, so the first-token instant has to be reconstructed:
    queue wait plus prefill is exactly the time before the first token.
    """
    arrival = row.get("arrival_ts")
    if not arrival:
        return None
    return arrival + (row.get("queued_time") or 0.0) + (row.get("prefill_time") or 0.0)


@dataclass
class Metrics:
    arm: str = ""
    question_id: str = ""
    rep: int = 1
    cache_state: str = "cold"

    ttft_s: float | None = None
    ttft_source_node: str | None = None

    kv_hit_rate: float | None = None
    external_hit_rate: float | None = None
    query_tokens: int = 0
    token_hits: int = 0
    external_token_hits: int = 0
    workflow_output_tokens: int = 0

    total_prefetches: int = 0
    late_prefetches: int = 0
    unused_prefetches: int = 0
    late_prefetch_pct: float | None = None
    prefetch_wait_mode: str | None = None

    sched_running_mean: float | None = None
    sched_running_max: int | None = None
    sched_waiting_mean: float | None = None
    sched_waiting_max: int | None = None
    sched_scheduled_total: int = 0
    sched_admissions_total: int = 0
    sched_preempted_total: int = 0

    requests: int = 0
    wall_clock_s: float | None = None
    trace_misses: int = 0
    off_pin_requests: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _window(rows: Iterable[dict[str, Any]], t0: float | None, t1: float | None,
            key: str = "arrival_ts") -> list[dict[str, Any]]:
    out = []
    for row in rows:
        ts = row.get(key)
        if ts is None:
            continue
        if t0 is not None and ts < t0:
            continue
        if t1 is not None and ts > t1:
            continue
        out.append(row)
    return out


def collect(
    cell: Path,
    *,
    arm: str = "",
    question_id: str = "",
    rep: int = 1,
    t0: float | None = None,
    t1: float | None = None,
    cache_state: str = "cold",
) -> Metrics:
    metrics = Metrics(arm=arm, question_id=question_id, rep=rep, cache_state=cache_state)

    if t0 is None:
        stamp = cell / "question_started_ts"
        if stamp.exists():
            try:
                t0 = float(stamp.read_text().strip())
            except ValueError:
                pass
    if t0 is not None and t1 is not None:
        metrics.wall_clock_s = t1 - t0

    stats_dir = cell / "stats"
    rows = _window(_read_glob(stats_dir, "finished_requests_engine*.jsonl"), t0, t1)
    real = [r for r in rows if not r.get("prefetch_only")]
    phantom = [r for r in rows if r.get("prefetch_only")]
    metrics.requests = len(real)

    if not rows:
        metrics.warnings.append("no request stats in window; check VLLM_REQUEST_STATS_DIR")

    _token_metrics(metrics, real)
    _ttft(metrics, real, t0)
    _prefetch(metrics, cell, phantom, real)
    _scheduler(metrics, stats_dir, t0, t1)
    _divergence(metrics, cell / "divergence.jsonl")
    return metrics


def _token_metrics(metrics: Metrics, real: list[dict[str, Any]]) -> None:
    metrics.query_tokens = sum(r.get("num_prompt_tokens") or 0 for r in real)
    metrics.token_hits = sum(r.get("num_local_cached_tokens") or 0 for r in real)
    metrics.external_token_hits = sum(r.get("num_external_cached_tokens") or 0 for r in real)
    metrics.workflow_output_tokens = sum(r.get("num_generation_tokens") or 0 for r in real)
    if metrics.query_tokens:
        metrics.kv_hit_rate = metrics.token_hits / metrics.query_tokens
        metrics.external_hit_rate = metrics.external_token_hits / metrics.query_tokens


def _ttft(metrics: Metrics, real: list[dict[str, Any]], t0: float | None) -> None:
    """Submit -> first token of the final output."""
    if not real or t0 is None:
        if t0 is None:
            metrics.warnings.append("no question_started_ts; TTFT not computable")
        return
    ordered = sorted(real, key=lambda r: r.get("arrival_ts") or 0.0)
    final = None
    for candidate in FINAL_NODE_CANDIDATES:
        matches = [r for r in ordered if r.get("langgraph_node") == candidate]
        if matches:
            final, metrics.ttft_source_node = matches[-1], candidate
            break
    if final is None:
        # No named terminal node: the last request of the question produced the
        # final output by definition.
        final = ordered[-1]
        metrics.ttft_source_node = final.get("langgraph_node") or "(last request)"
        metrics.warnings.append("final node not recognised; used the last request")
    first = first_token_ts(final)
    if first is not None:
        metrics.ttft_s = first - t0


def _prefetch(metrics: Metrics, cell: Path, phantom: list[dict[str, Any]],
              real: list[dict[str, Any]]) -> None:
    agent_log = _read_jsonl(cell / "agent_prefetch.jsonl")
    metrics.total_prefetches = len(phantom)
    if agent_log:
        modes = {str(r.get("wait")) for r in agent_log if "wait" in r}
        if modes:
            metrics.prefetch_wait_mode = ",".join(sorted(modes))
    if not phantom:
        return

    by_agent: dict[str | None, list[dict[str, Any]]] = {}
    for row in sorted(real, key=lambda r: r.get("arrival_ts") or 0.0):
        by_agent.setdefault(row.get("agent_id"), []).append(row)

    for ghost in phantom:
        issued = ghost.get("arrival_ts") or 0.0
        finished = ghost.get("finish_ts") or 0.0
        consumer = next(
            (r for r in by_agent.get(ghost.get("agent_id"), [])
             if (r.get("arrival_ts") or 0.0) >= issued),
            None,
        )
        if consumer is None:
            metrics.unused_prefetches += 1
        elif (consumer.get("arrival_ts") or 0.0) < finished:
            # The workload arrived while the phantom was still pulling: the
            # promotion did not finish in time to be a hit.
            metrics.late_prefetches += 1

    if metrics.prefetch_wait_mode == "True":
        # A blocking prefetch cannot be late: the client does not send the real
        # request until it returns. Reporting a 0% here would look like success.
        metrics.warnings.append("prefetches issued with wait=true; late % not meaningful")
        return
    metrics.late_prefetch_pct = metrics.late_prefetches / metrics.total_prefetches


def _scheduler(metrics: Metrics, stats_dir: Path, t0: float | None, t1: float | None) -> None:
    samples = [
        row for row in _window(
            _read_glob(stats_dir, "scheduler_engine*.jsonl"), t0, t1, key="ts"
        )
        if row.get("event") != "clock_anchor"
    ]
    if not samples:
        return
    running = [r.get("num_running_reqs") or 0 for r in samples]
    waiting = [r.get("num_waiting_reqs") or 0 for r in samples]
    metrics.sched_running_mean = statistics.fmean(running)
    metrics.sched_running_max = max(running)
    metrics.sched_waiting_mean = statistics.fmean(waiting)
    metrics.sched_waiting_max = max(waiting)
    metrics.sched_scheduled_total = sum(r.get("num_scheduled_reqs") or 0 for r in samples)
    metrics.sched_admissions_total = sum(r.get("num_new_scheduled_reqs") or 0 for r in samples)
    metrics.sched_preempted_total = sum(r.get("num_preempted_reqs") or 0 for r in samples)


def _divergence(metrics: Metrics, path: Path) -> None:
    rows = _read_jsonl(path)
    if not rows:
        return
    metrics.trace_misses = sum(1 for r in rows if r.get("reason") == "miss")
    pinned = [r for r in rows if r.get("reason") == "pinned"]
    metrics.off_pin_requests = sum(
        1 for r in pinned
        if r.get("pinned_exact_tokens")
        and r.get("live_completion_tokens") != r.get("pinned_max_tokens")
    )
    if metrics.trace_misses:
        metrics.warnings.append(
            f"{metrics.trace_misses} trace miss(es): trajectory diverged, cell not comparable"
        )
    if metrics.off_pin_requests:
        metrics.warnings.append(
            f"{metrics.off_pin_requests} request(s) did not decode the pinned token count"
        )
