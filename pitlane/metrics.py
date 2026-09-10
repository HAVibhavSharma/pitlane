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
import re
import statistics
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# The graph node whose response is the final output. ODR names it; a workflow
# that does not is handled by falling back to the last request of the job.
FINAL_NODE_CANDIDATES = ("final_report_generation", "final_report", "generate_patch")

# APC matches on whole blocks, so a phantom is only credited with residency it
# added in block-sized units. A credit of 1-15 tokens is the partial trailing
# block every prompt has and is noise, not a hit.
BLOCK_SIZE = 16

# A phantom is late when it left too close in front of the request it warms to
# have done anything with the time. Both stamps come from the request rows --
# `arrival_ts` on the phantom and on its consumer -- so this is the engine's own
# view on both sides, not a log join.
#
# There is no physical constant to pin the threshold to, so it is a knob and the
# lead distribution is reported next to it: read `late_prefetch_pct` together
# with `prefetch_lead_mean_s` / `prefetch_lead_min_s`, never alone. This is the
# default only; a run takes it from `BENCH_PREFETCH_LEAD_MIN_S` via `Config`,
# which is the one place user knobs are read.
LEAD_MIN_S = 0.1

# `POST /v1/echo` markers, as the server logs them. Two clocks would have to be
# reconciled to measure this any other way -- the client's decision instants
# live in its own process -- so the client posts them and the server stamps them
# on its own timeline, which is the same clock `arrival_ts` is on.
#
#   echo: event=max_lead agent_id=langgraph:1:...:supervisor issuer=workflow ...
#
# The tail is parsed as `key=value` pairs rather than matched field by field,
# because the markers carry different fields per event -- `call_id` and `tool`
# on a tool span, `route` on a warm -- and a regex per shape would need editing
# every time a call site adds one.
_ECHO_LINE = re.compile(
    r"(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+).*?echo: (?P<body>.*)$"
)
_ECHO_FIELD = re.compile(r"(\w+)=(\S+)")
_LEAD_EVENTS = ("min_lead", "max_lead")
_TOOL_EVENTS = ("tool_start", "tool_end")


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


def _echo_fields(line: str) -> tuple[float, dict[str, str]] | None:
    """One echo line as `(server timestamp, fields)`, or None if it is not one."""
    match = _ECHO_LINE.search(line)
    if not match:
        return None
    try:
        stamp = datetime.strptime(match["ts"], "%Y-%m-%d %H:%M:%S.%f").timestamp()
    except ValueError:
        return None
    return stamp, dict(_ECHO_FIELD.findall(match["body"]))


def _interval(value: Any) -> float | None:
    """A duration field from the request row, or None when it is not one.

    vLLM computes `prefill_time` as `first_token_ts - scheduled_ts` with no
    guard (`v1/metrics/stats.py:501`), and a request that produced no token
    leaves `first_token_ts` at 0.0 -- so the field comes back as a large
    negative monotonic value rather than as a missing one. Every phantom is in
    that state by construction: `_finalize_prefetch_only_request` terminates it
    with `new_token_ids=[]`. Dropping the impossible values keeps a column that
    means one thing.
    """
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0.0 else None


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
class RequestMetrics:
    """One chat completion, keyed by the ids that survive aggregation.

    Everything the cell reports is a sum over these rows, so the cell numbers
    stay derivable and a regression can be traced to the node that caused it
    rather than to the question. `seq` is the request's position on the wire,
    which is what joins the same prompt across arms under pinned replay --
    `agent_id` alone repeats whenever a node takes more than one turn.
    """

    job_id: str | None = None
    agent_id: str | None = None
    langgraph_node: str | None = None
    request_id: str | None = None
    seq: int = 0

    arrival_ts: float | None = None
    finish_ts: float | None = None
    ttft_s: float | None = None
    e2e_s: float | None = None
    queued_s: float | None = None
    prefill_s: float | None = None
    decode_s: float | None = None

    query_tokens: int = 0
    token_hits: int = 0
    external_token_hits: int = 0
    output_tokens: int = 0
    kv_hit_rate: float | None = None

    # Phantoms that named this request as their consumer.
    prefetches: int = 0
    late_prefetches: int = 0
    useful: bool = False
    credited_tokens: int = 0
    prefetch_lead_s: float | None = None

    # The `/v1/echo` markers, as the server stamped them. `max_lead_ts` is when
    # the workflow oracle issued the warm; `min_lead_ts` is when the graph
    # runtime parsed the tool call naming this node, which is the earliest a
    # real predictor could know. `lead_window_s` is the span between them: what
    # reading the recording buys over predicting, measured entirely from the two
    # markers and referred to nothing else.
    min_lead_ts: float | None = None
    max_lead_ts: float | None = None
    lead_window_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    useful_prefetches: int = 0
    late_prefetch_pct: float | None = None
    useful_prefetch_pct: float | None = None
    prefetch_lead_mean_s: float | None = None
    prefetch_lead_min_s: float | None = None
    prefetch_lead_min_threshold_s: float = LEAD_MIN_S
    lead_window_mean_s: float | None = None
    lead_markers: int = 0
    prefetch_wait_mode: str | None = None

    sched_running_mean: float | None = None
    sched_running_max: int | None = None
    sched_waiting_mean: float | None = None
    sched_waiting_max: int | None = None
    sched_scheduled_total: int = 0
    sched_admissions_total: int = 0
    sched_preempted_total: int = 0

    requests: int = 0
    per_request: list[dict[str, Any]] = field(default_factory=list)
    # Phantom spans, for the timeline. Same shape of fact as `per_request`:
    # engine `arrival_ts` / `finish_ts`, which is where `agent_prefetch_start` /
    # `agent_prefetch_end` in the server log get their instants from anyway.
    per_prefetch: list[dict[str, Any]] = field(default_factory=list)
    # Leaf tool spans from the `/v1/echo` markers: the CPU and I/O between model
    # calls, which no request row records because a tool call is not a request.
    per_tool: list[dict[str, Any]] = field(default_factory=list)
    tool_seconds: float = 0.0
    unmatched_tool_markers: int = 0
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
    lead_min_s: float = LEAD_MIN_S,
) -> Metrics:
    metrics = Metrics(arm=arm, question_id=question_id, rep=rep, cache_state=cache_state,
                      prefetch_lead_min_threshold_s=lead_min_s)

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
    _requests(metrics, real)
    _prefetch(metrics, cell, phantom, real, lead_min_s)
    _lead_markers(metrics, cell, real)
    _tool_spans(metrics, cell)
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


def _requests(metrics: Metrics, real: list[dict[str, Any]]) -> None:
    """One row per chat completion, in wire order. Prefetch fields filled later."""
    for seq, row in enumerate(sorted(real, key=lambda r: r.get("arrival_ts") or 0.0), 1):
        entry = RequestMetrics(
            job_id=row.get("job_id"),
            agent_id=row.get("agent_id"),
            langgraph_node=row.get("langgraph_node"),
            request_id=row.get("request_id"),
            seq=seq,
            arrival_ts=row.get("arrival_ts"),
            finish_ts=row.get("finish_ts"),
            query_tokens=row.get("num_prompt_tokens") or 0,
            token_hits=row.get("num_local_cached_tokens") or 0,
            external_token_hits=row.get("num_external_cached_tokens") or 0,
            output_tokens=row.get("num_generation_tokens") or 0,
            queued_s=_interval(row.get("queued_time")),
            prefill_s=_interval(row.get("prefill_time")),
            decode_s=_interval(row.get("decode_time")),
        )
        first = first_token_ts(row)
        if first is not None and entry.arrival_ts is not None:
            entry.ttft_s = first - entry.arrival_ts
        if entry.arrival_ts is not None and entry.finish_ts is not None:
            entry.e2e_s = entry.finish_ts - entry.arrival_ts
        if entry.query_tokens:
            entry.kv_hit_rate = entry.token_hits / entry.query_tokens
        metrics.per_request.append(entry.to_dict())


def _credited_tokens(ghost: dict[str, Any], consumer: dict[str, Any]) -> int:
    """HBM residency this phantom added that its consumer went on to hit.

    A phantom's own `num_local_cached_tokens` is how much of the prefix was
    already in HBM at the moment it ran, so everything past that mark is what it
    brought in -- promoted from LMCache or prefilled outright. The consumer's
    local hit is capped at the phantom's prompt because a consumer can hit
    further than the phantom ever covered, and that tail belongs to something
    else.

    Both sides are floored to a block: a phantom that pulls a prefix HBM already
    holds returns <= 0 here, which is the case this metric exists to reject.
    """
    covered = min(
        consumer.get("num_local_cached_tokens") or 0,
        ghost.get("num_prompt_tokens") or 0,
    )
    already = ghost.get("num_local_cached_tokens") or 0
    credited = (covered // BLOCK_SIZE - already // BLOCK_SIZE) * BLOCK_SIZE
    return max(credited, 0)


def _prefetch(metrics: Metrics, cell: Path, phantom: list[dict[str, Any]],
              real: list[dict[str, Any]], lead_min_s: float = LEAD_MIN_S) -> None:
    """Attribute every phantom to the chat completion it warmed.

    The unit is the consumer, not the cell: each phantom is charged to one
    request row, and the cell's counters are sums over those rows plus the
    phantoms that never found a consumer at all. Aggregating later is then a
    group-by on `agent_id` / `job_id` rather than a re-derivation.
    """
    agent_log = _read_jsonl(cell / "agent_prefetch.jsonl")
    metrics.total_prefetches = len(phantom)
    if agent_log:
        modes = {str(r.get("wait")) for r in agent_log if "wait" in r}
        if modes:
            metrics.prefetch_wait_mode = ",".join(sorted(modes))
    for ghost in sorted(phantom, key=lambda r: r.get("arrival_ts") or 0.0):
        started = ghost.get("arrival_ts")
        finished = ghost.get("finish_ts")
        queued = _interval(ghost.get("queued_time"))
        metrics.per_prefetch.append({
            "job_id": ghost.get("job_id"),
            "agent_id": ghost.get("agent_id"),
            "langgraph_node": ghost.get("langgraph_node"),
            "request_id": ghost.get("request_id"),
            "arrival_ts": started,
            "finish_ts": finished,
            "elapsed_ms": (
                (finished - started) * 1000.0
                if started is not None and finished is not None else None
            ),
            "prompt_tokens": ghost.get("num_prompt_tokens") or 0,
            "queued_s": queued,
            # The engine's own `prefill_time` is unusable on a phantom (see
            # `_interval`), so it is derived from the span instead: a phantom has
            # no decode, so everything that is not queue wait is the LMCache load
            # or the prefill `prefill_on_miss` let through.
            "prefill_s": (
                max(finished - started - (queued or 0.0), 0.0)
                if started is not None and finished is not None else None
            ),
            # Not "zero decode" but "no decode phase": `max_tokens=1` and the
            # prefetch-only finalize path means no sampling step ever runs.
            "decode_s": None,
        })
    if not phantom:
        return

    ordered = sorted(real, key=lambda r: r.get("arrival_ts") or 0.0)
    by_agent: dict[str | None, list[dict[str, Any]]] = {}
    for row in ordered:
        by_agent.setdefault(row.get("agent_id"), []).append(row)
    # `_requests` walked the same order, so position is the join.
    entry_of = {id(row): metrics.per_request[i] for i, row in enumerate(ordered)}

    leads: list[float] = []
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
            continue

        entry = entry_of[id(consumer)]
        entry["prefetches"] += 1

        # Lead: how far in front of the request it warms this phantom left.
        # Both stamps are the engine's `arrival_ts`, so the HTTP and
        # chat-template time on either side cancels and what is left is the gap
        # the prefetcher actually bought.
        consumer_start = consumer.get("arrival_ts") or 0.0
        lead = consumer_start - issued
        leads.append(lead)
        if lead < lead_min_s:
            entry["late_prefetches"] += 1
        if entry["prefetch_lead_s"] is None or lead > entry["prefetch_lead_s"]:
            entry["prefetch_lead_s"] = lead

        if consumer_start < finished:
            # It had not finished when the request arrived, so the blocks were
            # not there to hit and it cannot be useful. Not the same test as
            # `late`, which is a judgement about whether the lead was worth
            # having: a millisecond-long LMCache promotion can be late and still
            # land, and a seeded prefill can be on time and still miss this.
            continue

        # A React warm asks for `top_k` prefixes and they all name the same
        # consumer, so the request keeps the best credit rather than banking one
        # hit `top_k` times.
        credited = _credited_tokens(ghost, consumer)
        if credited > entry["credited_tokens"]:
            entry["credited_tokens"] = credited
            entry["useful"] = True

    metrics.late_prefetches = sum(e["late_prefetches"] for e in metrics.per_request)
    metrics.useful_prefetches = sum(1 for e in metrics.per_request if e["useful"])
    metrics.useful_prefetch_pct = metrics.useful_prefetches / metrics.total_prefetches

    if leads:
        metrics.prefetch_lead_mean_s = statistics.fmean(leads)
        metrics.prefetch_lead_min_s = min(leads)

    if metrics.prefetch_wait_mode == "True":
        # A blocking prefetch has no lead to measure: the client does not send
        # the real request until it returns, so every lead is ~0 and the late %
        # would read 100% by construction.
        metrics.warnings.append("prefetches issued with wait=true; late % not meaningful")
        return
    metrics.late_prefetch_pct = metrics.late_prefetches / metrics.total_prefetches


def _lead_markers(metrics: Metrics, cell: Path, real: list[dict[str, Any]]) -> None:
    """Record each request's `/v1/echo` lead markers, and the span between them.

    The only metric here read from the server log, and it has to be: the two
    instants are client-side decisions with no request of their own, so the
    marker the client posted and the server stamped is the only record. Routing
    them through `/v1/echo` puts both on the server's clock, which is what makes
    the span between them meaningful at all.

    The reported value is the markers' own: `lead_window_s = min_lead_ts -
    max_lead_ts`, the interval the oracle knew about a target before a real
    predictor could have. Request timestamps are used only to decide *which*
    request a marker belongs to, never to compute the value.

    A marker is paired with the next request of the same `agent_id`, latest
    marker first -- markers and requests are both in time order, so a node that
    takes several turns gets the marker belonging to its turn.
    """
    log = cell / "server.log"
    if not log.exists():
        return

    ordered = sorted(real, key=lambda r: r.get("arrival_ts") or 0.0)
    entry_of = {id(row): metrics.per_request[i] for i, row in enumerate(ordered)}
    by_agent: dict[str | None, list[dict[str, Any]]] = {}
    for row in ordered:
        by_agent.setdefault(row.get("agent_id"), []).append(row)

    for line in log.read_text(errors="replace").splitlines():
        parsed = _echo_fields(line)
        if parsed is None:
            continue
        stamp, fields = parsed
        if fields.get("event") not in _LEAD_EVENTS:
            continue
        metrics.lead_markers += 1
        consumer = next(
            (r for r in by_agent.get(fields.get("agent_id"), [])
             if (r.get("arrival_ts") or 0.0) >= stamp),
            None,
        )
        if consumer is None:
            continue
        entry = entry_of[id(consumer)]
        # The latest marker of each kind before the request wins: for a repeated
        # node that is the one belonging to this turn rather than an earlier one.
        field_name = f"{fields['event']}_ts"
        if entry[field_name] is None or stamp > entry[field_name]:
            entry[field_name] = stamp

    windows: list[float] = []
    paired = 0
    for entry in metrics.per_request:
        if entry["min_lead_ts"] is not None or entry["max_lead_ts"] is not None:
            paired += 1
        if entry["min_lead_ts"] is not None and entry["max_lead_ts"] is not None:
            entry["lead_window_s"] = entry["min_lead_ts"] - entry["max_lead_ts"]
            windows.append(entry["lead_window_s"])
    if windows:
        metrics.lead_window_mean_s = statistics.fmean(windows)

    if metrics.total_prefetches and not metrics.lead_markers:
        metrics.warnings.append(
            "prefetches issued but no /v1/echo lead markers in server.log"
        )
    elif metrics.lead_markers and not paired:
        # Markers were logged but none named an agent this cell served. That is
        # the join breaking, not an absence of data, and it would otherwise show
        # up only as empty columns.
        metrics.warnings.append(
            f"{metrics.lead_markers} /v1/echo lead marker(s) matched no request; "
            "agent_id mismatch or markers outside the question window"
        )


def _tool_spans(metrics: Metrics, cell: Path) -> None:
    """Pair `tool_start` / `tool_end` markers into one span per tool call.

    Paired on `call_id`, which is the tool call's own id -- the same discipline
    the phantom spans get from `req=prefetch::...`, and for the same reason:
    three tools run concurrently under one `asyncio.gather`, so anything
    weaker than an explicit id would interleave them.

    Only `researcher_tools` reaches the instrumented helper, so these are leaf
    tools -- search, MCP, `think_tool`. The supervisor's `ConductResearch`
    cannot appear here, which is deliberate: it is tens of seconds of chat
    completions already on the timeline, and counting it as a tool span would
    double-count the researcher's own work.
    """
    log = cell / "server.log"
    if not log.exists():
        return

    open_spans: dict[str, tuple[float, dict[str, str]]] = {}
    for line in log.read_text(errors="replace").splitlines():
        parsed = _echo_fields(line)
        if parsed is None:
            continue
        stamp, fields = parsed
        event = fields.get("event")
        if event not in _TOOL_EVENTS:
            continue
        call_id = fields.get("call_id")
        if not call_id:
            metrics.unmatched_tool_markers += 1
            continue
        if event == "tool_start":
            open_spans[call_id] = (stamp, fields)
            continue
        started = open_spans.pop(call_id, None)
        if started is None:
            # An end with no start: the log was truncated, or the run began
            # mid-tool. Counted rather than dropped silently.
            metrics.unmatched_tool_markers += 1
            continue
        begin, opening = started
        metrics.per_tool.append({
            "agent_id": opening.get("agent_id") or fields.get("agent_id"),
            "tool": opening.get("tool") or fields.get("tool"),
            "call_id": call_id,
            "start_ts": begin,
            "end_ts": stamp,
            # From the two server stamps, not from the client's own
            # `elapsed_ms`: one clock for every span on the timeline.
            "elapsed_s": max(stamp - begin, 0.0),
        })

    # A start with no end is a tool that never returned -- or, far more often, a
    # window that closed mid-call. Either way the span is unknown, not zero.
    metrics.unmatched_tool_markers += len(open_spans)
    metrics.per_tool.sort(key=lambda row: row["start_ts"])
    metrics.tool_seconds = sum(row["elapsed_s"] for row in metrics.per_tool)
    if metrics.unmatched_tool_markers:
        metrics.warnings.append(
            f"{metrics.unmatched_tool_markers} unpaired tool marker(s); "
            "tool time is under-counted for this cell"
        )


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
