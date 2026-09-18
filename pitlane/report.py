"""results.csv, requests.csv and summary.md.

One row per cell in `results.csv`; the markdown pivots it to one table per
question with a row per arm, which is the shape the write-up needs.
`requests.csv` is the level below: one row per chat completion, carrying the
`job_id` / `agent_id` that let it be grouped any other way afterwards. Every
cell-level number is a sum over those rows, so an aggregate is a group-by and
never a re-derivation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pitlane.metrics import FINAL_NODE_CANDIDATES, Metrics

_COLUMNS = [
    "arm", "question_id", "rep", "cache_state",
    "ttft_s", "kv_hit_rate", "query_tokens", "token_hits",
    "external_token_hits", "workflow_output_tokens",
    "total_prefetches", "population_prefetches",
    "late_prefetches", "late_prefetch_pct", "unused_prefetches",
    "accurate_prefetches", "accurate_prefetch_pct", "displaced_prefetches",
    "useful_prefetches", "useful_prefetch_pct",
    "distinct_prefetch_agents", "distinct_useful_agents", "distinct_useful_pct",
    "prefetch_lead_mean_s", "prefetch_lead_min_s",
    "prefetch_lead_min_threshold_s", "lead_window_mean_s", "lead_markers",
    "tool_seconds", "unmatched_tool_markers",
    "sched_running_mean", "sched_running_max", "sched_waiting_mean", "sched_waiting_max",
    "sched_scheduled_total", "sched_admissions_total", "sched_preempted_total",
    "requests", "wall_clock_s", "trace_misses", "off_pin_requests",
]


def append_row(results_csv: Path, metrics: Metrics) -> None:
    row = {key: metrics.to_dict().get(key) for key in _COLUMNS}
    new = not results_csv.exists()
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def drop_cell_rows(path: Path, arm: str, question_id: str, rep: int) -> int:
    """Remove one cell's rows from a run-level CSV, returning how many went.

    A resumed run re-runs the cells that did not finish, and every one of these
    files is appended to per cell. Without this a retried cell leaves its first
    attempt's row in place beside the second's, and every reader -- the
    summary, the per-question regroup, the timeline -- silently averages a cell
    with itself, weighting the arm that happened to fail.
    """
    if not path.exists():
        return 0
    rows = load_rows(path)
    if not rows:
        return 0
    keep = [
        row for row in rows
        if not (row.get("arm") == arm
                and row.get("question_id") == question_id
                and str(row.get("rep")) == str(rep))
    ]
    dropped = len(rows) - len(keep)
    if not dropped:
        return 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(keep)
    return dropped


def load_rows(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.exists():
        return []
    with results_csv.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: str | None, spec: str = "", suffix: str = "") -> str:
    if value in (None, "", "None"):
        return "—"
    try:
        return f"{float(value):{spec}}{suffix}" if spec else f"{value}{suffix}"
    except ValueError:
        return str(value)


def summary(results_csv: Path) -> str:
    rows = load_rows(results_csv)
    if not rows:
        return "# pitlane results\n\nNo cells completed.\n"

    lines = ["# pitlane results", ""]
    for question in sorted({r["question_id"] for r in rows}):
        subset = [r for r in rows if r["question_id"] == question]
        output_tokens = {r["workflow_output_tokens"] for r in subset}
        header = f"## {question}"
        if len(output_tokens) == 1:
            header += f" — {next(iter(output_tokens))} output tokens"
        lines += [header, ""]
        lines += [
            "| Arm | rep | cache | Chats | TTFT (s) | KV hit rate | Query tokens | "
            "Token hits | Prefetches | Accurate % | Useful | Useful % | Late % | "
            "Lead (mean s) | Oracle window (s) | Waiting (max) | Notes |",
            "|---|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in sorted(subset, key=lambda r: (r["arm"], int(r["rep"] or 1))):
            notes = []
            if row.get("trace_misses") not in ("", "0", None):
                notes.append("trace miss")
            if row.get("off_pin_requests") not in ("", "0", None):
                notes.append("off-pin")
            lines.append(
                f"| {row['arm']} | {row['rep']} | {row['cache_state']} | "
                # Chat completions served. Equal across arms is the pin holding:
                # pinned replay sends the same calls, so a difference means the
                # trajectory diverged and nothing beside it is comparable.
                f"{_fmt(row['requests'])} | "
                f"{_fmt(row['ttft_s'], '.2f')} | "
                f"{_fmt(row['kv_hit_rate'], '.2%')} | "
                f"{_fmt(row['query_tokens'])} | {_fmt(row['token_hits'])} | "
                f"{_fmt(row['total_prefetches'])} | "
                # Was the guess right, before any question of whether it paid:
                # the column an ablation of the predictor has to move.
                f"{_fmt(row.get('accurate_prefetch_pct'), '.0%')} | "
                f"{_fmt(row['useful_prefetches'])} | "
                f"{_fmt(row['useful_prefetch_pct'], '.0%')} | "
                f"{_fmt(row['late_prefetch_pct'], '.0%')} | "
                f"{_fmt(row['prefetch_lead_mean_s'], '.3f')} | "
                f"{_fmt(row['lead_window_mean_s'], '.3f')} | "
                f"{_fmt(row['sched_waiting_max'])} | {', '.join(notes) or '—'} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_summary(run_dir: Path) -> Path:
    path = run_dir / "summary.md"
    path.write_text(summary(run_dir / "results.csv"))
    return path


_REQUEST_COLUMNS = [
    "arm", "question_id", "rep", "seq", "job_id", "agent_id", "langgraph_node",
    "request_id", "arrival_ts", "finish_ts", "ttft_s", "e2e_s",
    "queued_s", "prefill_s", "decode_s",
    "query_tokens", "token_hits", "external_token_hits", "kv_hit_rate",
    "output_tokens", "prefetches", "late_prefetches", "useful",
    "credited_tokens", "prefetch_lead_s",
    "max_lead_ts", "min_lead_ts", "lead_window_s",
]


_PREFETCH_COLUMNS = [
    "arm", "question_id", "rep", "job_id", "agent_id", "langgraph_node",
    "request_id", "arrival_ts", "finish_ts", "elapsed_ms",
    "queued_s", "prefill_s", "decode_s", "prompt_tokens",
    "consumer_request_id", "displaced_by", "accurate",
]


def _append(path: Path, columns: list[str], scope: dict[str, Any],
            rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if new:
            writer.writeheader()
        for entry in rows:
            merged = {**scope, **entry}
            writer.writerow({key: merged.get(key) for key in columns})


def append_request_rows(requests_csv: Path, metrics: Metrics) -> None:
    """One row per chat completion, appended across every cell of the run."""
    _append(requests_csv, _REQUEST_COLUMNS, _scope(metrics), metrics.per_request)


def append_prefetch_rows(prefetches_csv: Path, metrics: Metrics) -> None:
    """One row per phantom, the same shape of fact as `requests.csv`.

    `decode_s` is always blank here and that is the honest value: a phantom runs
    no sampling step, so there is no decode phase to have taken zero seconds.
    """
    _append(prefetches_csv, _PREFETCH_COLUMNS, _scope(metrics), metrics.per_prefetch)


_TOOL_COLUMNS = [
    "arm", "question_id", "rep", "job_id", "agent_id", "tool", "call_id",
    "start_ts", "end_ts", "elapsed_s",
]


def append_tool_rows(tools_csv: Path, metrics: Metrics) -> None:
    """One row per leaf tool call, paired from the `/v1/echo` markers.

    `researcher_tools` only: the supervisor's `ConductResearch` spawns subgraphs
    whose work is already on the timeline as chat completions, and never reaches
    the instrumented helper.
    """
    _append(tools_csv, _TOOL_COLUMNS, _scope(metrics), metrics.per_tool)


def _scope(metrics: Metrics) -> dict[str, Any]:
    return {"arm": metrics.arm, "question_id": metrics.question_id, "rep": metrics.rep}


def write_metrics(cell: Path, metrics: Metrics) -> Path:
    path = cell / "metrics.json"
    path.write_text(json.dumps(metrics.to_dict(), indent=2) + "\n")
    return path


# -- per question -----------------------------------------------------------
#
# `results.csv` is one row per *cell*, and a batch cell is N questions in one
# process -- so its numbers are sums over questions that were never meant to be
# added together. The question is the unit everything is actually compared at,
# and the per-request rows already carry `job_id`, so this is a regroup of data
# that exists rather than a second collection pass.

_BY_QUESTION_COLUMNS = [
    "arm", "question_id", "rep", "job_id",
    "requests", "query_tokens", "token_hits", "external_token_hits",
    "kv_hit_rate", "output_tokens",
    "ttft_s", "final_ttft_s", "ttft_source_node", "wall_s",
    "queued_s", "prefill_s", "decode_s",
    "prefetches", "accurate_prefetches", "accurate_prefetch_pct",
    "displaced_prefetches", "unused_prefetches",
    "useful_prefetches", "useful_prefetch_pct",
    "distinct_prefetch_agents", "distinct_useful_agents", "distinct_useful_pct",
    "late_prefetches",
]


def _num(value: str | None) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _opt(value: str | None) -> float | None:
    """A CSV cell as a number, or None where it was blank rather than zero."""
    if value in (None, "", "None"):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _question_ttft(rows: list[dict[str, str]]) -> tuple[float | None, float | None, str | None]:
    """Question submit -> first token of the final output, and that call's own TTFT.

    The same quantity `results.csv` reports, measured from a different origin
    and for one question rather than one cell. The cell's TTFT counts from the
    runner's dispatch stamp, and a batch cell has exactly one of those for N
    questions -- so here the origin is the question's own first request
    arrival, which is the earliest instant that belongs to it alone. The two
    agree on an isolated cell up to the workflow's startup and seeding phase,
    which the dispatch stamp includes and this does not.

    `final_ttft_s` is the terminal call's own queue-plus-prefill, which is the
    part the final-report prompt seed exists to move; the difference between
    the two columns is everything the graph did before that call was dispatched.
    """
    dated = [r for r in rows if _opt(r.get("arrival_ts")) is not None]
    if not dated:
        return None, None, None
    ordered = sorted(dated, key=lambda r: _num(r.get("arrival_ts")))
    final: dict[str, str] | None = None
    node: str | None = None
    for candidate in FINAL_NODE_CANDIDATES:
        matches = [r for r in ordered if r.get("langgraph_node") == candidate]
        if matches:
            final, node = matches[-1], candidate
            break
    if final is None:
        # No named terminal node -- the last request of the question produced
        # the final output by definition. Same fallback as the cell-level read.
        final = ordered[-1]
        node = final.get("langgraph_node") or "(last request)"
    final_ttft = _opt(final.get("ttft_s"))
    if final_ttft is None:
        return None, None, node
    first_token = _num(final.get("arrival_ts")) + final_ttft
    return (round(first_token - _num(ordered[0].get("arrival_ts")), 3),
            round(final_ttft, 3), node)


def by_question(run_dir: Path) -> list[dict[str, Any]]:
    """One row per `(arm, question_id, rep, job_id)`, from the request rows."""
    requests = _read_csv(run_dir / "requests.csv")
    prefetches = _read_csv(run_dir / "prefetches.csv")
    if not requests:
        return []

    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in requests:
        key = (row.get("arm", ""), row.get("question_id", ""),
               row.get("rep", ""), row.get("job_id", ""))
        groups.setdefault(key, []).append(row)

    warmed: dict[tuple[str, str, str, str], set[str]] = {}
    phantoms: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in prefetches:
        key = (row.get("arm", ""), row.get("question_id", ""),
               row.get("rep", ""), row.get("job_id", ""))
        phantoms.setdefault(key, []).append(row)
        if row.get("agent_id"):
            warmed.setdefault(key, set()).add(row["agent_id"])

    out: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        arm, question, rep, job = key
        query = sum(_num(r.get("query_tokens")) for r in rows)
        hits = sum(_num(r.get("token_hits")) for r in rows)
        starts = [_num(r.get("arrival_ts")) for r in rows if r.get("arrival_ts")]
        ends = [_num(r.get("finish_ts")) for r in rows if r.get("finish_ts")]
        # Distinct *agents* helped, so a React node warmed on every turn counts
        # once. The denominator is the agents warmed at all, from the phantom
        # rows -- an agent can be warmed and never helped, and that is the case
        # the ratio exists to expose.
        helped = {r["agent_id"] for r in rows
                  if _truthy(r.get("useful")) and r.get("agent_id")}
        warm_set = warmed.get(key, set())
        useful = sum(1 for r in rows if _truthy(r.get("useful")))
        # Accuracy is a property of the phantom, so it is counted on the
        # phantom rows -- unlike `useful`, which is charged to the consumer so
        # that a `top_k` fan-out cannot bank one hit k times. The three
        # outcomes partition the phantoms: accurate + displaced + unused.
        warms = phantoms.get(key, [])
        prefetch_n = len(warms)
        # A run recorded before these columns existed has phantom rows without
        # them, and a blank `displaced_by` there means "never consumed" -- so
        # scoring such a file would read every warm as unused. That is reported
        # as unknown instead. An arm that simply never prefetches has nothing to
        # score and 0 is the honest count, which is why an empty list passes.
        scored = not warms or "accurate" in warms[0]
        accurate = sum(1 for r in warms if _truthy(r.get("accurate"))) if scored else None
        displaced = (
            sum(1 for r in warms if (_opt(r.get("displaced_by")) or 0) > 0)
            if scored else None
        )
        unused = (
            sum(1 for r in warms if _opt(r.get("displaced_by")) is None)
            if scored else None
        )
        ttft_s, final_ttft_s, ttft_node = _question_ttft(rows)
        out.append({
            "arm": arm, "question_id": question, "rep": rep, "job_id": job,
            "requests": len(rows),
            "query_tokens": int(query),
            "token_hits": int(hits),
            "external_token_hits": int(
                sum(_num(r.get("external_token_hits")) for r in rows)
            ),
            "kv_hit_rate": round(hits / query, 6) if query else None,
            "output_tokens": int(sum(_num(r.get("output_tokens")) for r in rows)),
            # Submit -> first token of the final output, from this question's
            # own first request: a batch cell has one dispatch stamp for N
            # questions, so the cell's origin cannot be reused here.
            "ttft_s": ttft_s,
            "final_ttft_s": final_ttft_s,
            "ttft_source_node": ttft_node,
            "wall_s": round(max(ends) - min(starts), 3) if starts and ends else None,
            "queued_s": round(sum(_num(r.get("queued_s")) for r in rows), 3),
            "prefill_s": round(sum(_num(r.get("prefill_s")) for r in rows), 3),
            "decode_s": round(sum(_num(r.get("decode_s")) for r in rows), 3),
            "prefetches": prefetch_n,
            "accurate_prefetches": accurate,
            "accurate_prefetch_pct": (
                round(accurate / prefetch_n, 6)
                if accurate is not None and prefetch_n else None
            ),
            "displaced_prefetches": displaced,
            "unused_prefetches": unused,
            "useful_prefetches": useful,
            "useful_prefetch_pct": round(useful / prefetch_n, 6) if prefetch_n else None,
            "distinct_prefetch_agents": len(warm_set),
            "distinct_useful_agents": len(helped),
            "distinct_useful_pct": (
                round(len(helped) / len(warm_set), 6) if warm_set else None
            ),
            "late_prefetches": sum(int(_num(r.get("late_prefetches"))) for r in rows),
        })
    return out


def write_by_question(run_dir: Path) -> Path | None:
    rows = by_question(run_dir)
    if not rows:
        return None
    path = run_dir / "by_question.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_BY_QUESTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _BY_QUESTION_COLUMNS})
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
