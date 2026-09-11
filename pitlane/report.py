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

from pitlane.metrics import Metrics

_COLUMNS = [
    "arm", "question_id", "rep", "cache_state",
    "ttft_s", "kv_hit_rate", "query_tokens", "token_hits",
    "external_token_hits", "workflow_output_tokens",
    "total_prefetches", "late_prefetches", "late_prefetch_pct", "unused_prefetches",
    "useful_prefetches", "useful_prefetch_pct",
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
            "| Arm | rep | cache | TTFT (s) | KV hit rate | Query tokens | Token hits | "
            "Prefetches | Useful | Useful % | Late % | Lead (mean s) | "
            "Oracle window (s) | Waiting (max) | Notes |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in sorted(subset, key=lambda r: (r["arm"], int(r["rep"] or 1))):
            notes = []
            if row.get("trace_misses") not in ("", "0", None):
                notes.append("trace miss")
            if row.get("off_pin_requests") not in ("", "0", None):
                notes.append("off-pin")
            lines.append(
                f"| {row['arm']} | {row['rep']} | {row['cache_state']} | "
                f"{_fmt(row['ttft_s'], '.2f')} | "
                f"{_fmt(row['kv_hit_rate'], '.2%')} | "
                f"{_fmt(row['query_tokens'])} | {_fmt(row['token_hits'])} | "
                f"{_fmt(row['total_prefetches'])} | "
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
