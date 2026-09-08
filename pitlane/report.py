"""results.csv and summary.md.

One row per cell in the CSV; the markdown pivots it to one table per question
with a row per arm, which is the shape the write-up needs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pitlane.metrics import Metrics

_COLUMNS = [
    "arm", "question_id", "rep", "cache_state",
    "ttft_s", "kv_hit_rate", "query_tokens", "token_hits",
    "external_token_hits", "workflow_output_tokens",
    "total_prefetches", "late_prefetches", "late_prefetch_pct", "unused_prefetches",
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
            "Prefetches | Late % | Waiting (max) | Notes |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
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
                f"{_fmt(row['late_prefetch_pct'], '.0%')} | "
                f"{_fmt(row['sched_waiting_max'])} | {', '.join(notes) or '—'} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_summary(run_dir: Path) -> Path:
    path = run_dir / "summary.md"
    path.write_text(summary(run_dir / "results.csv"))
    return path


def write_metrics(cell: Path, metrics: Metrics) -> Path:
    path = cell / "metrics.json"
    path.write_text(json.dumps(metrics.to_dict(), indent=2) + "\n")
    return path
