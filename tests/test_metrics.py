"""Metrics collection against synthetic artifacts.

The shapes here mirror what vLLM's FileStatLogger writes; the point is that the
arithmetic and the joins are pinned down without a GPU in the loop.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitlane import metrics  # noqa: E402

T0 = 1_000_000.0


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def build_cell(tmp: Path) -> Path:
    cell = tmp / "ours" / "q1" / "rep1"
    stats = cell / "stats"
    requests = [
        # supervisor turn: 10k prompt, 2k already in HBM, first token 0.5s in
        dict(request_id="r1", job_id="j1", agent_id="a1", langgraph_node="research_supervisor",
             prefetch_only=False, arrival_ts=T0 + 1, finish_ts=T0 + 9,
             queued_time=0.2, prefill_time=0.3, decode_time=7.5,
             num_prompt_tokens=10_000, num_cached_tokens=2_000,
             num_local_cached_tokens=2_000, num_external_cached_tokens=0,
             num_generation_tokens=300),
        # phantom warming the next node, finishes at T0+11
        dict(request_id="p1", job_id="j1", agent_id="a1", langgraph_node="researcher",
             prefetch_only=True, arrival_ts=T0 + 9.5, finish_ts=T0 + 11.0,
             queued_time=0.1, prefill_time=1.4, num_prompt_tokens=8_000,
             num_local_cached_tokens=0, num_generation_tokens=1),
        # the consumer arrives at T0+10.5, i.e. BEFORE the phantom finished: late
        dict(request_id="r2", job_id="j1", agent_id="a1", langgraph_node="researcher",
             prefetch_only=False, arrival_ts=T0 + 10.5, finish_ts=T0 + 20,
             queued_time=0.5, prefill_time=1.0, decode_time=8.0,
             num_prompt_tokens=20_000, num_cached_tokens=6_000,
             num_local_cached_tokens=5_000, num_external_cached_tokens=1_000,
             num_generation_tokens=700),
        # final report: first token at 30 + 1.0 + 2.0 = T0+33 -> TTFT 33s
        dict(request_id="r3", job_id="j1", agent_id="a1", langgraph_node="final_report_generation",
             prefetch_only=False, arrival_ts=T0 + 30, finish_ts=T0 + 60,
             queued_time=1.0, prefill_time=2.0, decode_time=27.0,
             num_prompt_tokens=70_000, num_cached_tokens=3_000,
             num_local_cached_tokens=3_000, num_external_cached_tokens=0,
             num_generation_tokens=1_000),
    ]
    _write(stats / "finished_requests_engine0_x.jsonl", requests)
    _write(stats / "scheduler_engine0_x.jsonl", [
        dict(event="clock_anchor", ts=T0, monotonic=42.0, engine_index=0, pid=1),
        dict(ts=T0 + 2, num_scheduled_reqs=1, num_new_scheduled_reqs=1,
             num_running_reqs=1, num_waiting_reqs=0, num_preempted_reqs=0),
        dict(ts=T0 + 11, num_scheduled_reqs=2, num_new_scheduled_reqs=1,
             num_running_reqs=2, num_waiting_reqs=3, num_preempted_reqs=1),
        dict(ts=T0 + 40, num_scheduled_reqs=1, num_new_scheduled_reqs=0,
             num_running_reqs=1, num_waiting_reqs=0, num_preempted_reqs=0),
        dict(ts=T0 + 9_999, num_scheduled_reqs=9, num_new_scheduled_reqs=9,
             num_running_reqs=9, num_waiting_reqs=9, num_preempted_reqs=9),  # out of window
    ])
    _write(cell / "divergence.jsonl", [
        dict(reason="pinned", pinned_exact_tokens=True, live_completion_tokens=300,
             pinned_max_tokens=300),
        dict(reason="pinned", pinned_exact_tokens=True, live_completion_tokens=690,
             pinned_max_tokens=700),  # off-pin
    ])
    _write(cell / "agent_prefetch.jsonl", [dict(agent_id="a1", wait=False, top_k=1)])
    (cell / "question_started_ts").write_text(f"{T0}\n")
    return cell


def check(tmp: Path) -> None:
    cell = build_cell(tmp)
    m = metrics.collect(cell, arm="ours", question_id="q1", t0=T0, t1=T0 + 100)

    assert m.requests == 3, m.requests
    assert m.query_tokens == 100_000, m.query_tokens
    assert m.token_hits == 10_000, m.token_hits
    assert abs(m.kv_hit_rate - 0.10) < 1e-9, m.kv_hit_rate
    assert m.external_token_hits == 1_000
    assert m.workflow_output_tokens == 2_000, m.workflow_output_tokens

    assert m.ttft_source_node == "final_report_generation"
    assert abs(m.ttft_s - 33.0) < 1e-6, m.ttft_s

    assert m.total_prefetches == 1
    assert m.late_prefetches == 1, m.late_prefetches
    assert m.unused_prefetches == 0
    assert m.late_prefetch_pct == 1.0

    assert m.sched_running_max == 2, m.sched_running_max
    assert m.sched_waiting_max == 3
    assert m.sched_preempted_total == 1
    assert m.sched_scheduled_total == 4, m.sched_scheduled_total

    assert m.off_pin_requests == 1
    assert m.trace_misses == 0
    assert any("pinned token count" in w for w in m.warnings), m.warnings
    print("metrics:", json.dumps(
        {k: v for k, v in m.to_dict().items() if v not in (None, 0, "", [])}, indent=2))
    print("\nall assertions passed")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        check(Path(tmp))
