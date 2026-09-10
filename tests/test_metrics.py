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


def _read(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    # Lead is 10.5 - 9.5 = 1.0s, well past the 0.1s floor, so not late by lead.
    assert m.late_prefetches == 0, m.late_prefetches
    assert m.late_prefetch_pct == 0.0
    assert abs(m.prefetch_lead_mean_s - 1.0) < 1e-9, m.prefetch_lead_mean_s
    # It still had not finished when the consumer arrived (finish T0+11 vs
    # arrival T0+10.5), so the blocks were not there and it cannot be useful.
    assert m.unused_prefetches == 0
    assert m.useful_prefetches == 0, m.useful_prefetches
    assert m.useful_prefetch_pct == 0.0

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


def build_useful_cell(tmp: Path) -> Path:
    """Three phantoms, one per case the useful metric has to separate."""
    cell = tmp / "ours" / "q2" / "rep1"
    stats = cell / "stats"
    requests = [
        # -- useful: prefix nothing had computed, seeded and prefilled, then hit.
        dict(request_id="p1", job_id="j2", agent_id="seed", langgraph_node="compress_research",
             prefetch_only=True, arrival_ts=T0 + 1, finish_ts=T0 + 2,
             num_prompt_tokens=3_200, num_cached_tokens=0,
             num_local_cached_tokens=0, num_external_cached_tokens=0,
             num_generation_tokens=1),
        dict(request_id="r1", job_id="j2", agent_id="seed", langgraph_node="compress_research",
             prefetch_only=False, arrival_ts=T0 + 3, finish_ts=T0 + 20,
             queued_time=0.1, prefill_time=0.2, decode_time=16.0,
             num_prompt_tokens=3_236, num_cached_tokens=3_168,
             num_local_cached_tokens=3_168, num_external_cached_tokens=0,
             num_generation_tokens=100),

        # -- not useful: the phantom's own prefix was already HBM-resident, so
        # it promoted nothing. Its consumer's large hit was happening anyway.
        dict(request_id="p2", job_id="j2", agent_id="warm", langgraph_node="supervisor",
             prefetch_only=True, arrival_ts=T0 + 21, finish_ts=T0 + 22,
             num_prompt_tokens=2_000, num_cached_tokens=2_000,
             num_local_cached_tokens=2_000, num_external_cached_tokens=0,
             num_generation_tokens=1),
        dict(request_id="r2", job_id="j2", agent_id="warm", langgraph_node="supervisor",
             prefetch_only=False, arrival_ts=T0 + 23, finish_ts=T0 + 30,
             queued_time=0.1, prefill_time=0.2, decode_time=6.0,
             num_prompt_tokens=2_065, num_cached_tokens=2_048,
             num_local_cached_tokens=2_048, num_external_cached_tokens=0,
             num_generation_tokens=100),

        # -- fan-out: two phantoms for one agent, one consumer. Only the better
        # of the two may be credited, or the same hit is banked twice.
        dict(request_id="p3a", job_id="j2", agent_id="react", langgraph_node="researcher",
             prefetch_only=True, arrival_ts=T0 + 31, finish_ts=T0 + 32,
             num_prompt_tokens=900, num_cached_tokens=0,
             num_local_cached_tokens=0, num_external_cached_tokens=0,
             num_generation_tokens=1),
        dict(request_id="p3b", job_id="j2", agent_id="react", langgraph_node="researcher",
             prefetch_only=True, arrival_ts=T0 + 31, finish_ts=T0 + 32,
             num_prompt_tokens=448, num_cached_tokens=0,
             num_local_cached_tokens=0, num_external_cached_tokens=0,
             num_generation_tokens=1),
        dict(request_id="r3", job_id="j2", agent_id="react", langgraph_node="researcher",
             prefetch_only=False, arrival_ts=T0 + 33, finish_ts=T0 + 40,
             queued_time=0.1, prefill_time=0.2, decode_time=6.0,
             num_prompt_tokens=893, num_cached_tokens=896,
             num_local_cached_tokens=896, num_external_cached_tokens=0,
             num_generation_tokens=100),
    ]
    _write(stats / "finished_requests_engine0_x.jsonl", requests)
    _write(cell / "agent_prefetch.jsonl", [dict(agent_id="seed", wait=False, top_k=1)])
    (cell / "question_started_ts").write_text(f"{T0}\n")
    return cell


# Verbatim shapes from a real server.log, so the parser is pinned against what
# `echo_router._format` actually emits rather than against a guess.
ECHO_LOG = """\
(APIServer pid=429518) INFO 2026-09-09 19:02:03.915 [echo_router.py:80] echo: \
event=min_lead agent_id=seed client_ns=353831275638030 issuer=langgraph job_id=1 \
predicted_node=compress_research producer_node=researcher source=task_start
(APIServer pid=429518) INFO 2026-09-09 19:02:03.815 [echo_router.py:80] echo: \
event=max_lead agent_id=seed client_ns=353831275638030 issuer=workflow job_id=1 \
route=replay_prefetch
(APIServer pid=429518) INFO 2026-09-09 19:02:03.900 [echo_router.py:80] echo: \
event=prefetch_skipped agent_id=seed issuer=langgraph reason=claimed_by_other_caller
(APIServer pid=429518) INFO 2026-09-09 19:02:03.900 [api_router.py:635] not an echo line
"""


def check_lead_markers(tmp: Path) -> None:
    """min_lead / max_lead come off the server log, and the value is theirs alone."""
    import datetime as _dt

    cell = build_useful_cell(tmp / "leads")
    # Shift the rows so the "seed" request sits after the marker lines; only the
    # ordering matters here, since the reported value is marker-to-marker.
    arrival = _dt.datetime.strptime(
        "2026-09-09 19:02:04.915", "%Y-%m-%d %H:%M:%S.%f").timestamp()
    rows = _read(cell / "stats" / "finished_requests_engine0_x.jsonl")
    shift = arrival - next(r["arrival_ts"] for r in rows
                           if r["agent_id"] == "seed" and not r["prefetch_only"])
    for row in rows:
        row["arrival_ts"] += shift
        row["finish_ts"] += shift
    _write(cell / "stats" / "finished_requests_engine0_x.jsonl", rows)
    (cell / "server.log").write_text(ECHO_LOG)

    m = metrics.collect(cell, arm="ours", question_id="q3",
                        t0=rows[0]["arrival_ts"] - 1, t1=rows[-1]["finish_ts"] + 1)
    per = {e["agent_id"]: e for e in m.per_request}
    # Two lead markers parsed; `prefetch_skipped` and the api_router line are not.
    assert m.lead_markers == 2, m.lead_markers
    # max_lead at .815, min_lead at .915: the oracle knew 100ms before the graph
    # runtime could have, and that span is the whole metric.
    assert abs(per["seed"]["lead_window_s"] - 0.1) < 1e-6, per["seed"]["lead_window_s"]
    assert per["seed"]["max_lead_ts"] < per["seed"]["min_lead_ts"]
    assert per["warm"]["lead_window_s"] is None
    assert abs(m.lead_window_mean_s - 0.1) < 1e-6
    assert not [w for w in m.warnings if "echo" in w], m.warnings

    # Markers that name an agent this cell never served must not be silent: the
    # columns would just come out empty and read as "no markers were emitted".
    orphan = build_useful_cell(tmp / "orphans")
    (orphan / "server.log").write_text(ECHO_LOG)
    o = metrics.collect(orphan, arm="ours", question_id="q3", t0=T0, t1=T0 + 100)
    assert o.lead_markers == 2, o.lead_markers
    assert o.lead_window_mean_s is None
    assert any("matched no request" in w for w in o.warnings), o.warnings

    print("lead markers:", m.lead_markers,
          "window=%.3fs" % per["seed"]["lead_window_s"],
          "| orphan warning:", any("matched no request" in w for w in o.warnings))
    print("\nall lead-marker assertions passed")


def check_useful(tmp: Path) -> None:
    cell = build_useful_cell(tmp)
    m = metrics.collect(cell, arm="ours", question_id="q2", t0=T0, t1=T0 + 100)

    assert m.total_prefetches == 4, m.total_prefetches
    # Every phantom leads its consumer by 2s, so none is late by lead, and all
    # four finished a second before it arrived.
    assert m.late_prefetches == 0, m.late_prefetches
    assert m.unused_prefetches == 0, m.unused_prefetches
    assert abs(m.prefetch_lead_mean_s - 2.0) < 1e-9, m.prefetch_lead_mean_s

    # p1 (seed) and one of p3a/p3b. p2 promoted a prefix HBM already held.
    assert m.useful_prefetches == 2, m.useful_prefetches
    assert m.useful_prefetch_pct == 0.5, m.useful_prefetch_pct
    # p3a and p3b share one consumer, so the fan-out is credited once.
    # Per-request attribution: the seed is charged to r1, the already-warm
    # promotion to r2, and both react warms to r3 with only the better credited.
    per = {e["request_id"]: e for e in m.per_request}
    assert len(per) == 3, per.keys()
    assert per["r1"]["agent_id"] == "seed" and per["r1"]["job_id"] == "j2"
    assert per["r1"]["prefetches"] == 1 and per["r1"]["useful"] is True
    assert per["r1"]["credited_tokens"] == 3_168, per["r1"]["credited_tokens"]
    assert per["r2"]["prefetches"] == 1 and per["r2"]["useful"] is False
    assert per["r2"]["credited_tokens"] == 0
    assert per["r3"]["prefetches"] == 2, per["r3"]["prefetches"]
    assert per["r3"]["useful"] is True
    assert per["r3"]["credited_tokens"] == 896, per["r3"]["credited_tokens"]
    # The cell's counters are sums over those rows, never separately derived.
    assert m.useful_prefetches == sum(1 for e in m.per_request if e["useful"])
    assert m.late_prefetches == sum(e["late_prefetches"] for e in m.per_request)
    assert [e["seq"] for e in m.per_request] == [1, 2, 3]
    print("useful:", m.useful_prefetches, "/", m.total_prefetches,
          "=", m.useful_prefetch_pct)
    for entry in m.per_request:
        print("  ", entry["seq"], entry["agent_id"], entry["langgraph_node"],
              "prefetches=%d" % entry["prefetches"],
              "useful=%s" % entry["useful"],
              "credited=%d" % entry["credited_tokens"],
              "lead=%s" % entry["prefetch_lead_s"])

    # Same rows, but the phantoms leave 50ms in front of their consumers: every
    # one is late by lead even though all of them still finished in time.
    tight = build_useful_cell(tmp / "tight")
    rows = _read(tight / "stats" / "finished_requests_engine0_x.jsonl")
    for row in rows:
        if row["prefetch_only"]:
            consumer = next(r for r in rows if not r["prefetch_only"]
                            and r["agent_id"] == row["agent_id"])
            row["arrival_ts"] = consumer["arrival_ts"] - 0.05
            row["finish_ts"] = consumer["arrival_ts"] - 0.01
    _write(tight / "stats" / "finished_requests_engine0_x.jsonl", rows)
    t = metrics.collect(tight, arm="ours", question_id="q2", t0=T0, t1=T0 + 100)
    assert t.late_prefetches == 4, t.late_prefetches
    assert t.late_prefetch_pct == 1.0
    # Late by lead, but the blocks did land, so the cache effect is unchanged.
    assert t.useful_prefetches == 2, t.useful_prefetches
    assert abs(t.prefetch_lead_mean_s - 0.05) < 1e-9, t.prefetch_lead_mean_s
    print("tight leads:", t.late_prefetch_pct, "late,", t.useful_prefetches, "useful")

    print("\nall useful-prefetch assertions passed")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        check(Path(tmp))
        check_useful(Path(tmp))
        check_lead_markers(Path(tmp))
