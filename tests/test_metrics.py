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

    # Per-request phase timings, with the phantom's poisoned fields rejected.
    per = {e["request_id"]: e for e in m.per_request}
    assert abs(per["r1"]["queued_s"] - 0.2) < 1e-9
    assert abs(per["r1"]["prefill_s"] - 0.3) < 1e-9
    assert abs(per["r1"]["decode_s"] - 7.5) < 1e-9
    # ttft is queue + prefill by construction, so the three must agree.
    assert abs(per["r1"]["ttft_s"] - (per["r1"]["queued_s"] + per["r1"]["prefill_s"])) < 1e-9
    ghost = m.per_prefetch[0]
    assert ghost["decode_s"] is None, ghost["decode_s"]
    # 1.5s span minus 0.1s queue: derived, because the engine's own
    # `prefill_time` on a token-less request is a negative monotonic value.
    assert abs(ghost["prefill_s"] - 1.4) < 1e-6, ghost["prefill_s"]

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


def check_poisoned_intervals(tmp: Path) -> None:
    """A phantom's `prefill_time` is negative in the raw rows; it must not land."""
    cell = build_useful_cell(tmp / "poison")
    rows = _read(cell / "stats" / "finished_requests_engine0_x.jsonl")
    for row in rows:
        if row["prefetch_only"]:
            # What vLLM actually writes for a request that produced no token:
            # first_token_ts (0.0) - scheduled_ts (monotonic).
            row["prefill_time"] = -35847.219
            row["decode_time"] = 0.0
            row["queued_time"] = 0.05
    _write(cell / "stats" / "finished_requests_engine0_x.jsonl", rows)
    m = metrics.collect(cell, arm="ours", question_id="q5", t0=T0, t1=T0 + 100)
    for ghost in m.per_prefetch:
        assert ghost["prefill_s"] is not None and ghost["prefill_s"] >= 0.0, ghost
        assert ghost["decode_s"] is None, ghost
        assert ghost["queued_s"] == 0.05, ghost
    # The seed phantom spans 1s and queued 0.05s of it.
    seed = next(g for g in m.per_prefetch if g["agent_id"] == "seed")
    assert abs(seed["prefill_s"] - 0.95) < 1e-6, seed["prefill_s"]
    print("poisoned intervals rejected:", len(m.per_prefetch), "phantoms")
    print("\nall interval assertions passed")


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
    # Distinct counts are per agent, not per call: `react` is warmed twice and
    # counts once, and `warm` was warmed but never helped, which is the case
    # the ratio exists to expose.
    assert m.distinct_prefetch_agents == 3, m.distinct_prefetch_agents
    assert m.distinct_useful_agents == 2, m.distinct_useful_agents
    assert abs(m.distinct_useful_pct - 2 / 3) < 1e-9, m.distinct_useful_pct
    # Per-call and per-agent answer different questions and must not be equal
    # here, or the fixture is not exercising the difference.
    assert m.useful_prefetches == 2 and m.total_prefetches == 4

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


def build_timeline_cell(tmp: Path) -> Path:
    """A warmup phantom, a repeated agent, and three genuinely parallel calls."""
    cell = tmp / "ours" / "q4" / "rep1"
    sup = "langgraph:1:research_supervisor:supervisor"
    tools = "langgraph:1:research_supervisor:supervisor_tools:researcher_tools"
    rows = [
        # Population phase: must never reach the timeline.
        dict(request_id="w1", job_id="j4", agent_id="langgraph:*:**:supervisor",
             langgraph_node="supervisor", prefetch_only=True,
             arrival_ts=T0 + 0.1, finish_ts=T0 + 0.2, num_prompt_tokens=100),
        dict(request_id="p1", job_id="j4", agent_id=sup, langgraph_node="supervisor",
             prefetch_only=True, arrival_ts=T0 + 1.0, finish_ts=T0 + 1.019,
             num_prompt_tokens=838, num_local_cached_tokens=0),
        dict(request_id="s1", job_id="j4", agent_id=sup, langgraph_node="supervisor",
             prefetch_only=False, arrival_ts=T0 + 1.1, finish_ts=T0 + 5.0,
             queued_time=0.1, prefill_time=0.2, num_prompt_tokens=911,
             num_local_cached_tokens=320, num_generation_tokens=100),
        # Three concurrent tool calls: same agent id, must stay three bars.
        *[
            dict(request_id=f"t{i}", job_id="j4", agent_id=tools,
                 langgraph_node="researcher_tools", prefetch_only=False,
                 arrival_ts=T0 + 5.1 + i * 0.001, finish_ts=T0 + 30.0 + i * 4,
                 queued_time=0.1, prefill_time=8.0, num_prompt_tokens=13_000 + i,
                 num_local_cached_tokens=96, num_generation_tokens=200)
            for i in range(3)
        ],
        # Second turn of the same agent -> Chat #2, not a merge.
        dict(request_id="s2", job_id="j4", agent_id=sup, langgraph_node="supervisor",
             prefetch_only=False, arrival_ts=T0 + 45.0, finish_ts=T0 + 60.0,
             queued_time=0.1, prefill_time=0.4, num_prompt_tokens=2_065,
             num_local_cached_tokens=1_040, num_generation_tokens=300),
    ]
    _write(cell / "stats" / "finished_requests_engine0_x.jsonl", rows)
    (cell / "question_started_ts").write_text(f"{T0}\n")
    return cell


# Two concurrent tools plus a `tool_start` whose end never came, in the shape
# `echo_router._format` emits. `t2` and `t3` overlap on purpose: they run under
# one `asyncio.gather`, so only `call_id` can tell them apart.
TOOL_LOG_TEMPLATE = """\
(APIServer pid=1) INFO {t0} [echo_router.py:80] echo: \
event=tool_start agent_id={agent} call_id=call_a client_ns=1 tool=tavily_search
(APIServer pid=1) INFO {t1} [echo_router.py:80] echo: \
event=tool_start agent_id={agent} call_id=call_b client_ns=2 tool=tavily_search
(APIServer pid=1) INFO {t2} [echo_router.py:80] echo: \
event=tool_end agent_id={agent} call_id=call_b client_ns=3 elapsed_ms=412.0 tool=tavily_search
(APIServer pid=1) INFO {t3} [echo_router.py:80] echo: \
event=tool_end agent_id={agent} call_id=call_a client_ns=4 elapsed_ms=980.0 tool=tavily_search
(APIServer pid=1) INFO {t3} [echo_router.py:80] echo: \
event=tool_start agent_id={agent} call_id=call_c client_ns=5 tool=think_tool
"""


def check_tool_spans(tmp: Path) -> None:
    """Leaf tool spans pair on `call_id`, and an unpaired marker is reported."""
    from pitlane import timeline
    import datetime as _dt

    cell = build_timeline_cell(tmp / "tools")
    rows = _read(cell / "stats" / "finished_requests_engine0_x.jsonl")
    tools_agent = "langgraph:1:research_supervisor:supervisor_tools:researcher_tools"
    base = next(r["arrival_ts"] for r in rows if r["agent_id"] == tools_agent)

    def stamp(offset: float) -> str:
        return _dt.datetime.fromtimestamp(base + offset).strftime(
            "%Y-%m-%d %H:%M:%S.%f")[:-3]

    (cell / "server.log").write_text(TOOL_LOG_TEMPLATE.format(
        agent=tools_agent, t0=stamp(0.1), t1=stamp(0.2),
        t2=stamp(0.612), t3=stamp(1.08)))

    m = metrics.collect(cell, arm="ours", question_id="q6", t0=T0, t1=T0 + 100)
    by_call = {row["call_id"]: row for row in m.per_tool}
    assert set(by_call) == {"call_a", "call_b"}, sorted(by_call)
    # Interleaved ends: `call_b` closes first, so only the id can pair them.
    assert abs(by_call["call_a"]["elapsed_s"] - 0.98) < 2e-3, by_call["call_a"]
    assert abs(by_call["call_b"]["elapsed_s"] - 0.412) < 2e-3, by_call["call_b"]
    assert by_call["call_a"]["tool"] == "tavily_search"
    # `call_c` opened and never closed: counted, not silently dropped.
    assert m.unmatched_tool_markers == 1, m.unmatched_tool_markers
    assert any("unpaired tool marker" in w for w in m.warnings), m.warnings
    assert abs(m.tool_seconds - 1.392) < 4e-3, m.tool_seconds

    ordered = timeline.events(m)
    tools = [e for e in ordered if e.kind == "tool"]
    assert [e.index for e in tools] == [1, 2]
    assert "tavily_search" in tools[0].label and "ms)" in tools[0].label
    body = timeline.mermaid(m)
    assert body.count(":done,") == 2, body
    assert "doneTaskBkgColor" in body
    ids = [ln.split(",")[1].strip() for ln in body.splitlines()
           if any(f":{t}," in ln for t in ("crit", "active", "done"))]
    assert len(ids) == len(set(ids)), ids
    assert "leaf tool calls: **2**" in timeline.table(m)
    print("tool spans:", len(m.per_tool), "paired,",
          m.unmatched_tool_markers, "unpaired, %.3fs total" % m.tool_seconds)

    # A stock build logs `09-11 20:11:49`: no year, whole seconds. Requiring
    # the wide format silently dropped every one of its markers, which renders
    # as an arm that called no tools. The year comes from another line in the
    # same file -- here the access line the fixture prepends.
    stock = "\n".join(
        line.replace(stamp(0.1), "09-11 20:11:49")
            .replace(stamp(0.2), "09-11 20:11:49")
            .replace(stamp(0.612), "09-11 20:11:50")
            .replace(stamp(1.08), "09-11 20:11:51")
        for line in TOOL_LOG_TEMPLATE.format(
            agent=tools_agent, t0=stamp(0.1), t1=stamp(0.2),
            t2=stamp(0.612), t3=stamp(1.08)).splitlines()
    )
    year = _dt.datetime.fromtimestamp(base).year
    (cell / "server.log").write_text(
        f'(APIServer pid=1) INFO:     {year}-09-11 20:11:48.001 127.0.0.1:1 - '
        f'"POST /v1/echo HTTP/1.1" 200 OK\n{stock}\n'
    )
    m2 = metrics.collect(cell, arm="baseline", question_id="q6", t0=T0, t1=T0 + 100)
    stock_calls = {row["call_id"]: row for row in m2.per_tool}
    assert set(stock_calls) == {"call_a", "call_b"}, sorted(stock_calls)
    # Subtracting the stamps would give 2.0 and 1.0 -- the quantisation, not
    # the call. The marker's own `elapsed_ms` is what the client measured, and
    # it is what a span keeps when the stamps are too coarse to subtract.
    assert abs(stock_calls["call_a"]["elapsed_s"] - 0.980) < 1e-6, stock_calls
    assert abs(stock_calls["call_b"]["elapsed_s"] - 0.412) < 1e-6, stock_calls
    # The start stays on the server clock; only the length comes from the client.
    assert stock_calls["call_a"]["start_ts"] == stock_calls["call_b"]["start_ts"]
    assert abs((stock_calls["call_a"]["end_ts"] - stock_calls["call_a"]["start_ts"])
               - stock_calls["call_a"]["elapsed_s"]) < 1e-6
    assert stock_calls["call_a"]["tool"] == "tavily_search"
    print("stock-format log:", len(m2.per_tool), "paired from yearless "
          "whole-second stamps, durations from elapsed_ms")
    print("\nall tool-span assertions passed")


def check_unsplit_cached_tokens(tmp: Path) -> None:
    """An older build reports one `num_cached_tokens` and no local/external split.

    Reading only the split field made the whole arm look like it served no
    cached tokens at all -- 0 hits against a real prompt count -- which is the
    same shape as a genuinely cold cache and cannot be told apart from one.
    """
    cell = tmp / "continuum" / "q7" / "rep1"
    _write(cell / "stats" / "finished_requests_engine0_x.jsonl", [
        dict(request_id="r1", job_id="j1", arrival_ts=T0 + 1, finish_ts=T0 + 9,
             queued_time=0.2, prefill_time=0.3, decode_time=7.5,
             num_prompt_tokens=10_000, num_cached_tokens=4_000,
             num_generation_tokens=300),
        dict(request_id="r2", job_id="j1", arrival_ts=T0 + 10, finish_ts=T0 + 20,
             queued_time=0.1, prefill_time=0.4, decode_time=9.0,
             num_prompt_tokens=20_000, num_cached_tokens=6_000,
             num_generation_tokens=500),
    ])

    m = metrics.collect(cell, arm="continuum", question_id="q7", t0=T0, t1=T0 + 100)
    assert m.requests == 2, m.requests
    assert m.query_tokens == 30_000, m.query_tokens
    assert m.token_hits == 10_000, m.token_hits
    assert abs(m.kv_hit_rate - 1 / 3) < 1e-9, m.kv_hit_rate
    # The single field is local+external, so zero external is not a measurement.
    assert any("local+external" in w for w in m.warnings), m.warnings
    assert [r["token_hits"] for r in m.per_request] == [4_000, 6_000]

    # A split build keeps its exact meaning, and says nothing.
    split = metrics.collect(build_cell(tmp / "split"), arm="ours",
                            question_id="q1", t0=T0, t1=T0 + 100)
    assert split.token_hits == 10_000, split.token_hits
    assert split.external_token_hits == 1_000, split.external_token_hits
    assert not any("local+external" in w for w in split.warnings), split.warnings
    print("unsplit build: hits", m.token_hits, "of", m.query_tokens,
          "tokens, flagged as local+external")
    print("\nall cached-token assertions passed")


def check_truncated_stats(tmp: Path) -> None:
    """The access log answers more chats than the stats file has rows.

    `FileStatLogger` writes buffered and the collector runs before the server
    exits, so the tail of a run can simply be absent. It arrives as a smaller
    `requests` -- a shorter workload, not an error -- which is how a 43-request
    run came to be reported and compared as a 34-request one.
    """
    import datetime as _dt

    cell = tmp / "continuum" / "q8" / "rep1"
    stamps = [
        _dt.datetime.strptime(f"2026-09-12 03:{15 + i}:57.336",
                              "%Y-%m-%d %H:%M:%S.%f").timestamp()
        for i in range(5)
    ]
    # Five answered, but only the first three rows reached the file.
    _write(cell / "stats" / "finished_requests_engine0_x.jsonl", [
        dict(request_id=f"r{i}", job_id="j1", arrival_ts=stamps[i],
             finish_ts=stamps[i] + 1, num_prompt_tokens=1_000,
             num_cached_tokens=100, num_generation_tokens=10)
        for i in range(3)
    ])
    access = "\n".join(
        f'(APIServer pid=1) INFO:     2026-09-12 03:{15 + i}:57.336 127.0.0.1:5 - '
        f'"POST /v1/chat/completions HTTP/1.1" 200 OK'
        for i in range(5)
    )
    (cell / "server.log").write_text(access + "\n")
    m = metrics.collect(cell, arm="continuum", question_id="q8",
                        t0=stamps[0] - 1, t1=stamps[-1] + 1)
    assert m.requests == 3, m.requests
    assert any("5 chat completions but only 3" in w for w in m.warnings), m.warnings

    # A reused server's log spans other cells, and lines outside this cell's
    # window are another cell's work, not this one's missing rows.
    narrow = metrics.collect(cell, arm="continuum", question_id="q8",
                             t0=stamps[0] - 1, t1=stamps[2] + 0.5)
    assert not any("chat completions but only" in w for w in narrow.warnings), narrow.warnings
    print("truncated stats: warned on 5 served vs 3 collected; "
          "windowed count stays quiet")
    print("\nall truncation assertions passed")


def check_routing_from_log(tmp: Path) -> None:
    """A build that logs routing instead of recording it still gets lanes.

    The older fork does not carry `langgraph_node` into the engine, so its
    stats rows are anonymous and every chat span was dropped off the chart.
    The API layer states the same fields once per request against the id it
    hands to `generate`, which is the id the stats row is keyed by.
    """
    cell = tmp / "continuum" / "q9" / "rep1"
    _write(cell / "stats" / "finished_requests_engine0_x.jsonl", [
        dict(request_id="chatcmpl-aaa", job_id="1", arrival_ts=T0 + 1,
             finish_ts=T0 + 5, num_prompt_tokens=5_000, num_cached_tokens=500,
             num_generation_tokens=100),
        dict(request_id="chatcmpl-bbb", job_id="1", arrival_ts=T0 + 10,
             finish_ts=T0 + 15, num_prompt_tokens=6_000, num_cached_tokens=600,
             num_generation_tokens=120),
        # No line for this one: a gap stays a gap rather than borrowing.
        dict(request_id="chatcmpl-ccc", job_id="1", arrival_ts=T0 + 20,
             finish_ts=T0 + 25, num_prompt_tokens=7_000, num_cached_tokens=700,
             num_generation_tokens=140),
    ])
    (cell / "server.log").write_text(
        "(APIServer pid=1) INFO 2026-09-12 03:15:57.000 [serving_chat.py:257] "
        "request_routing: request_id=chatcmpl-aaa job_id=1 "
        "langgraph_node=supervisor\n"
        "(APIServer pid=1) INFO 2026-09-12 03:16:07.000 [serving_chat.py:257] "
        "request_routing: request_id=chatcmpl-bbb agent_id=langgraph:1:researcher "
        "langgraph_node=researcher\n"
    )

    m = metrics.collect(cell, arm="continuum", question_id="q9", t0=T0, t1=T0 + 100)
    by_id = {r["request_id"]: r for r in m.per_request}
    assert by_id["chatcmpl-aaa"]["langgraph_node"] == "supervisor", by_id
    assert by_id["chatcmpl-bbb"]["agent_id"] == "langgraph:1:researcher", by_id
    assert by_id["chatcmpl-ccc"]["langgraph_node"] is None, by_id

    # A build that records the fields itself is the authority on its own rows.
    own = build_cell(tmp / "own")
    (own / "server.log").write_text(
        "(APIServer pid=1) INFO 2026-09-12 03:15:57.000 [serving_chat.py:257] "
        "request_routing: request_id=r1 langgraph_node=WRONG\n"
    )
    kept = metrics.collect(own, arm="ours", question_id="q1", t0=T0, t1=T0 + 100)
    node = next(r["langgraph_node"] for r in kept.per_request if r["request_id"] == "r1")
    assert node == "research_supervisor", node
    print("routing from log: 2 of 3 filled, recorded fields not overwritten")
    print("\nall routing assertions passed")


def check_run_timeline(tmp: Path) -> None:
    """Combined charts: every arm, split per question, each arm rebased.

    `baseline` rows carry no `agent_id` -- only `/v1/agents/*` stamps one -- so
    the fixture leaves it empty for that arm, which is the shape that used to
    drop the whole arm off the chart.
    """
    import csv as _csv
    from pitlane import timeline

    run = tmp / "runtl"
    run.mkdir(parents=True, exist_ok=True)
    base = 1_800_000_000.0
    sup = "langgraph:{j}:research_supervisor:supervisor"
    reqs, pfs = [], []
    # `ours` ran 2.5 hours after `baseline`: on real clock time the two would
    # be distant blocks, which is the thing rebasing exists to fix.
    for arm, offset in (("baseline", 0.0), ("ours", 9000.0)):
        for job in (1, 2):
            t = base + offset + (job - 1) * 100
            reqs.append(dict(
                arm=arm, question_id="batch2", rep=1, job_id=job,
                # Baseline goes through plain /v1/chat/completions: no agent id.
                agent_id="" if arm == "baseline" else sup.format(j=job),
                langgraph_node="supervisor", arrival_ts=t, finish_ts=t + 4))
            if arm == "ours":
                pfs.append(dict(arm=arm, question_id="batch2", rep=1, job_id=job,
                                agent_id=sup.format(j=job),
                                langgraph_node="supervisor",
                                arrival_ts=t - 0.088, finish_ts=t - 0.069))
    for name, rows in (("requests.csv", reqs), ("prefetches.csv", pfs)):
        cols = ["arm", "question_id", "rep", "job_id", "agent_id",
                "langgraph_node", "arrival_ts", "finish_ts"]
        with (run / name).open("w", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    written = timeline.write_run(run)
    names = sorted(p.name for p in written)
    assert names == [
        "batch2__job1.md", "batch2__job1.mmd",
        "batch2__job1__baseline.md", "batch2__job1__baseline.mmd",
        "batch2__job1__ours.md", "batch2__job1__ours.mmd",
        "batch2__job2.md", "batch2__job2.mmd",
        "batch2__job2__baseline.md", "batch2__job2__baseline.mmd",
        "batch2__job2__ours.md", "batch2__job2__ours.mmd",
    ], names

    # A per-arm file holds that arm only, and keeps true wall-clock times: with
    # one arm there is nothing to rebase against, so the title does not claim a
    # rebase and the stamps are the engine's own.
    solo = (run / "timelines" / "batch2__job1__baseline.mmd").read_text()
    assert "section ours · " not in solo, solo
    assert "rebased" not in solo, solo
    solo_start = next(ln.split(", ")[-2] for ln in solo.splitlines() if ":active," in ln)
    assert solo_start.startswith("20"), solo_start

    body = (run / "timelines" / "batch2__job1.mmd").read_text()
    # The arm with no agent id must still be drawn, on the same lane name.
    assert "section baseline · supervisor" in body, body
    assert "section ours · supervisor" in body, body
    # job 2 must not leak into job 1's chart
    assert ":2:" not in body.split("gantt")[1], body
    # Rebasing: each arm's first event lands on the same stamp, though the two
    # arms ran 2.5 hours apart. Without it the chart compares nothing.
    firsts = {}
    arm = None
    for line in body.splitlines():
        if line.strip().startswith("section "):
            arm = line.split("section ", 1)[1].split(" · ")[0]
        elif ":active," in line or ":crit," in line:
            firsts.setdefault(arm, line.split(", ")[-2])
    assert set(firsts) == {"baseline", "ours"}, firsts
    assert len(set(firsts.values())) == 1, firsts

    # The phase split rides in the task name, and a phase that does not exist
    # is omitted rather than written as zero: a phantom has no decode phase.
    from pitlane.timeline import Event
    chat = Event("a", "chat", 1, 0.0, 8.0, queued_s=0.1, prefill_s=0.3, decode_s=7.5)
    assert chat.label == "Chat #1 (q 0.10s · p 0.30s · d 7.50s)", chat.label
    ghost = Event("a", "prefetch", 1, 0.0, 0.019, queued_s=0.002, prefill_s=0.017)
    assert "d " not in ghost.label, ghost.label
    assert Event("a", "chat", 2, 0.0, 1.0).label == "Chat #2"

    table = (run / "timelines" / "batch2__job1.md").read_text()
    assert "**baseline**" in table and "**ours**" in table
    assert "t=0 is this arm's first event" in table
    assert "Prefill (s)" in table and "Decode (s)" in table
    print("run timeline:", names)
    print("\nall run-timeline assertions passed")


def check_timeline(tmp: Path) -> None:
    from pitlane import timeline

    cell = build_timeline_cell(tmp)
    m = metrics.collect(cell, arm="ours", question_id="q4", rep=1, t0=T0, t1=T0 + 100)
    events = timeline.events(m)

    # The population phantom is gone; the concrete one is not.
    assert all("*" not in e.agent_id for e in events), [e.agent_id for e in events]
    assert sum(1 for e in events if e.kind == "prefetch") == 1
    assert sum(1 for e in events if e.kind == "chat") == 5, events

    # Parallel calls stay separate and are numbered, never merged.
    tools = [e for e in events if e.agent_id.endswith("researcher_tools")]
    assert [e.index for e in tools] == [1, 2, 3], [e.index for e in tools]
    assert len({(e.start, e.end) for e in tools}) == 3

    # A repeated agent numbers chronologically. Lanes are graph nodes now, so
    # both arms land on the same row name and `supervisor` is exact.
    sup = [e for e in events if e.agent_id == "supervisor" and e.kind == "chat"]
    assert [e.index for e in sup] == [1, 2], [e.agent_id for e in events]

    # The short prefetch keeps its true length, and says so in its label.
    warm = next(e for e in events if e.kind == "prefetch")
    assert abs(warm.duration_s - 0.019) < 1e-6, warm.duration_s
    assert "19.0 ms" in warm.label, warm.label

    body = timeline.mermaid(m)
    assert body.count(":crit,") == 1 and body.count(":active,") == 5
    assert "langgraph:" not in body.split("dateFormat")[1], "prefix not stripped"
    # Every task id is unique, or Mermaid silently drops the duplicates.
    ids = [ln.split(",")[1].strip() for ln in body.splitlines() if ":crit," in ln or ":active," in ln]
    assert len(ids) == len(set(ids)) == 6, ids

    md = timeline.table(m)
    assert "chat-completion calls: **5**" in md
    assert "concrete-agent prefetches: **1**" in md

    written = timeline.write(tmp / "timelines", m)
    print("timeline:", [p.name for p in written],
          "| chats=5 prefetches=1, 3 parallel tool calls preserved")
    print("\nall timeline assertions passed")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        check(Path(tmp))
        check_useful(Path(tmp))
        check_poisoned_intervals(Path(tmp))
        check_lead_markers(Path(tmp))
        check_timeline(Path(tmp))
        check_tool_spans(Path(tmp))
        check_unsplit_cached_tokens(Path(tmp))
        check_truncated_stats(Path(tmp))
        check_routing_from_log(Path(tmp))
        check_run_timeline(Path(tmp))
