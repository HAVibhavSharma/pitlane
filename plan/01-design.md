# pitlane — design

The module is `pitlane`. It automates the manual A/B loop: bring up a serving stack, wait for it, run one
LangGraph workflow prompt against it, tear down, collect metrics. Target repos:
`open_deep_research` and `swe-agent` (both expose `tests/run_evaluate.py` and
`tests/run_evaluate_node_eviction.py`).

Mode for now: **isolated prompt** — one question per measurement unit, full
setup and reset before the next. Batching several questions onto one server
boot is a config flag, not the default.

---

## 1) Key features of the module

- **One command per experiment.** `bench run --exp exp1 --questions 1` walks
  every arm × question × repetition and leaves a results table behind.
- **Arms are data, not code.** `arms.yaml` holds the exact launch command,
  env overlay and workflow flags for each of: `record`, `baseline`,
  `continuum`, `ours`. Adding an arm is a YAML block.
- **Workflow adapters.** The runner knows how to invoke ODR and swe-agent;
  they differ only in script name and flag spelling (`--max-queries` /
  `--completions-per-query` vs `--max-instances` /
  `--completions-per-instance`).
- **Trace-aware.** Checks `ODR_TRACE_PATH` for a trace covering the requested
  question set; runs the record phase only if it is missing (default 10
  questions), then pins every eval arm to it (`ODR_TRACE_MODE=pinned`,
  `ODR_TRACE_ON_MISS=strict`).
- **Refuses to hang the machine.** Preflight on GPU 0, host RAM, disk and
  port 8000; abort with a clear reason rather than OOM the box.
- **Everything in tmux, fixed session names.** Survives an ssh drop; every
  session is attachable mid-run for eyeballing.
- **Per-question metric epoch.** `/v1/kv_metrics/reset` + HBM flush + a fresh
  stats window per question, so questions are never pooled by accident.
- **Idempotent and resumable.** A completed `(arm, question, rep)` cell is
  skipped on re-run unless `--force`.
- **No sudo anywhere.** tmux, HTTP polling, file reads.

---

## 2) Key components

| Component | Responsibility |
|---|---|
| `preflight` | GPU 0 free (`nvidia-smi`: no other process, free VRAM ≥ threshold), host RAM free ≥ `l1-size-gb` + margin, `/disk2` free space, port 8000 unbound, redis reachable (arm `ours` only). Hard-fails before anything starts. |
| `resource_monitor` | Samples GPU util/mem, host RAM, disk every ~5 s into `resources.csv` for the whole run; kills the run if RAM or VRAM crosses the abort threshold. |
| `tmux_supervisor` | Fixed session names (`lmcache`, `vllm_baseline`, `vllm_continuum`, `vllm_ours`, `workflow`). Create / kill / send-keys / `capture-pane -pJ` / wait-for-exit-sentinel. Idle sessions from a crashed run are reaped at start. |
| `lmcache_manager` | **Wipe and restart are one atomic operation**: stop the running LMCache server, delete `$LMCACHE_L2_DIR`, start it again, wait for the port. Never one without the other — the server's L1 index is in memory, so wiping the dir under a live server leaves it serving keys whose backing files are gone, and restarting without wiping carries the previous cell's L2 in. Skipped for `continuum` (that arm uses `LMCACHE_CONFIG_FILE` + `LMCacheConnectorV1`, no separate server). |
| `server_manager` | Launches the arm's `vllm serve` from the arm's repo dir with its env overlay, redirecting to the arm's log (`test.log` / `test_cont.log` / `test_our.log`), then **readiness-gates** on `GET /v1/models` (plus `/health`) with a timeout, tailing the log for fatal patterns so a dead boot fails fast instead of at the timeout. |
| `workflow_runner` | Builds the full env (`.env` + arm overlay + per-question overrides), runs the adapter's command in the `workflow` session, streams to `workflow.log`, waits on exit code. |
| `question_selector` | Resolves the question set (index or id) from the benchmark the workflow uses; keeps question identity stable across arms so cells line up. |
| `epoch_manager` | Between questions on a shared server: `POST /v1/kv_metrics/reset`, HBM flush (leave the workflow's default flush on), stats-file offset bookmark, fresh `ODR_TRACE_REPORT` and agent-log paths. |
| `metrics_collector` | Parses `VLLM_REQUEST_STATS_DIR/finished_requests_engine0_*.jsonl`, the divergence report, and the vLLM agent log into one `metrics.json` per cell. |
| `reporter` | Aggregates cells into `results.csv` + a per-question comparison table (one row per arm: TTFT, hit rate, query tokens, token hits, late prefetch %), plus a per-run `summary.md`. |
| `state_store` | `state.json` per run: which cells are done, their artifact paths, git SHAs of all three repos, and the resolved env (secrets redacted). |

---

## 3) Keys / config — `.env` (force-inherited, user-owned)

Only what genuinely has to be secret or machine-specific. The orchestrator
loads this file, overlays the arm's own env, and refuses to start if a
required key is missing.

```dotenv
# --- secrets (the only real keys) ---
TAVILY_API_KEY=...
LANGSMITH_API_KEY=...          # optional; unset disables tracing
OPENAI_API_KEY=EMPTY           # vLLM ignores it, the client requires it

# --- where things live ---
BENCH_WORKFLOW_REPO=/home/vibhav/open_deep_research      # workflow under test
VLLM_BASELINE_REPO=/home/vibhav/Build/vllm-baseline
VLLM_CONTINUUM_REPO=/home/vibhav/Build/vllm-continuum
VLLM_OURS_REPO=/home/vibhav/Build/KVCOMM-VLLM/vllm

# one virtualenv per build -- three vllm checkouts cannot share site-packages
VLLM_BASELINE_VENV=/home/vibhav/venvs/baseline
VLLM_CONTINUUM_VENV=/home/vibhav/venvs/continuum
VLLM_OURS_VENV=/home/vibhav/venvs/ours
WORKFLOW_VENV=/home/vibhav/venvs/odr
BENCH_ROOT=/disk2/vibhav/bench                          # all run artifacts
ODR_TRACE_DIR=/disk2/vibhav/traces
TAVILY_CACHE_DIR=/disk2/vibhav/tavily_cache
LMCACHE_L2_DIR=/disk2/vibhav/lmcache                    # wiped per start

# --- knobs that change what is measured ---
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct-AWQ
ODR_FROZEN_DATE=2026-08-04
KV_FORECAST_REDIS_URL=redis://127.0.0.1:6379/0
BENCH_GPU=0
BENCH_MIN_FREE_RAM_GB=260        # l1-size-gb 200 + margin
BENCH_SERVER_READY_TIMEOUT_S=1800
BENCH_PREFETCH_LEAD_MIN_S=0.1    # below this lead a phantom counts as late
```

Each vLLM build is installed into **its own virtualenv**, and the four `*_VENV`
keys are how pitlane finds them. This is not a convenience: three checkouts of
`vllm` cannot share one `site-packages`, so with them unset `vllm serve` resolves
on `PATH` and every arm boots whichever build the shell happened to activate —
a run that completes, produces plausible numbers, and compares a stack against
itself. `stack.start_server` invokes `<venv>/bin/vllm` by absolute path and sets
`VIRTUAL_ENV` plus a `PATH` prefix for anything it spawns; `workflow.run` does
the same with `<WORKFLOW_VENV>/bin/python`. An arm may override which entry it
uses with `venv = "..."` in `arms.toml`, which otherwise defaults to its `repo`
key. `pitlane preflight` reports each one and warns when it is unset, since
falling back to `PATH` is a fallback and never the intent.

Everything else from the manual runbook (`MODEL_PROVIDER`, the four
`*_MODEL_MAX_TOKENS`, `LANGGRAPH_PROMPT_PSEUDO_DYNAMIC`, the
`LANGGRAPH_VLLM_AGENT_*` block, `VLLM_NODE_EVICTION_*`, `ODR_TRACE_*`) is
**derived per arm** in `arms.yaml` — not user-edited per run, so an arm can
never accidentally inherit another arm's flags. `LANGGRAPH_VLLM_AGENT_*` is
explicitly `unset` in every arm but `ours`.

---

## 4) Workflow

```
preflight ──► resolve question set ──► need a trace? ──► [record phase]
                                                          │
        ┌─────────────────────────────────────────────────┘
        ▼
for question q in Q:            # isolated-prompt mode
  for arm in [baseline, continuum, ours]:     # interleaved, not blocked
    for rep in 1..R:
      setup(arm) → run(q) → collect → teardown(arm)
```

**setup(arm)**
1. Kill leftover sessions.
2. `baseline` / `ours`: **stop LMCache → wipe `$LMCACHE_L2_DIR` → start LMCache → wait for port** (one step; never wipe under a live server). `continuum`: skip.
3. Start the arm's vLLM in its own tmux session, log to the arm's log file.
4. Poll `/v1/models` until ready (or fail on log error pattern / timeout).
5. `POST /v1/kv_metrics/reset`; note stats-file offset and wall-clock `t0`.

**run(q)**
6. Compose env: `.env` + arm overlay + per-cell paths
   (`ODR_TRACE_REPORT=<cell>/divergence.jsonl`,
   `LANGGRAPH_VLLM_AGENT_LOG_PATH=<cell>/agent_prefetch.jsonl`,
   `VLLM_REQUEST_STATS_DIR=<cell>/stats/`).
7. Run the adapter command for one question, one completion, in the
   `workflow` session; wait for exit.

**collect** — record `t1`, copy/segment the artifacts into the cell dir,
compute `metrics.json`, append a row to `results.csv`.

**teardown(arm)** — stop vLLM, stop LMCache, confirm VRAM released, wait for
GPU idle before the next arm boots.

Record phase is the same shape with `ODR_TRACE_MODE=record`, the baseline
stack, `--ablation-mode baseline --skip-cold-phase --no-kv-metrics-reset`, and
N questions in one pass (default 10) — recording is workload definition, not
measurement.

*Batching:* `reuse_server_across_questions: true` keeps one boot for several
questions; the epoch manager still resets metrics, flushes HBM and opens a new
stats window per question. It does **not** touch LMCache: a wipe needs a
restart, and a restart needs vLLM to reconnect, so cold-cache-per-question and
reuse-the-server are mutually exclusive by construction. That is exactly the
warmth risk you were avoiding manually, so with reuse on the run stamps
`cache_state: warm` on every cell after the first and the reporter refuses to
average warm and cold cells together.

---

## 5) Metrics and where they land

Source of truth is vLLM's per-request JSONL
(`VLLM_REQUEST_STATS_DIR/finished_requests_engine0_*.jsonl`), which already
carries `job_id`, `agent_id`, `langgraph_node`, `prefetch_only`,
`num_prompt_tokens`, `num_cached_tokens`, `num_local_cached_tokens`,
`num_external_cached_tokens`, `num_computed_tokens`, `queued_time`,
`prefill_time`, `decode_time`, `e2e_latency`, `arrival_ts`, `finish_ts`.
Rows are scoped to the cell by `job_id`, falling back to the `[t0, t1]`
arrival window.

| Metric | Definition | Computed from |
|---|---|---|
| **TTFT** | Time from the user submitting the question to the first token of the final output. One number per question. | runner stamp + request JSONL |
| **KV hit rate** | Tokens already resident in HBM / query tokens: `Σ num_local_cached_tokens / Σ num_prompt_tokens` over real requests. External (LMCache) hits reported as a separate column, never folded in. | request JSONL |
| **Query tokens** | `Σ num_prompt_tokens` — every prompt token the question sent to the server. The hit rate's denominator, as an absolute count. | request JSONL |
| **Token hits** | `Σ num_local_cached_tokens` — how many of those were already in HBM. The hit rate's numerator, as an absolute count. | request JSONL |
| **Total prefetches** | Count of phantoms issued for the question — one per `/v1/agents/prefetch` fan-out, i.e. `prefetch_only=true` rows. The denominator of the late %, reported as an absolute count so a small percentage of a handful of prefetches is not read as a small percentage of many. | `agent_prefetch.jsonl` + request JSONL |
| **Late prefetch %** | Of the phantoms issued for the question: late if it left too close in front of the request it warms to have bought anything, `consumer.arrival_ts - phantom.arrival_ts < BENCH_PREFETCH_LEAD_MIN_S`. Both stamps are the engine's own `arrival_ts` on the two request rows, so HTTP receipt and chat-template time cancel and what is left is the lead the prefetcher actually bought. `unused` (no consumer ever arrived) counted separately. → `late_prefetches`, `late_prefetch_pct` | request JSONL (this cell only) |
| **Prefetch lead** | `consumer.arrival_ts - phantom.arrival_ts`, mean and min over the phantoms that had a consumer. The distribution behind the late %, and the thing to look at before choosing the threshold. → `prefetch_lead_mean_s`, `prefetch_lead_min_s`, threshold echoed as `prefetch_lead_min_threshold_s` | request JSONL |
| **Max lead / min lead** | The two `/v1/echo` markers as the server stamped them: `max_lead_ts`, when the workflow oracle issued the warm, and `min_lead_ts`, when the graph runtime parsed the tool call naming the node — the earliest a real predictor could know. Recorded as the instants themselves. → `max_lead_ts`, `min_lead_ts` | `server.log` `/v1/echo` markers |
| **Lead window** | `min_lead_ts - max_lead_ts` — how far ahead of a real predictor the oracle knew about this target. Computed from the two markers and referred to nothing else; request timestamps decide only *which* request a marker belongs to. → `lead_window_s` per request, `lead_window_mean_s` per cell | `server.log` `/v1/echo` markers |
| **Useful prefetches** | Count of phantoms that put tokens in HBM which were not already there *and* whose tokens a real request then hit: `credited = min(consumer.num_local_cached_tokens, phantom.num_prompt_tokens) - phantom.num_local_cached_tokens`, floored to whole blocks, `credited > 0`. A phantom whose own prefix was already fully HBM-resident promoted nothing and is never useful, however large its consumer's hit. Absolute count, for the same reason **Total prefetches** is one. → `useful_prefetches` | request JSONL (this cell only) |
| **Useful prefetch %** | `useful_prefetches / total_prefetches`. A phantom that had no consumer, or had not finished when its consumer arrived, cannot be useful; the remainder is the phantom that landed in time, added nothing, and was hit anyway. → `useful_prefetch_pct` | derived |
| **Workflow output tokens** | `Σ num_generation_tokens` over the question's real requests — everything the workflow generated, the per-question total shown in the results header. Pinned replay decodes the recorded count exactly, so this must equal the recorded trace's total; a mismatch means the pin did not hold. | request JSONL (checked against the trace) |
| **Scheduler occupancy** | Over the question's window, from the scheduler timeline: mean and max `num_running_reqs` / `num_waiting_reqs`, total `num_scheduled_reqs` and `num_new_scheduled_reqs` (admissions), plus preemptions. Says whether an arm's latency came from queueing rather than from cache behaviour. | `scheduler_engine0_*.jsonl` |

Measuring TTFT: the runner stamps `t_submit` when it dispatches the question;
the final output is the graph's terminal LLM call (ODR:
`final_report_generation`). Its first-token instant is read from the server
stats, because in pinned mode the client is handed recorded bytes and a
client-side stopwatch would time the replay instead of the model.

Field note: `num_computed_tokens` in the JSONL is the *miss* side — vLLM sets
it to `num_prompt_tokens - num_cached_tokens` (`vllm/v1/metrics/stats.py:300`),
i.e. tokens it had to compute. The tokens already in HBM are
`num_local_cached_tokens`, so that is what the hit rate uses.

Min and max lead, and why these two alone come from the log. Every other
metric here is read from a request row, and deliberately so. These two are not
requests: they are client-side decision instants — the moment the graph runtime
parsed the tool call naming the next node, and the moment the workflow oracle
issued the warm — and nothing on the server would otherwise record them.
Measuring them client-side would mean reconciling two processes' clocks to
within the few hundred milliseconds being measured, which is exactly the
precision two clocks do not have. So the client posts each one to `POST
/v1/echo` and the server logs it against its own clock. That endpoint exists for
this and nothing else.

The reported value is the markers' own. `lead_window_s = min_lead_ts -
max_lead_ts` is the interval the oracle knew about a target before a real
predictor could have, and it needs no third reference point: both ends are
echoes, on one clock, and the span between them is the quantity. Request
timestamps are used only to decide which request a marker belongs to.

Read it as the transferability of the result. An oracle arm whose
`lead_window_mean_s` is near zero is not getting its advantage from lookahead —
the graph runtime knew the same thing at the same time — and its numbers should
survive being driven by a real predictor. A large window is the opposite
warning, and says how much of the arm's benefit evaporates when the oracle goes
away.

Parsing is a grep for `echo: event=<name> agent_id=<id>` with the line's own
timestamp; other echo events (`prefetch_skipped`, say) are ignored, and
`lead_markers` counts what was matched so a silently missing marker stream shows
up as a warning rather than as an absent column. Markers pair with the next
request of the same `agent_id`, latest marker first, so a node that takes
several turns gets the marker belonging to the turn rather than to an earlier
one.

Everything above is measured **per chat completion**, and the cell numbers are
sums over those rows. Each request carries its own `job_id`, `agent_id`,
`langgraph_node` and `seq` (its position on the wire, which is what joins the
same prompt across arms under pinned replay -- `agent_id` alone repeats whenever
a node takes more than one turn), alongside its `ttft_s`, `query_tokens`,
`token_hits`, and the phantoms charged to it: `prefetches`, `late_prefetches`,
`useful`, `credited_tokens`, `prefetch_lead_s`. A phantom is charged to the one
request it warmed, so `useful_prefetches` at cell level is
`sum(useful)` and nothing is derived twice.

That is the level to aggregate *from*, later and however the question needs it.
A cell-level hit rate says a run improved; the per-request rows say which node
did, which is the difference between "13.5% hit rate" and "the three
`researcher_tools` prefills are untouched and the whole gain is in compress".
They land in `requests.csv` at the run root, one row per chat completion across
every cell, and in `metrics.json` under `per_request` for the cell alone.

Late prefetch, and why it is a lead test rather than an overlap test. The
question a phantom has to answer is "did it leave early enough to matter", and
that is `consumer.arrival_ts - phantom.arrival_ts` — the lead. Both stamps live
on the request rows this cell already writes, one per `prefetch_only=true` row
and one per real row, so the metric is a join on two variables and never a parse
of the server log. `arrival_ts` is the engine's, on both sides, so the HTTP
receipt and chat-template tokenisation that sit in front of each cancel out.

There is no physical constant to pin the threshold to, so it is a knob
(`BENCH_PREFETCH_LEAD_MIN_S` in `common.env`, default 0.1 s) and the lead
distribution is
reported beside the percentage. Read them together: a 0% late on a mean lead of
90 ms, against seeded prefills that take on the order of a second, is not a
healthy run — it is a run whose threshold is set below anything it could have
caught.

Late is not the gate on useful, and the two cross. **Useful prefetches** needs
the blocks to have been resident when the consumer arrived,
`consumer.arrival_ts >= phantom.finish_ts`, which is a different question from
whether the lead was worth having. A millisecond-long LMCache promotion can be
late by lead and still land in time to be hit; a seeded prefill needing ~1.2 s
can clear the lead threshold with 200 ms and still not be there. So the residency
check gates `useful` directly and is not reported as a metric of its own.

Post `f47e7521b` (engine-side origination removed): phantoms now
come only from a client calling `POST /v1/agents/prefetch`, so
`agent_prefetch.jsonl` is the complete list and joins 1:1 with the
`prefetch_only=true` rows — the metric no longer has to guess at drains the
client never saw. Two knock-ons:

- `wait` decides whether the metric is meaningful, and both callers set it
  explicitly, so the endpoint's own default never applies. The measurement
  path (`replay_prefetch.py:343,414`) sends `wait=false` — fire-and-forget, so
  the phantom overlaps the producer's decode and the overlap test is the right
  one. The seeding phase (`system_prompt_population.py:352`) sends `wait=true`
  on purpose, to block until L1 is warm; its phantoms are setup, not
  measurement, and the collector excludes them by the question window. The
  runner records the flag per cell and refuses to compute late % if a cell's
  prefetches were issued with `wait=true`, where a phantom cannot be late by
  construction.

- `VLLM_NODE_EVICTION_PREFETCH_DRAIN` and `_INTERVAL_S` no longer exist; drop
  them from the `ours` overlay. Also, plain `/v1/chat/completions` no longer
  registers prefixes — only `/v1/agents/*` does — so the `ours` arm must run
  its system-prompt-population phase or the registry stays empty and the run
  silently records zero prefetches. The collector fails the cell if
  `total_prefetches == 0` on that arm.

Useful prefetch %, and why the phantom's own row is the whole test. A
phantom is useful when it *added* HBM residency that a real request then hit.
Both halves matter, and both are readable off this cell alone.

The phantom's own `num_local_cached_tokens` is the measurement that makes this
work: it is, by definition, how much of that prefix was **already in HBM at the
moment the phantom ran**. Everything past it is what the phantom brought in —
promoted from LMCache (`num_external_cached_tokens`) or prefilled outright
(`num_computed_tokens`). So the credit for a phantom and the consumer that
extends its prefix is

```
credited = min(consumer.num_local_cached_tokens, phantom.num_prompt_tokens)
           - phantom.num_local_cached_tokens
useful   = credited > 0
```

and the two ends of the range are exactly the cases to separate:

- A phantom that was itself a full local hit (`num_local_cached_tokens ==
  num_prompt_tokens`) pulled a prefix out of LMCache that HBM already held.
  `credited <= 0`, not useful — no matter how large the consumer's hit is, that
  hit was going to happen anyway.
- A seeded phantom whose prefix nothing has ever computed has
  `num_local_cached_tokens ~= 0`, prefills, and the consumer hits the whole
  thing. `credited` is the prompt, and the phantom is the only reason those
  blocks exist.

Blocks, not tokens, are the unit APC matches on, so round both sides down to a
multiple of `block_size` (16) before comparing; a `credited` of 1-15 tokens is a
partial trailing block and is noise, not a hit.

Two bookkeeping rules. A phantom with no consumer is `unused`, and one whose
consumer arrived before it finished is `late` — neither can be useful, so
`useful + late + unused <= total_prefetches` and the remainder is the phantom
that landed in time, added nothing, and was hit anyway. And a React warm asks
for `top_k` prefixes, so several phantoms name one consumer: credit the single
phantom with the largest `credited` and count the others as the residual, or
the same hit is banked `top_k` times.

Useful is not the same as helpful, and this is the metric's one blind spot.
Under `prefill_on_miss` a seeded phantom whose prefix LMCache cannot hold runs a
real prefill, registers the blocks in APC, and the request arriving tens of
milliseconds later hits them — maximal `credited`, for work the phantom itself
just did while the request queued behind that same prefill. Measured on a
1-question ODR run: `compress_research` hit 3168 of 3236 prompt tokens for
**-199 ms** of TTFT, and `final_report_generation` hit 2240 of 2244 for
**+17 ms**. Both score fully useful. Only one helped, and neither helped in
proportion to its hit, because the saving is bounded by the phantom's lead over
its consumer (88 ms and 51 ms) and not by what it cached.

So print Useful prefetch % next to the per-request `delta TTFT` it is meant to
explain, and read a high useful % with flat TTFT as the self-prefill case rather
than as a win. Telling the two apart needs the phantom's finish time, which the
server now logs directly: `agent_prefetch_start` / `agent_prefetch_end` (with
`elapsed_ms`) from `vllm/v1/agent_prefetch/submitter.py`, on by default and
switched by `VLLM_PREFETCH_LOG_SPANS`. Nothing else answers it — `kv_hbm_ttft`
skips phantoms on purpose, since
`NodeEvictionController.on_request_finished` gates on
`ttft_chat_completions_only` so the policy's own warming traffic cannot flatter
its own average, and the HTTP access line is the submit rather than the finish
because the endpoint answers `wait=false`.

Scheduler occupancy comes from a file that did not exist before this plan:
`FileStatLogger` now also writes `scheduler_engine<idx>_<ts>.jsonl` into
`VLLM_REQUEST_STATS_DIR`, one sample per engine step carrying
`num_scheduled_reqs`, `num_new_scheduled_reqs`, `num_running_reqs`,
`num_waiting_reqs`, `num_skipped_waiting_reqs`, `num_preempted_reqs`,
`num_finished_reqs` and `kv_cache_usage`. Samples are written on change plus a
heartbeat (`VLLM_REQUEST_STATS_SCHEDULER_INTERVAL_S`, default 1 s) so an idle
stretch stays distinguishable from a stalled logger without one line per step.
Rows are scoped to a question by the same `[t0, t1]` window as everything else.

Timestamps: every artifact is on the wall clock in epoch seconds and is
joinable without guesswork — request rows (`arrival_ts` / `finish_ts`), the
scheduler timeline (`ts`, plus a `clock_anchor` first line pinning
`time.time()` to `time.monotonic()` for that engine process), the eviction
decision log (`ts`, epoch ms), and the server log itself, whose lines now
carry a full date and milliseconds. Second resolution was not enough: an
engine step is ~10 ms, so whole batches shared a timestamp.

Layout:

```
$BENCH_ROOT/<run_id>/
  state.json  results.csv  requests.csv  summary.md  resources.csv  env.redacted
  <arm>/<question_id>/rep<k>/
      metrics.json  server.log  workflow.log
      stats/finished_requests_engine0_*.jsonl
      stats/scheduler_engine0_*.jsonl
      divergence.jsonl  agent_prefetch.jsonl
```

`results.csv` is one row per cell and `requests.csv` one row per chat completion
(with `arm` / `question_id` / `rep` prefixed, so it is groupable on its own);
`summary.md` pivots `results.csv` into one table per question, a row per arm,
with the question's total tokens in the header.

---

## Runbooks — the commands each stack needs, in order

The orchestrator automates exactly these; run them by hand when debugging one
arm. Shared config in [`runbooks/common.env`](runbooks/common.env), secrets in
`~/.bench.env`.

- [`runbooks/README.md`](runbooks/README.md) — index and conventions (tmux
  names, readiness gate, LMCache wipe rule, preflight)
- [`runbooks/00-record.md`](runbooks/00-record.md) — record the trace on the
  baseline stack (run once per question set)
- [`runbooks/01-baseline.md`](runbooks/01-baseline.md) — vLLM + LMCache MP + LRU
- [`runbooks/02-continuum.md`](runbooks/02-continuum.md) — vLLM-Continuum (no
  LMCache server)
- [`runbooks/03-ours.md`](runbooks/03-ours.md) — vLLM + agent prefetch + node
  eviction (Redis, `/v1/agents/*`, seeding required)

