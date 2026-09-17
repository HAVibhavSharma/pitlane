# Arm: ours — vLLM + agent prefetch + node eviction

The only arm that uses `/v1/agents/*`, the node-eviction policy and Redis.

```bash
set -a; . ~/.bench.env; . ./common.env; set +a
export ARM=ours QID=q1 REP=1
export CELL=/disk2/vibhav/bench/$(date +%Y%m%d)/$ARM/$QID/rep$REP
mkdir -p "$CELL/stats" "$VLLM_LOG_DIR/debug" "$VLLM_LOG_DIR/kv-prediction-transitions"
```

### 1. Preflight

Same checks as [00-record.md](00-record.md) step 1, plus Redis:

```bash
redis-cli -u redis://127.0.0.1:6379/0 ping        # expect PONG
```

### 2. LMCache — stop, wipe, start

```bash
tmux kill-session -t lmcache 2>/dev/null
rm -rf "$LMCACHE_L2_DIR"
tmux new-session -d -s lmcache
tmux send-keys -t lmcache 'LMCACHE_LOG_KV_HASH=1 lmcache server \
  --l1-size-gb 200 --eviction-policy LRU --chunk-size 16 \
  --host 0.0.0.0 --port 10903 \
  --l2-adapter "{\"type\":\"fs\",\"base_path\":\"'"$LMCACHE_L2_DIR"'\"}"' Enter
```

### 3. vLLM — our build

`VLLM_NODE_EVICTION_PREFETCH_DRAIN` and `_INTERVAL_S` are **gone** as of
`f47e7521b` (engine-side prefetch origination removed); do not export them.

```bash
tmux kill-session -t vllm 2>/dev/null
tmux new-session -d -s vllm -c "$VLLM_OURS_REPO"
tmux send-keys -t vllm 'export VLLM_NODE_EVICTION_POLICY=1 \
  VLLM_NODE_EVICTION_REDIS_URL=redis://127.0.0.1:6379/0 \
  VLLM_NODE_EVICTION_CONFIG='"$VLLM_OURS_REPO"'/node_eviction.json \
  VLLM_NODE_EVICTION_DECISION_LOG='"$CELL"'/evictions.jsonl \
  VLLM_REQUEST_STATS_DIR='"$CELL"'/stats' Enter
tmux send-keys -t vllm 'LMCACHE_MP_FULL_HIT_ONLY=0 VLLM_USE_DEEP_GEMM=0 \
. "$VLLM_OURS_VENV/bin/activate"
vllm serve '"$MODEL_NAME"' --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --hf-overrides "{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}}" \
  --gpu-memory-utilization 0.95 \
  --block-size 16 \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\"}" \
  --enable-prefix-caching \
  --seed 0 \
  --enable-prompt-tokens-details > '"$CELL"'/server.log 2>&1' Enter

until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done; echo READY
```

### 4. Workflow — pinned replay with the agent path on

```bash
cd "$WORKFLOW_REPO"
export LANGGRAPH_VLLM_AGENT_ENABLE=1
export LANGGRAPH_VLLM_AGENT_BASE_URL=http://localhost:8000/v1/agents
export LANGGRAPH_VLLM_AGENT_MODEL=$MODEL_NAME
export LANGGRAPH_VLLM_AGENT_NAMESPACE=langgraph
export LANGGRAPH_VLLM_AGENT_WARMUP=0
export LANGGRAPH_VLLM_AGENT_TIMEOUT_SECONDS=30
export LANGGRAPH_VLLM_AGENT_LOG_PATH=$CELL/agent_prefetch.jsonl

export KV_PREDICTION_HORIZON=5
export KV_EVICTION_DISABLE_PREFETCH=0
export KV_FORECAST_REDIS_URL=redis://127.0.0.1:6379/0
export KV_PREDICTION_TRANSITION_STORE=$VLLM_LOG_DIR/kv-prediction-transitions/transitions.json

export ODR_TRACE_MODE=pinned
export ODR_TRACE_ON_MISS=strict
export ODR_TRACE_REPORT=$CELL/divergence.jsonl

date +%s.%N > "$CELL/question_started_ts"
"$WORKFLOW_VENV/bin/python" tests/run_evaluate_node_eviction.py \
  --max-queries 1 --completions-per-query 1 \
  2>&1 | tee "$CELL/workflow.log"
```

Chat calls route to `/v1/agents/chat/completions` automatically: with
`LANGGRAPH_VLLM_AGENT_ENABLE=1` the request carries an `agent_id`, and ODR's
`get_model_config` then appends `/agents` to `OPENAI_BASE_URL`. That endpoint
records each served prompt in the prefix registry, which is what a later
prefetch warms — plain `/v1/chat/completions` stopped registering at
`f47e7521b`, so without it the registry holds only the seeded system prompts
(~160 tokens) and no phantom can touch the accumulated conversation. Costs one
extra tokenization per request, paid only by this arm. Set
`ODR_AGENT_CHAT_ROUTE=0` to go back to the plain endpoint.

The system-prompt-population phase runs by default and is **required** here:
since `f47e7521b`, plain `/v1/chat/completions` no longer registers prefixes —
only `/v1/agents/*` does — so skipping it leaves the registry empty and every
prefetch silently no-ops. Do not pass `--skip-system-prompt-population` unless
a previous invocation seeded the same server.

### 4b. Where prefetches come from

The replay oracle is gone. It read the recorded response before the live
request went out, resolved the upcoming tool calls through the graph's
transition rules and warmed the successor with the producer's whole
prefill+decode as lead time — an upper bound on what perfect, perfectly-early
prediction could buy, which is why it only ever belonged in its own arm. It
and its five routes (`replay_prefetch`, `_nested`, `_completion`,
`compress_research_seed`, `final_report_seed`) were removed from
open_deep_research, along with `prefetch_timing`. No
`job_*.replay_prefetch.jsonl` is written any more, and none of
`ODR_REPLAY_PREFETCH*`, `ODR_COMPRESS_SEED`, `ODR_FINAL_REPORT_SEED`,
`ODR_PREFETCH_JIT` or `ODR_REACT_PREFETCH_TOP_K` does anything.

Two sources remain, and `agent_prefetch.jsonl` holds both:

| source | fires | lead |
|---|---|---|
| system prompt population | once, before the measured phase | n/a — seeding, not prediction |
| langgraph's live predictor | when a tool-call name appears in the producer's streaming deltas | small, and the point of the arm |

The live predictor's lead is the number to watch. A tool call is emitted near
the *end* of a generation — the model writes content first — and nothing runs
between the producer's eos and the predicted node, since `supervisor_tools`
invokes the researcher subgraph directly. `prefetch_lead_mean_s` and the late
percentage in `results.csv` are measuring exactly that squeeze; under the
oracle they were measuring the recording instead.

### The prompt seeds

Two prompts in this graph share no usable prefix with anything the server has
already computed, so both prefill from scratch on the critical path:

- **`final_report_generation`** — 11k-70k chars, of which a measured 132 are
  shareable with anything else, because `{research_brief}` sits 132 characters
  into the template and shifts everything after it.
- **`compress_research`** — opens with its own system block and then copies the
  researcher conversation verbatim, so it shares no prefix with the chain it
  copies: zero shared leading messages on 33/33 compress calls in a recorded
  run.

`prompt_seeds.py` rebuilds both from requests that have **already gone out**
and prefills them during a gap. It replaces the oracle's two seed routes and
reads no recording, in any trace mode — asserted structurally by
`tests/check_prompt_seeds.py`, which also checks the final-report seed is
byte-identical to the prompt the node goes on to send.

| seed | built from | fires at |
|---|---|---|
| `final_report_seed` | the supervisor request going out (brief, findings) + the buffer lifted from `write_research_brief` | every supervisor turn whose findings changed |
| `compress_research_seed` | the researcher request going out, message 0 swapped for compress's system block | the turn that will hit the iteration cap |
| `compress_research_seed_extended` | that, plus the reply just received | the same turn's response, if it carries tool calls |

The compress seed is short by one assistant message on the request pass —
it cannot contain a reply that has not been generated. The extended pass adds
it, and has the turn's tool phase to prefill in. What is never covered is the
final turn's tool results, which exist only at the instant
`compress_research` is dispatched.

Only the iteration-cap exit is seeded. The other two exits (`no tool calls`,
`ResearchComplete`) are properties of the reply, and by the time they are known
`researcher_tools` routes straight to compression with no gap to prefill in.

`BENCH_PROMPT_SEEDS=0` turns both off; `ODR_COMPRESS_SEED=0` and
`ODR_FINAL_REPORT_SEED=0` turn off one each. Rows land in
`job_*.prompt_seeds.jsonl`.

These are speculative **prefills**, not promotions — the one thing here whose
wrong guesses cost compute rather than a POST. Run it as its own arm.

`prefill_on_miss` still applies to what the population phase sends: a phantom
whose prefix LMCache does not hold prefills instead of aborting. The scheduler
admits such a phantom only into a step with **no real request running**, and
drops it if no idle step arrives, so the cost lands on idle GPU time and HBM
pressure rather than another request's token budget.

```bash
VLLM_PREFETCH_PREFILL_MAX_RUNNING=0       # real requests tolerated (0 = strictly idle)
VLLM_PREFETCH_PREFILL_DEFER_TIMEOUT_S=30  # then finish it without prefilling
```

Check the seeds landed and the prefills found a gap:

```bash
jq -r '.event' "$CELL/agent_prefetch.jsonl" | sort | uniq -c

# did the deferred prefills ever get an idle step?
python - <<'EOS'
import json, glob, os
for path in glob.glob(os.path.join(os.environ["CELL"], "stats", "scheduler_engine*.jsonl")):
    last = [json.loads(l) for l in open(path)][-1]
    print(path.split("/")[-1],
          "deferrals:", last.get("num_prefetch_prefill_deferrals"),
          "expired:", last.get("num_prefetch_prefill_expired"))
EOS
```

`expired` tracking `deferrals` means the run never had an idle step, so no
phantom prefill ever ran and the warms fell back to promotion only.

### 5. Collect, then teardown

```bash
python tests/analyze_divergence.py "$ODR_TRACE_REPORT"
wc -l "$CELL/agent_prefetch.jsonl"    # 0 prefetches ⇒ the cell is invalid
grep -c prefetch_only "$CELL"/stats/finished_requests_engine0_*.jsonl
tmux kill-session -t vllm; tmux kill-session -t lmcache
```
