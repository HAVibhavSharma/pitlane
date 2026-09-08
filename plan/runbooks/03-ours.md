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
python tests/run_evaluate_node_eviction.py \
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

### 4b. Prefetch routes and their switches

Three routes fire in this arm; each writes its own `event` into
`job_*.replay_prefetch.jsonl`, so they can be separated after the run and one
can be turned off without touching the others.

| event | fires | seed |
|---|---|---|
| `replay_prefetch` | before the producing call | none |
| `replay_prefetch_nested` | before the producing call | none |
| `replay_prefetch_completion` | the instant the response returns | next turn's exact messages |

```bash
ODR_REPLAY_PREFETCH=0                # the two up-front routes
ODR_REPLAY_PREFETCH_ON_COMPLETION=0  # the completion route
ODR_REPLAY_PREFETCH_SEED_MESSAGES=0  # keep the completion route, drop its seed
```

All three send `prefill_on_miss`, which the server defaults on: a phantom whose
prefix LMCache does not hold prefills it instead of aborting. The scheduler
only admits such a phantom into a step with **no real request running**, and
drops it if no idle step arrives — so the cost lands on idle GPU time and HBM
pressure, never on another request's token budget.

```bash
VLLM_PREFETCH_PREFILL_MAX_RUNNING=0       # real requests tolerated (0 = strictly idle)
VLLM_PREFETCH_PREFILL_DEFER_TIMEOUT_S=30  # then finish it without prefilling
```

Check the seeds landed and the prefills found a gap:

```bash
python - <<'EOS'
import json, os, glob
for path in glob.glob(os.path.join(os.environ["CELL"], "*.replay_prefetch.jsonl")):
    rows = [json.loads(l) for l in open(path)]
    for event in sorted({r["event"] for r in rows}):
        sel = [r for r in rows if r["event"] == event]
        took = sum(1 for r in sel if r.get("seeded_from_messages"))
        pre = sum(1 for r in sel if r.get("prefill_on_miss"))
        print(f"{event:30s} {len(sel):4d} rows, {took:4d} seeds accepted, "
              f"{pre:4d} allowed to prefill")
EOS

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

`seeds accepted` at 0 on rows that sent one means the server predates
`messages=` on `/v1/agents/prefetch` — it warmed the registry's shorter prefix
instead, silently. `expired` tracking `deferrals` means the run never had an
idle step, so no phantom prefill ever ran and the warms fell back to promotion
only.

### 5. Collect, then teardown

```bash
python tests/analyze_divergence.py "$ODR_TRACE_REPORT"
wc -l "$CELL/agent_prefetch.jsonl"    # 0 prefetches ⇒ the cell is invalid
grep -c prefetch_only "$CELL"/stats/finished_requests_engine0_*.jsonl
tmux kill-session -t vllm; tmux kill-session -t lmcache
```
