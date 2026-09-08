# Arm: baseline — vLLM + LMCache MP + LRU

Replays the recorded trace against the stock stack. This is the control.

```bash
set -a; . ~/.bench.env; . ./common.env; set +a
export ARM=baseline QID=q1 REP=1
export CELL=/disk2/vibhav/bench/$(date +%Y%m%d)/$ARM/$QID/rep$REP
mkdir -p "$CELL/stats"
```

### 1. Preflight — same checks as [00-record.md](00-record.md) step 1

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

### 3. vLLM — baseline build

```bash
tmux kill-session -t vllm 2>/dev/null
tmux new-session -d -s vllm -c "$VLLM_BASELINE_REPO"
tmux send-keys -t vllm 'LMCACHE_MP_FULL_HIT_ONLY=0 VLLM_USE_DEEP_GEMM=0 \
VLLM_REQUEST_STATS_DIR='"$CELL"'/stats \
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
curl -sX POST localhost:8000/v1/kv_metrics/reset    # start the metric epoch
```

### 4. Workflow — pinned replay

```bash
cd "$WORKFLOW_REPO"
unset LANGGRAPH_VLLM_AGENT_ENABLE LANGGRAPH_VLLM_AGENT_BASE_URL
export LANGGRAPH_ABLATION_MODE=full
export ODR_TRACE_MODE=pinned
export ODR_TRACE_ON_MISS=strict
export ODR_TRACE_REPORT=$CELL/divergence.jsonl

date +%s.%N > "$CELL/question_started_ts"
python tests/run_evaluate.py \
  --max-queries 1 --completions-per-query 1 --ablation-mode full \
  2>&1 | tee "$CELL/workflow.log"
```

### 5. Collect, then teardown

```bash
python tests/analyze_divergence.py "$ODR_TRACE_REPORT"   # 0 misses, 0 off-pin
ls "$CELL"/stats/                                        # finished_requests_*, scheduler_*
tmux kill-session -t vllm; tmux kill-session -t lmcache
```
