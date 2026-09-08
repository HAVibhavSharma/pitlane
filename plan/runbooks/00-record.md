# Record phase — build the trace (baseline stack)

Run once per question set. Produces `$ODR_TRACE_PATH`, the workload definition
every eval arm then replays. Record **one** completion per question;
repetitions belong in the measurement phase.

Skip if the trace already covers the questions you want.

```bash
set -a; . ~/.bench.env; . ./common.env; set +a
```

### 1. Preflight

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv   # GPU 0 idle?
free -g | awk '/Mem:/ {print "free RAM (GB):", $7}'                     # ≥ 200 + margin
ss -ltn | grep -q ':8000 ' && echo "PORT 8000 BUSY" || echo "port 8000 free"
```

### 2. LMCache — stop, wipe, start

```bash
tmux kill-session -t lmcache 2>/dev/null
rm -rf "$LMCACHE_L2_DIR"

tmux new-session -d -s lmcache
tmux send-keys -t lmcache 'LMCACHE_LOG_KV_HASH=1 lmcache server \
  --l1-size-gb 200 \
  --eviction-policy LRU \
  --chunk-size 16 \
  --host 0.0.0.0 \
  --port 10903 \
  --l2-adapter "{\"type\":\"fs\",\"base_path\":\"'"$LMCACHE_L2_DIR"'\"}"' Enter
```

### 3. vLLM — baseline build

```bash
tmux kill-session -t vllm 2>/dev/null
tmux new-session -d -s vllm -c "$VLLM_BASELINE_REPO"
tmux send-keys -t vllm 'LMCACHE_MP_FULL_HIT_ONLY=0 VLLM_USE_DEEP_GEMM=0 \
VLLM_REQUEST_STATS_DIR=./vllm_request_stats/ \
vllm serve '"$MODEL_NAME"' --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --hf-overrides "{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}}" \
  --gpu-memory-utilization 0.95 \
  --block-size 16 \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\"}" \
  --enable-prefix-caching \
  --seed 0 \
  --enable-prompt-tokens-details > test.log 2>&1' Enter

until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done; echo READY
```

### 4. Workflow — record

```bash
cd "$WORKFLOW_REPO"
unset LANGGRAPH_VLLM_AGENT_ENABLE LANGGRAPH_VLLM_AGENT_BASE_URL
export LANGGRAPH_ABLATION_MODE=full
export ODR_TRACE_MODE=record
export ODR_TRACE_PATH="$TRACE_DIR/drb_v1_3.jsonl"       # new file; append is refused

python tests/run_evaluate.py \
  --max-queries 10 --completions-per-query 1 \
  --ablation-mode baseline --skip-cold-phase --no-kv-metrics-reset
```

### 5. Verify, then freeze

```bash
wc -l "$ODR_TRACE_PATH"
python - <<'PY'
import json, os, collections
p = os.environ["ODR_TRACE_PATH"]
jobs = collections.Counter(json.loads(l)["job_id"] for l in open(p))
print(len(jobs), "jobs:", dict(jobs))
PY
```

Freeze `$TAVILY_CACHE_DIR` from here on — a cache miss mid-study fetches live
and injects new content into one arm only.

### 6. Teardown

```bash
tmux kill-session -t vllm; tmux kill-session -t lmcache
```
