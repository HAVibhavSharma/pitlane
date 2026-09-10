# Arm: continuum — vLLM-Continuum

Differs from the other two arms: **no LMCache server**. Continuum drives
LMCache in-process through `LMCACHE_CONFIG_FILE` + `LMCacheConnectorV1`, so
there is nothing to start on port 10903 and nothing to wipe between runs
beyond whatever that config points at.

```bash
set -a; . ~/.bench.env; . ./common.env; set +a
export ARM=continuum QID=q1 REP=1
export CELL=/disk2/vibhav/bench/$(date +%Y%m%d)/$ARM/$QID/rep$REP
mkdir -p "$CELL/stats"
```

### 1. Preflight — same checks as [00-record.md](00-record.md) step 1

`--log-config-file` timestamps uvicorn's access lines. This build has no
built-in timestamped access log (ours and baseline get it from
`vllm/logging_utils/access_log_filter.py`), and an access line with no clock
cannot be placed against the engine timeline.

Plus: confirm `$VLLM_CONTINUUM_REPO/lmcache-config.yaml` exists and note the
store path it declares; clear it if the run must start cold.

### 2. vLLM — continuum build (started from its own repo dir)

```bash
tmux kill-session -t lmcache 2>/dev/null      # must NOT be running for this arm
tmux kill-session -t vllm 2>/dev/null
tmux new-session -d -s vllm -c "$VLLM_CONTINUUM_REPO"
tmux send-keys -t vllm 'LMCACHE_CONFIG_FILE=$(pwd)/lmcache-config.yaml \
VLLM_USE_DEEP_GEMM=0 VLLM_REQUEST_STATS_DIR='"$CELL"'/stats \
. "$VLLM_CONTINUUM_VENV/bin/activate"
vllm serve '"$MODEL_NAME"' --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --hf-overrides "{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}}" \
  --gpu-memory-utilization 0.95 \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}" \
  --block-size 16 \
  --log-config-file '"$PITLANE"'/plan/runbooks/uvicorn-log-config.json \
  --enable-prefix-caching \
  --seed 0 \
  --enable-prompt-tokens-details > '"$CELL"'/server.log 2>&1' Enter

until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done; echo READY
curl -sX POST localhost:8000/v1/kv_metrics/reset
```

### 3. Workflow — pinned replay

```bash
cd "$WORKFLOW_REPO"
unset LANGGRAPH_VLLM_AGENT_ENABLE LANGGRAPH_VLLM_AGENT_BASE_URL
export ODR_TRACE_MODE=pinned
export ODR_TRACE_ON_MISS=strict
export ODR_TRACE_REPORT=$CELL/divergence.jsonl

date +%s.%N > "$CELL/question_started_ts"
"$WORKFLOW_VENV/bin/python" tests/run_evaluate.py \
  --max-queries 1 --completions-per-query 1 \
  --ablation-mode baseline --no-kv-metrics-reset \
  2>&1 | tee "$CELL/workflow.log"
```

### 4. Collect, then teardown

```bash
python tests/analyze_divergence.py "$ODR_TRACE_REPORT"
tmux kill-session -t vllm
```
