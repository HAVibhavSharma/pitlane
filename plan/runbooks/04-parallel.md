# Two stacks on one box

Three arms over two GPUs: two run side by side, the third starts on whichever
card frees up first. Roughly halves an eleven-hour batch.

```bash
./launch-parallel.sh 50
```

## Read this first

The two GPUs are separate. Everything else is not: the arms share CPU, memory
bandwidth, the disk under `$LMCACHE_L2_DIR`, and the network to Tavily and
LangSmith.

| Column | Co-tenancy safe? |
|---|---|
| `query_tokens`, `token_hits`, `external_token_hits`, `kv_hit_rate` | yes — counted per request, on this stack's own server |
| `total_prefetches`, `useful_prefetches`, prefetch leads | yes |
| `trace_misses`, `off_pin_requests` | yes |
| `ttft_s`, `wall_clock_s`, `queued_s`, `tool_seconds`, `sched_*` | **no** — measured against a neighbour |

`queued_s` is the one to watch. Question 35 of the n50 batch died because
continuum's slowest `summarize_webpage` came within 0.3 s of the 240 s timeout
and fell back to the raw page, which no longer matched the trace; that arm's
`queued_s` was 1,084 s against baseline's 667 s. A co-tenant pushes that number
the wrong way. `SUMMARIZATION_TIMEOUT_S=600` in `common.env` buys back the
margin, but the failure mode is worth remembering before running two arms hot.

So: use this for token-level results, for functional coverage, and to find
failures early at half the wall clock. Re-run sequentially with
`./launch-batch.sh` for any latency number you intend to publish.

Every cell carries the answer in its own row. `host_share` is `solo`,
`shared` (a neighbour for the whole cell) or `mixed` (the third arm, which
starts beside the survivor and ends alone) — and the launcher sets it, because
the launcher is the only thing that knows. `launch-batch.sh` writes `solo`,
this one writes `shared` and `mixed`, both as an overlay passed last so it wins
over the box's `.env`. There is nothing to remember to change between runs.

## What makes the two stacks independent

Four things used to be constants and are now per stack, in
[`stack-a.env`](stack-a.env) and [`stack-b.env`](stack-b.env):

| | stack A | stack B |
|---|---|---|
| `BENCH_GPU` → `CUDA_VISIBLE_DEVICES` | 0 | 1 |
| `BENCH_PORT` (vLLM) | 8000 | 8001 |
| `BENCH_LMCACHE_PORT` | 10903 | 10904 |
| `BENCH_INSTANCE` → tmux sessions | `vllm`, `lmcache` | `vllm-b`, `lmcache-b` |
| `KV_FORECAST_REDIS_URL` | db 0 | db 1 |

The tmux names matter more than they look: `start_server` kills the session it
is about to use, so two stacks sharing the name `vllm` means the second launch
tears down the first arm's server — hours in, with no error anywhere.

The arms no longer hard-code `localhost:8000` either; `arms.toml` says
`{port}`, which resolves to the stack's own. An arm pointed at the wrong port
does not fail, it measures the other arm's server.

Attach to either with the usual command plus the suffix:

```bash
tmux attach -t vllm      # stack A
tmux attach -t vllm-b    # stack B
```

## RAM is the constraint, not VRAM

LMCache's L1 pool is host memory and each stack allocates its own, so the
single-run default of 200 GB becomes 400 GB. Both overlays ship at
`BENCH_LMCACHE_L1_GB=100`, and that is not a squeeze — the n50 batch's own
`resources.csv` says the ceiling was never approached:

| arm | host RAM consumed over the batch |
|---|---|
| baseline | 4.6 GB |
| continuum | 5.3 GB |
| ours | 17.9 GB |

The pool starts at 20 GB, so peak usage was ~25 GB for baseline and continuum
and ~38 GB for ours, against a 200 GB ceiling, on a 503 GB box. `ours` caches
most, which is what you would expect from the arm with prefetch.

Keep the two **equal**, and do not go far below 100. While no arm reaches its
ceiling, L1 size changes nothing measurable; the moment one does, LRU starts
firing and the ceiling stops being a resource setting and becomes an
experimental variable — in the arm whose eviction policy is the thing under
test.

```bash
free -g                 # total, and what is already in use
nvidia-smi topo -m      # which CPU socket each GPU hangs off
numactl -H              # whether the box has two NUMA nodes to bind to
```

The pool grows lazily from 20 GB, so both preflights pass at launch and the
squeeze arrives hours later. `BENCH_ABORT_FREE_RAM_GB` (32 by default) is the
backstop and it is host-wide: whichever stack is running when free RAM crosses
it is the one that aborts, not necessarily the one that took the memory.

If `numactl -H` shows two nodes, bind each stack to the socket its GPU hangs
off. It removes most of the remaining interference:

```bash
numactl --cpunodebind=0 --membind=0 ./launch-parallel.sh 50   # not the whole
                                                              # script -- see below
```

Bind the `pitlane run` calls individually rather than the launcher, since the
two halves belong on different nodes; `run_arm` in `launch-parallel.sh` is the
one line to prefix.

## Resume

Both processes share one run id and therefore one ledger, which is why writes
to it are locked — unlocked, eight concurrent writers lost 37 of 48 entries in
testing, and a lost entry means `--resume` re-runs a cell that is already
complete on disk. Resume works exactly as it does for a sequential run:

```bash
RESUME=1 RUN_ID=batch50_20260917_101500 ./launch-parallel.sh 50
```

## Per-arm logs

The three arms write to `$BENCH_ROOT/$RUN_ID/.pitlane/launch-<arm>.log` rather
than the terminal, since three interleaved stdouts are unreadable.

```bash
tail -f "$BENCH_ROOT/$RUN_ID/.pitlane/launch-baseline.log"
```
