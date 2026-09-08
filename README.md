# pitlane

Stop the car, swap the engine, run a timed lap on a fixed circuit.

pitlane brings up a vLLM serving stack, waits for it to actually answer, runs
**one** LangGraph workflow question against it under a pinned trace, tears it
down, and writes the metrics. The workload is byte-identical across stacks, so
what differs in the numbers is the serving layer and nothing else.

Stacks it swaps between (`pitlane arms`):

| Arm | Stack |
|---|---|
| `record` | baseline, recording the trace that everything else replays |
| `baseline` | vLLM + LMCache MP connector + LRU |
| `continuum` | vLLM-Continuum (in-process LMCache, no separate server) |
| `ours` | vLLM + agent prefetch + node eviction |

Workflows: `open_deep_research` and `swe-agent`. The adapter is chosen by
sniffing which flags the repo's `tests/run_evaluate.py` accepts, so a new
workflow repo needs no config.

## Install

Nothing to install — stdlib only, Python 3.11+ (for `tomllib`).

```bash
git clone <this> && cd pitlane
python3 -m pitlane.cli arms
```

Optionally `pip install -e .` to get a `pitlane` entry point.

## Configure

Two files. Secrets in `~/.bench.env`, everything else in
[`plan/runbooks/common.env`](plan/runbooks/common.env):

```bash
# ~/.bench.env
TAVILY_API_KEY=...
LANGSMITH_API_KEY=...
```

Both are plain shell env files, so the same files drive the by-hand
[runbooks](plan/runbooks/README.md).

## Use

```bash
pitlane preflight --arm ours        # GPU 0 idle? RAM? ports? redis? repos?
pitlane record --questions 10       # build the trace (once per question set)
pitlane run --arms baseline,continuum,ours --questions q1,q2 --reps 3
pitlane report /disk2/vibhav/bench/20260908_101500
```

`--dry-run` walks the whole matrix without starting anything, printing the
exact command each arm would run. Use it after editing `arms.toml`.

Run pitlane itself inside tmux. The servers live in their own detached
sessions (`lmcache`, `vllm`) and survive a dropped connection, but the
workflow is a child process of pitlane and does not.

## What it guarantees

- **LMCache wipe and restart are one operation**, in that order. Wiping the L2
  dir under a live server leaves its in-memory L1 index pointing at deleted
  files.
- **A server is up because it answered**, never because a sleep expired. The
  gate polls `/v1/models` and fails fast on a fatal line in the server log
  instead of burning the whole timeout.
- **One question per boot** by default, with `/v1/kv_metrics/reset` before the
  workflow starts. Reusing a boot marks every later cell `warm`, and the
  reporter keeps warm and cold cells apart.
- **Arms cannot inherit each other's flags.** An arm blanks the env keys it
  must not have (`LANGGRAPH_VLLM_AGENT_*` on everything but `ours`), and those
  are removed from the child environment rather than merely unset in a shell.
- **Interleaved, not blocked**: A, B, C, A, B, C… so drift in machine state
  spreads across arms instead of landing on whichever ran last.

## Artifacts

```
$BENCH_ROOT/<run_id>/
  results.csv  summary.md
  <arm>/<question>/rep<k>/
      metrics.json  command.txt  question_started_ts
      server.log  workflow.log  divergence.jsonl  agent_prefetch.jsonl
      stats/finished_requests_engine0_*.jsonl
      stats/scheduler_engine0_*.jsonl
```

Metrics per cell: TTFT (submit → first token of the final output), KV hit rate
(`Σ num_local_cached_tokens / Σ num_prompt_tokens`), query tokens, token hits,
workflow output tokens, total and late prefetches, scheduler occupancy. See
`pitlane/metrics.py` for the exact definitions and
[`plan/01-design.md`](plan/01-design.md) for
why each is defined that way.

## Documentation

[`plan/`](plan/README.md) — the design (metric definitions, component
responsibilities, execution order) and the per-stack runbooks.

## Tests

```bash
python3 tests/test_metrics.py     # collector arithmetic, synthetic artifacts
```
