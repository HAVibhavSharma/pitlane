# Ablating the `ours` arm

`ours` is four mechanisms at once. This is how to find out which of them the
numbers come from.

```bash
ARMS="ours_full ours_no_seeds ours_no_pseudo ours_predictor_only" \
  ./launch-batch.sh 50
```

One run id, so `summary.md` pivots them against each other. Modes are arms, not
a runtime flag, because a cell is keyed by `(arm, question, rep)` — two modes
under one arm name would write to the same directory.

## The four switches

| switch | variable | reaches |
|---|---|---|
| `node_eviction` | `VLLM_NODE_EVICTION_POLICY` | the vLLM server. Off is upstream LRU — **and no controller, so no `kv_hbm_ttft` line to diff against** |
| `prefetch` | `KV_EVICTION_DISABLE_PREFETCH` (inverted) | langgraph's live predictor |
| `prompt_seeds` | `ODR_PROMPT_SEEDS` | the compress and final-report prompt seeds |
| `pseudo_dynamic` | `LANGGRAPH_PROMPT_PSEUDO_DYNAMIC` | whether a manufactured segment fills a prefix or blocks it |

`prefetch` is the master switch, not a peer: turning it off makes the workflow
clear `LANGGRAPH_VLLM_AGENT_ENABLE` and `..._BASE_URL`, which takes the other
two down with it. A mode that sets them with prefetch off is rejected at load
rather than run — it would be `no_prefetch` under another name, and its numbers
would be attributed to the wrong thing.

`node_eviction` is the only one on the other side of the wire, and the only one
independent of the rest.

## The modes

| arm | eviction | prefetch | seeds | pseudo | reads as |
|---|---|---|---|---|---|
| `ours_full` | on | on | on | on | the ceiling |
| `ours_no_seeds` | on | on | **off** | on | what the prompt seeds are worth |
| `ours_no_pseudo` | on | on | on | **off** | what filling a prefix past `{date}` is worth |
| `ours_predictor_only` | on | on | **off** | **off** | the live predictor on its own |
| `ours_no_prefetch` | on | **off** | — | — | node eviction on its own |
| `ours_no_eviction` | **off** | on | on | on | everything warmed, LRU underneath |
| `ours_none` | **off** | **off** | — | — | the floor, on the ours build |

`ours_full` should reproduce `ours`. If it does not, something in `arms.toml`
disagrees with `ablation.toml` and every other row is suspect — run it first.

`ours_none` is the floor *on this build*, which is not the same as `baseline`:
same binary, both halves off. The gap between `ours_none` and `baseline` is
what the build costs before any mechanism is switched on, and it should be
nothing. Worth checking once.

## Reading them

Each mode is `ours` in everything but its four switches — same repo, same venv,
same server args, same script — so a change to the stack reaches all seven
without seven edits. That is also the trap: they are seven boots of the same
eleven-hour batch. Pick the two or three that answer the question.

The cheapest useful pair is `ours_full` against `ours_predictor_only`: it
separates the two things ODR adds (the seeds, and filling prefixes) from the
one thing langgraph adds (predicting the next node), in two runs.

Columns that mean different things per mode:

- **`ours_no_eviction` and `ours_none` have no `kv_hbm_ttft` rows.** Upstream
  means no controller. For an eviction off-arm that still reports, set
  `VLLM_NODE_EVICTION_OBSERVE=1` instead — the controller indexes and logs but
  never reorders the free queue.
- **`ours_no_prefetch` and `ours_none` write no `agent_prefetch.jsonl`.** The
  population phase still runs; its rows carry `issuer: population`.
- **`accurate_prefetch_pct` is not comparable between a seeded arm and an
  unseeded one.** It asks whether the node a phantom named is what ran next, and
  a prompt seed deliberately warms a node that runs much later — so every seed
  counts as displaced and `ours_full` scores below `ours_no_seeds` while
  predicting exactly as well. Group `prefetches.csv` by `langgraph_node` and
  read the predictor's nodes (`researcher`, `researcher_tools`, `supervisor`)
  rather than the cell number. Within one seeding setting the column is the
  cleanest signal there is for whether a switch touched the predictor at all.
- **The prompt seeds are speculative prefills, not promotions.** `ours_full`
  spends compute `ours_no_seeds` does not. It lands on idle steps by
  construction, but it is real, and it is why that pair is the one to run if
  you only run one.

## The BENCH_ flags and the modes do not mix

`BENCH_NODE_EVICTION`, `BENCH_PREFETCH` and `BENCH_PROMPT_SEEDS` are ignored on
an ablation arm. A mode states its switches, and a variable set for the whole
box must not quietly overrule one — otherwise `ours_full` with
`BENCH_PROMPT_SEEDS=0` in `.env` would run as `ours_no_seeds` while still
reporting itself as `ours_full`. They still apply to plain `ours`, which is
what they are for: a one-off change without adding a mode.

Leave all three empty in `.env` and pick the arm instead.

## Adding a mode

`pitlane/ablation.toml`, one block, any subset of the four switches. Omitted
switches keep whatever the arm and environment already say. The arm name is
`ours_<block name>`; `pitlane arms` lists it.

Ablation arms are opt-in: they are excluded from the default matrix, so a bare
`pitlane run` still boots three stacks rather than ten.
