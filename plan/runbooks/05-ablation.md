# Ablating the `ours` arm

`ours` is four mechanisms at once. This is how to find out which of them the
numbers come from — five arms, each one switch thinner than the last.

```bash
ARMS="ours_full ours_no_seeds ours_predictor_only ours_no_prefetch ours_none" \
  ./launch-batch.sh 50
```

One run id, so `summary.md` pivots them against each other. Modes are arms, not
a runtime flag, because a cell is keyed by `(arm, question, rep)` — two modes
under one arm name would write to the same directory.

## The five switches

| switch | variable | reaches |
|---|---|---|
| `population` | `--skip-system-prompt-population` (a flag, not a variable) | whether the prefix registry is seeded at all. Only settable to `false` on a mode that also turns `prefetch` off — see below |
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

## The five modes

A ladder, not a grid: each rung removes exactly one switch from the rung above
it, which is what makes a difference between two of them attributable to that
switch.

| arm | eviction | prefetch | seeds | pseudo | reads as |
|---|---|---|---|---|---|
| `ours_full` | on | on | on | on | the ceiling |
| `ours_no_seeds` | on | on | **off** | on | full, minus the speculative prefills |
| `ours_predictor_only` | on | on | off | **off** | prefetch and eviction, nothing on top |
| `ours_no_prefetch` | on | **off** | — | — | eviction alone |
| `ours_none` | **off** | off | — | — | the floor: upstream LRU, nothing warmed |

One switch per step is what makes a difference attributable:

- `ours_full` − `ours_no_seeds` is **what the prompt seeds are worth**: the
  compress and final-report prefills, the only mechanism here whose wrong
  guesses cost compute rather than a POST.
- `ours_no_seeds` − `ours_predictor_only` is **what filling a prefix is
  worth**: how much a warm gains by running past its first manufactured
  segment instead of stopping at one. `{date}` sits ~5% into every system
  prompt, so this is most of a warm's length.
- `ours_predictor_only` − `ours_no_prefetch` is **what langgraph adds**:
  predicting the next node at all.
- `ours_no_prefetch` − `ours_none` is **what node eviction adds**, on one
  binary with everything else held still. This is the gap the policy has to
  earn, and it is the reason `ours_none` exists rather than using `baseline`
  for it.
- `ours_none` − `baseline` is **what the build costs** before any mechanism is
  switched on, and it should be nothing. Different binaries, so it is the one
  gap that is not a clean attribution — worth confirming once rather than
  every batch.

The first two together are what open_deep_research contributes; run
`ours_full` against `ours_predictor_only` alone if the schedule will not take
five, and `ours_no_seeds` is the rung to drop first.

`ours_full` should reproduce `ours`. If it does not, something in `arms.toml`
disagrees with `ablation.toml` and every other row is suspect — run it first.

The population phase is a fifth switch and is not a rung: on wherever prefetch
is, because there it is a precondition rather than a mechanism, and off in
`ours_no_prefetch` because there it is the only warming left. They are
separable and a mode that splits them is two lines in `ablation.toml`, but each
one is another eleven hours, so split them only once a result says the layer
matters.

## Reading them

Each mode is `ours` in everything but its switches — same repo, same venv, same
server args, same script — so a change to the stack reaches all five without
five edits. That is also the cost: every mode is a full re-run of the whole
batch, so five modes is five batches, and the ladder is the set where each run
answers a question the others cannot.

Columns that mean different things per mode:

- **`ours_no_prefetch` and `ours_none` warm nothing at all.** No
  `agent_prefetch.jsonl`, and no population phase either: they are the two
  modes that pass `--skip-system-prompt-population`. Everywhere else that phase is mandatory —
  the registry is written only by `/v1/agent_chat` and `/v1/agents/*`, so an
  unseeded registry makes every warm a silent no-op, and a mode that skips it
  with prefetch on is rejected at load. Here there is no lookup to leave empty,
  and the phase would prefill every node's system prompt into HBM, which is
  warming ahead of a request by another name. So `population_prefetches` is 0
  in this arm and non-zero in the other two, where the collector holds those
  rows out of every prefetch column.
- **`ours_none` has no `kv_hbm_ttft` rows; the other four do.** That line comes
  from the eviction controller, and switching the policy off means there is no
  controller. So the floor cannot be compared to the rungs above it on the
  latency breakdown, only on the request rows every arm writes. If you want the
  breakdown from an eviction-off arm, leave `node_eviction` alone and set
  `VLLM_NODE_EVICTION_OBSERVE=1` instead — the controller then indexes and logs
  but never reorders the free queue. That is a different arm from this one:
  `ours_none` is meant to cost what the build costs, and an observing
  controller is not free.
- **`accurate_prefetch_pct` is not comparable between a seeded arm and an
  unseeded one.** It asks whether the node a phantom named is what ran next, and
  a prompt seed deliberately warms a node that runs much later — so every seed
  counts as displaced and `ours_full` scores below `ours_no_seeds` while
  predicting exactly as well. `ours_no_seeds` and the two rungs below it are
  comparable on this column; `ours_full` is not comparable to any of them. Group `prefetches.csv` by `langgraph_node` and
  read the predictor's nodes (`researcher`, `researcher_tools`, `supervisor`)
  rather than the cell number. Within one seeding setting the column is the
  cleanest signal there is for whether a switch touched the predictor at all.
- **The prompt seeds are speculative prefills, not promotions.** `ours_full`
  spends compute `ours_no_seeds` does not. It lands on idle steps by
  construction, but it is real, and that pair is now a single-switch
  difference rather than two.

## The BENCH_ flags and the modes do not mix

`BENCH_NODE_EVICTION`, `BENCH_PREFETCH` and `BENCH_PROMPT_SEEDS` are ignored on
an ablation arm. A mode states its switches, and a variable set for the whole
box must not quietly overrule one — otherwise `ours_full` with
`BENCH_PROMPT_SEEDS=0` in `.env` would run with half of `ours_predictor_only`'s
switches while still reporting itself as `ours_full`. They still apply to plain `ours`, which is
what they are for: a one-off change without adding a mode.

Leave all three empty in `.env` and pick the arm instead.

## Adding a mode

`pitlane/ablation.toml`, one block, any subset of the five switches. Omitted
switches keep whatever the arm and environment already say. The arm name is
`ours_<block name>`; `pitlane arms` lists it.

Ablation arms are opt-in: they are excluded from the default matrix, so a bare
`pitlane run` still boots the three measurement stacks and none of these.
