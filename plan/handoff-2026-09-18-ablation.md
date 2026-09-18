# Handoff — 2026-09-18

Session focus: **make the `ours` arm decomposable.** It was four mechanisms
fused together with no way to switch one off, and one of the four was an
oracle reading the future out of the trace. That one is gone; the other three
now have flags and a set of ablation arms built from them.

Also in here: a Tavily key pool, two stacks on two GPUs, and a langgraph date
fix that quietly changed what an existing flag is worth.

Harness background is in [01-design.md](01-design.md) and
[02-running.md](02-running.md). Per-task detail is in the runbooks, listed
below — don't re-derive any of it here.

Supersedes [handoff-2026-09-15-tavily-pinned.md](handoff-2026-09-15-tavily-pinned.md),
whose "State" section is now wrong.

## Read first

**Nothing in this session has touched a GPU.** Every check was a fake client,
a stub transport, or a config assertion. The first real exercise is
[ABLATION-SMOKE-TEST.md](ABLATION-SMOKE-TEST.md), one question per mode,
written to be run by an agent on the box.

**`ours` is not comparable to the previous n50 numbers.** Three separate
reasons, any one of which is sufficient: the oracle was removed, the prompt
seeds replaced two of its routes, and langgraph's prefix resolver stopped
manufacturing the wrong date. Re-baseline before comparing anything.

## What changed

Commit messages carry the reasoning; read them rather than re-explaining.

| Repo | Commit | What |
|---|---|---|
| open_deep_research | `2df719c` | Tavily key pool — one key per account, round-robin, refused keys dropped mid-run |
| open_deep_research | `fec2bbf` | env.template: the warmed date is no longer a wrong guess |
| open_deep_research | `60fc8d9` | replay oracle removed; `prompt_seeds.py` replaces its two seed routes with live-only data |
| langgraph-dev | `24e4cf08` | the prefix builder's date resolver reads `ODR_FROZEN_DATE` |
| pitlane | `3eb49a6` | `TAVILY_API_KEYS` in common.env / example.env |
| pitlane | `1ec2cc2` | two arms at once, one per GPU |
| pitlane | `f1d5521`, `bb83e64` | `host_share` added, then dropped — three values that cost more attention than they returned |
| pitlane | `3a4b642` | the three `ours` switches and the ablation arms |

Branches: ODR `Stable-compile-time-helper-test-2`, langgraph-dev
`Prediction-ODR-test`, pitlane `prefetch-usefulness-metrics`. All pushed as of
`3a4b642`; anything after that is listed under **Open** below.

## The four mechanisms, and how to switch them

| switch | variable | layer |
|---|---|---|
| `BENCH_NODE_EVICTION` | `VLLM_NODE_EVICTION_POLICY` | vLLM server |
| `BENCH_PREFETCH` | `KV_EVICTION_DISABLE_PREFETCH` (inverted) | workflow |
| `BENCH_PROMPT_SEEDS` | `ODR_PROMPT_SEEDS` | workflow |
| — | `LANGGRAPH_PROMPT_PSEUDO_DYNAMIC` | workflow, global |

Three facts that are not obvious from the names:

- **`prefetch` is the master switch.** Off clears the agent env, taking the
  prompt seeds and the pseudo-dynamic flag with it. A mode that sets either
  with prefetch off is rejected at load.
- **The `BENCH_` flags are ignored on an ablation arm.** A mode states its own
  switches; a box-wide variable cannot overrule one. Without that guard
  `ours_full` with `BENCH_PROMPT_SEEDS=0` in `.env` would run as
  `ours_no_seeds` and still report itself as `ours_full`.
- **`BENCH_SEED_PREFIXES` must stay empty.** The system prompt population
  phase is not an ablation: the registry is written only by `/v1/agent_chat`
  and `/v1/agents/*`, so skipping it leaves every lookup empty and every
  prefetch a silent no-op.

Modes live in `pitlane/ablation.toml`, become `ours_<mode>` arms derived from
`[ours]`, and are excluded from the default matrix. See
[runbooks/05-ablation.md](runbooks/05-ablation.md).

## The oracle, and what replaced it

`replay_prefetch.py` resolved the *next* response out of a pinned trace, so it
warmed a successor with the producing call's whole prefill and decode as lead.
That is an upper bound on perfect, perfectly-early prediction, not a property
of this system. Removed with `prefetch_timing.py`.

Two of its five routes were not predictions and were worth keeping:
`compress_research` and `final_report_generation` open prompts that share no
usable prefix with anything the server has computed — zero shared leading
messages on 33/33 compress calls, 132 shared characters for the report — so
both prefill from scratch on the critical path.

`prompt_seeds.py` rebuilds those two from live data only. The rule it keeps:
**this call's own response, never a later record.** Under pinned replay its
response hook is handed the recording, because that is the reply the agent is
acting on — the same value a live run has at that instant. Asserted
structurally in `tests/check_prompt_seeds.py`: no import of `trace_store`, and
no function may take `entry`, `store`, `recorded_body` or `next_record`, which
was the shape of every oracle planner.

The price is one message. The compress seed is built while the researcher's
last turn is still generating, so it cannot contain that reply; a second seed
adds it once it arrives, with the turn's tool phase to prefill in. Only the
iteration-cap exit is seeded — the other two exits are properties of the
reply, and by the time they are known `researcher_tools` routes straight to
compression with no gap to use. See [runbooks/03-ours.md](runbooks/03-ours.md)
§4b.

## The date fix, which changed a flag's meaning

langgraph's prefix builder had its own copy of `get_today_str` with the
`ODR_FROZEN_DATE` branch dropped, so every warm was built against the wall
clock while the request it was meant to match carried the frozen date. They
diverged at `{date}`, which sits ~5% into every system prompt.

That made `LANGGRAPH_PROMPT_PSEUDO_DYNAMIC=0` nearly free — the warm died
there anyway. With the date correct, `1` builds on to the next manufactured
segment (`research_system_prompt` runs to `{mcp_prompt}` instead of stopping
at `{date}`), so prefetch volume and usefulness both move. It is the one flag
whose meaning changed, and `ours_no_pseudo` exists to measure it.

## Two stacks on one box

Four things were single-stack constants and are now per stack: the vLLM port,
the LMCache port, the tmux session names, and `CUDA_VISIBLE_DEVICES` (which
was never set at all — `BENCH_GPU` only fed preflight). The tmux names mattered
most: `start_server` kills the session it is about to use, so two stacks named
`vllm` meant the second launch tore down the first arm's server hours in.

The fifth was quieter: the resume ledger is read-modify-write through a fixed
temp name, and two processes under one run id lost each other's entries —
**11 of 48 survived** in testing. Now flocked.

Latency is not co-tenancy-safe; token counts and hit rates are. See
[runbooks/04-parallel.md](runbooks/04-parallel.md), which has the column-by-column
split and the measured RAM figures (the n50 batch peaked at ~25-38 GB against
a 200 GB L1 ceiling, which is why the parallel overlays ship 100 each).

## Tavily key pool

`TAVILY_API_KEYS` takes one key per account, comma separated. Queries go
round-robin so the spend divides evenly; 401/403/432/433 retires a key and
retries the same query on the next one, 429 benches it and three benchings
retire it. Timeouts and 5xx do not rotate. `python tests/check_tavily_keys.py`
in the workflow repo probes them all before a long run.

The box currently has three keys configured, not four.

## State

- **open_deep_research** `60fc8d9`, pushed, clean.
- **langgraph-dev** `24e4cf08`, pushed, clean. Running its test suite dirties a
  tracked `libs/langgraph/debug.txt` — worth gitignoring, since a dirty tree
  there is what silenced `/v1/agents` once before.
- **pitlane** `3a4b642` pushed; see **Open**.
- Untracked junk in pitlane's root: `.DS_Store`, `batch50_20260915_230638/`
  and its `.tar` (an extracted run, the source of the RAM figures above).

## Open

Uncommitted in pitlane at the time of writing: the guard that makes the
`BENCH_` flags ignore ablation arms (`stack.py`, `workflow.py`), its six
assertions in `tests/test_arm_flags.py`, and the matching note in
`05-ablation.md`. Commit before running anything.

Known loose ends, none blocking:

- `vllm` (`remove-engine-prefetch`): `controller.py:696` reads
  `self.config.prefetch_min_coverage`, which is not a field on
  `NodeEvictionConfig` — safe only because its method `_is_resident` has no
  callers left. And `node_eviction.json` still ships ten `prefetch_*` keys that
  the config dataclass does not declare, so `from_dict` warns and drops them at
  every boot. `prefetch_agent_namespace` is the one that could mislead:
  `system_prompt_population.py`'s docstring says the server reads the namespace
  from it, and on this branch it does not.
- Three pre-existing failures in `libs/langgraph/tests/test_utils.py`
  (`test_prompt_composition_*`), confirmed on a clean tree — not from this
  session.
- `pitlane/tests/test_resume_e2e.py`'s `StubConfig` needed a new field three
  times this session. A fourth flag should push `workflow.run` to read these
  through one accessor.
- `test_metrics.py:197` still uses `route=replay_prefetch` as a fixture string.
  The collector parses routes generically, so it is a valid parser test under a
  dead name.

## Suggested skills

None specific. The work is spread over three repos; `git log` in each is the
authority, and the runbooks are the task-level detail.
