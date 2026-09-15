# Handoff — 2026-09-15

Session focus: making Tavily searches **execute live while their output is
replaced by the cache**, so `tavily_search.query` spans time a real search
instead of a disk read. Plus a leftover crash fix in the node-eviction eval
script.

Background on the harness itself is in [01-design.md](01-design.md) and
[02-running.md](02-running.md); don't re-derive it here.

## What changed

Two repos. Commit messages carry the reasoning — read them rather than
re-explaining.

| Repo | Commit | What |
|---|---|---|
| open_deep_research | `6233b77` | `TAVILY_CACHE_PINNED` — on a cache hit the query still goes to Tavily, the live response is discarded, the cached one is returned |
| open_deep_research | `e97a236` | the probe's log line needed its own handler; root logger sits at WARNING so `logging.info` was being dropped |
| open_deep_research | `263a45b` | fixes `NameError: name 'signal' is not defined` that killed a 50-question node-eviction run after model load |
| pitlane | `a8b27d8` | `export TAVILY_CACHE_PINNED=1` in `runbooks/common.env`; documented at `0` in `example.env` |

Code entry points: `src/open_deep_research/utils.py` — flag at `:473`,
`_tavily_live_probe` at `:557`, hit path at `:620-636`, span field at `:672`.

## State

- **open_deep_research**: 1 unpushed commit (`e97a236`). The rest are on origin.
- **pitlane**: in sync with `origin/prefetch-usefulness-metrics`. Untracked
  `batch2__job2__baseline.mmd` in the repo root is a stray copied artifact —
  safe to delete, was never committed.
- **vllm / baseline-vllm / vllm-continuum**: untouched this session, in sync.
  continuum's modified `hbm_summary.py` + test are pre-existing, not from here.
- Nothing has run against real hardware. Every check below was a fake client.

## How the feature behaves

- Fires only on a cache **hit**. A miss is the old path: fetch, freeze, return
  the live result — nothing to substitute.
- Scope is the per-query fetch, not the whole tool call. The ~45 s parent
  `tavily_search` span is `summarize_webpage` LLM calls, already live-and-pinned
  through `trace_store`.
- Trajectory-safe: the agent receives the same frozen results either way, so
  prompts and token counts are unchanged and **the existing trace stays valid**.
  Only wall-clock moves.
- Probe runs outside the per-key lock, deliberately: holding it would serialise
  duplicate queries a live run issues concurrently.
- One attempt, never raises. A 429 degrades to "this span under-reports" with a
  warning, never a failed run.

## Next session

1. **Push `e97a236`** before anything runs — without it the divergence log line
   is invisible and a working run looks like the flag never took.
2. **`common.env` on the box (chisel-8) is a separate copy** from this checkout.
   Pull pitlane there or add the export by hand, or the flag is set here and
   nowhere that runs.
3. Set it for **all three arms or none**. One arm alone carries seconds of extra
   tool time and the comparison stops being one.
4. Rebuild the three vLLM venvs if they predate the flush/echo/log-format
   commits (see each repo's log).
5. Verify after the first pinned run:
   `grep -c tavily_pinned <run>/<arm>/<cell>/workflow.log` — **workflow.log**,
   not server.log. Zero means the env vars didn't propagate to the child, or
   every query missed the cache.
6. Expect `tavily_search.query` spans to jump from 1-10 ms to real search
   latency, and the parent span to grow by about that much.

## Watch out for

- Two bugs this session came from **partial copies** between
  `tests/run_evaluate.py` and `tests/run_evaluate_node_eviction.py`. If you
  touch one, diff the shared regions of both. An AST unbound-name scan catches
  what eyeballing misses.
- Env vars are read at import. Exporting mid-run does nothing to a process
  already up, and a bare assignment at a shell prompt never reaches the child
  (inside `common.env` it's fine — the runbooks source it with `set -a`).
- Pinned mode spends a real API call per query per arm. A 3-arm x 50-question
  matrix multiplies Tavily usage roughly 3x, and rate limiting shows up as
  under-reported spans rather than an error.

## Verification already done

Fake-client test, not committed, at
`<scratchpad>/t_pin.py` (session scratchpad; recreate if gone). Covers: miss
fetches and caches; hit with the flag off makes zero calls; hit with it on calls
out, discards that client's result, returns the cached one, and the await lands
in elapsed time; three concurrent duplicates take ~0.2 s not 0.6 s; a raising
probe still serves cache; a fully-failed miss still propagates. The INFO line
was confirmed emitted with the flag set at import.

## Suggested skills

- `/code-review ultra` on the ODR branch before the next long run — the last two
  regressions here were both in eval-harness plumbing that only fails after
  model load, which is an expensive place to find out.
- No project-specific skills are needed for the Tavily work itself.
