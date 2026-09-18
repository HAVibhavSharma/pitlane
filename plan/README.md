# pitlane — documentation

| Document | What it covers |
|---|---|
| [01-design.md](01-design.md) | The design: features, components, config keys, execution order, and the exact metric definitions with their sources |
| [02-running.md](02-running.md) | How to start an evaluation for 1, 10 or 30 questions, on ODR or swe-agent |
| [handoff-2026-09-18-ablation.md](handoff-2026-09-18-ablation.md) | Current handoff: the `ours` arm made decomposable — the oracle removed, prompt seeds in its place, the three switches and the ablation arms, two stacks on two GPUs, the Tavily key pool |
| [handoff-2026-09-15-tavily-pinned.md](handoff-2026-09-15-tavily-pinned.md) | Superseded. Kept for the reasoning behind `TAVILY_CACHE_PINNED` |
| [ABLATION-SMOKE-TEST.md](ABLATION-SMOKE-TEST.md) | One question per ablation mode, with the evidence each switch must leave — run it on the box before committing to a full batch |
| [runbooks/](runbooks/README.md) | The commands each serving stack needs, in order — what the tool automates, and what to run by hand when debugging one arm |

## Where a question is answered

- *What does TTFT / KV hit rate / late or useful prefetch mean here?* →
  [01-design.md, §5](01-design.md) (and `pitlane/metrics.py` for the code)
- *Which env vars do I have to set?* → [01-design.md, §3](01-design.md), then
  [runbooks/common.env](runbooks/common.env)
- *How do I bring up one stack by hand?* →
  [runbooks/01-baseline.md](runbooks/01-baseline.md),
  [02-continuum.md](runbooks/02-continuum.md),
  [03-ours.md](runbooks/03-ours.md)
- *How do I build the trace the arms replay?* →
  [runbooks/00-record.md](runbooks/00-record.md)
- *How do I run 1 / 10 / 30 questions?* → [02-running.md](02-running.md)
- *How do I run the whole matrix?* → [../README.md](../README.md)
- *How do I switch one mechanism of `ours` off?* →
  [runbooks/05-ablation.md](runbooks/05-ablation.md); check it works first with
  [ABLATION-SMOKE-TEST.md](ABLATION-SMOKE-TEST.md)
- *Can I run two arms at once on two GPUs?* →
  [runbooks/04-parallel.md](runbooks/04-parallel.md) — yes, but latency stops
  being comparable
- *What happened last session, and what is safe to trust?* →
  [handoff-2026-09-18-ablation.md](handoff-2026-09-18-ablation.md)
