# pitlane — documentation

| Document | What it covers |
|---|---|
| [01-design.md](01-design.md) | The design: features, components, config keys, execution order, and the exact metric definitions with their sources |
| [02-running.md](02-running.md) | How to start an evaluation for 1, 10 or 30 questions, on ODR or swe-agent |
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
