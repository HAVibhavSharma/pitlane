# Runbooks

The exact command sequence per stack, in execution order. These are the source
of truth the orchestrator encodes; run them by hand when debugging one arm.

| File | Stack | vLLM repo |
|---|---|---|
| [00-record.md](00-record.md) | baseline, recording the trace | `$VLLM_BASELINE_REPO` |
| [01-baseline.md](01-baseline.md) | vLLM + LMCache MP + LRU | `$VLLM_BASELINE_REPO` |
| [02-continuum.md](02-continuum.md) | vLLM-Continuum | `$VLLM_CONTINUUM_REPO` |
| [03-ours.md](03-ours.md) | vLLM + prefetch + node eviction | `$VLLM_OURS_REPO` |

## Conventions

- **Secrets** live in `~/.bench.env` (`TAVILY_API_KEY`, `LANGSMITH_API_KEY`);
  everything else is in [common.env](common.env). Every runbook starts with:
  ```bash
  set -a; . ~/.bench.env; . "$(dirname "$0")/common.env"; set +a
  ```
- **tmux sessions**, one per role, always the same names:
  `lmcache`, `vllm`, `workflow`. Start detached (`tmux new-session -d -s NAME`)
  so an ssh drop does not kill the run.
- **LMCache wipe and restart are one operation**, in this order: stop → delete
  `$LMCACHE_L2_DIR` → start. Wiping under a live server leaves its in-memory
  L1 index pointing at deleted files.
- **Readiness gate** after every `vllm serve`, never a fixed sleep:
  ```bash
  until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done
  ```
- **One question per boot** is the default. Reusing a boot for several
  questions means a warm LMCache from the second question on — mark those runs
  `warm` and do not average them with cold ones.
- **Preflight** before any of it: GPU 0 idle, host RAM free ≥ LMCache
  `--l1-size-gb` + margin, port 8000 unbound, `/disk2` has room.
