# Runbooks

The exact command sequence per stack, in execution order. These are the source
of truth the orchestrator encodes; run them by hand when debugging one arm.

| File | Stack | vLLM repo | virtualenv |
|---|---|---|---|
| [00-record.md](00-record.md) | baseline, recording the trace | `$VLLM_BASELINE_REPO` | `$VLLM_BASELINE_VENV` |
| [01-baseline.md](01-baseline.md) | vLLM + LMCache MP + LRU | `$VLLM_BASELINE_REPO` | `$VLLM_BASELINE_VENV` |
| [02-continuum.md](02-continuum.md) | vLLM-Continuum | `$VLLM_CONTINUUM_REPO` | `$VLLM_CONTINUUM_VENV` |
| [03-ours.md](03-ours.md) | vLLM + prefetch + node eviction | `$VLLM_OURS_REPO` | `$VLLM_OURS_VENV` |

## Conventions

- **Every key is in [`example.env`](../../example.env)** at the repo root.
  Copy it to `.env` (gitignored) and fill it in; that is the one file to edit
  on a new box. Secrets may stay in `~/.bench.env` instead. Every runbook
  starts with:
  ```bash
  set -a; . ~/.bench.env
  [ -f "$(dirname "$0")/../../.env" ] && . "$(dirname "$0")/../../.env"
  . "$(dirname "$0")/common.env"; set +a
  ```
  `.env` is guarded because it may not exist yet; without it `common.env`'s
  defaults apply. Order matters only in that a real environment variable wins
  over `common.env`'s default for the same key.
- **One virtualenv per build.** Three `vllm` checkouts cannot share a
  `site-packages`, so each has its own, and each pairs with a `*_REPO`. Set the
  four `*_VENV` keys to the venv **roots** -- pitlane appends `bin/vllm` and
  `bin/python` itself -- and note that `WORKFLOW_VENV` must carry the langgraph
  fork rather than the released package, since ODR imports
  `langgraph.pregel._vllm_agent`. `pitlane preflight` reports each one. A
  runbook activates the right one before `vllm serve`:
  ```bash
  . "$VLLM_OURS_VENV/bin/activate"      # or _BASELINE_ / _CONTINUUM_
  ```
  The workflow runs under `$WORKFLOW_VENV` instead. `pitlane` skips activation
  and calls `$VENV/bin/vllm` and `$WORKFLOW_VENV/bin/python` by absolute path;
  by hand, forgetting this boots the previous arm's build with the new arm's
  flags, which fails silently.
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
