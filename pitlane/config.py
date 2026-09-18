"""Run configuration: paths, secrets, and the layout of a run's artifacts.

Everything the user controls lives in env files (`~/.bench.env` for secrets,
`common.env` for the rest). Nothing here reaches out to the network or the
filesystem beyond reading those files -- so `pitlane preflight` can validate a
config without touching the GPU.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_EXPORT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a shell-style env file into a dict.

    Deliberately not `source`: the env files are read by both this tool and by
    a human pasting the runbooks into a shell, so they must stay valid shell,
    but running them would execute whatever else is in the file.
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _EXPORT.match(line)
        if not match:
            continue
        name, raw = match.group(1), match.group(2).strip()
        # `: "${FOO:?...}"` style guards parse as junk; skip anything that is
        # not a plain assignment.
        if raw.startswith(("?", "!")):
            continue
        try:
            parts = shlex.split(raw, comments=True)
        except ValueError:
            continue
        # `FOO=` is an assignment to the empty string, not the absence of one.
        # Skipping it would mean a blank key in a later file cannot clear a
        # value an earlier file set -- and blank is exactly how a template says
        # "unset", so `LMCACHE_L2_DIR=` in `.env` would silently keep
        # common.env's path and the run would use a disk tier it was told not
        # to. Callers that treat "" as unset use `_or`.
        values[name] = os.path.expandvars(parts[0]) if parts else ""
    return values


class ConfigError(RuntimeError):
    """Raised for a config that cannot produce a valid run."""


@dataclass(frozen=True)
class Paths:
    bench_root: Path
    trace_dir: Path
    # Optional: unset runs LMCache with L1 only (its 200 GB of CPU memory) and
    # no disk tier below it. `--l2-adapter` is repeatable and defaults to an
    # empty list, and LMCache's storage manager guards every L2 path on the
    # list being non-empty, so nothing has to be stubbed out.
    lmcache_l2_dir: Path | None
    tavily_cache_dir: Path
    vllm_log_dir: Path
    workflow_repo: Path
    vllm_repos: dict[str, Path]
    # Each vLLM build is installed into its own virtualenv -- three checkouts
    # of vllm cannot share one site-packages. Empty means "whatever is on PATH",
    # which is only safe when a single arm is ever run on this machine; with
    # more than one it silently serves the same build every time.
    venvs: dict[str, Path] = field(default_factory=dict)
    workflow_venv: Path | None = None
    # The LMCache server is a third process with a third install. Usually the
    # same environment as one of the vLLM builds, but not necessarily -- so it
    # is named rather than inferred, and falls back to the arm's own venv.
    lmcache_venv: Path | None = None


_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}


def _tri_state(env: dict[str, str], key: str) -> bool | None:
    """True / False / None, where None means "the arm decides".

    A plain bool would need a default, and every default here is wrong: `False`
    would silently disable the policy for the arm whose whole purpose is to run
    it, `True` would enable it for baseline and continuum. Absent has to stay
    distinguishable from set-to-0.
    """
    raw = env.get(key, "").strip().lower()
    if not raw:
        return None
    if raw in _TRUTHY:
        return True
    if raw in _FALSEY:
        return False
    raise ConfigError(
        f"{key}={raw!r} is neither true nor false; use 1 or 0 (or leave it "
        f"empty to let the arm decide)"
    )


def _or(env: dict[str, str], key: str, default: str) -> str:
    """`env[key]` if it has a value, else `default`.

    A blank key is not a value. Shell env files are written by filling in the
    right-hand side of a template, so a key the operator has not got to yet is
    present and empty rather than absent -- and `dict.get(key, default)` hands
    back that empty string, never reaching the default.
    """
    return env.get(key, "").strip() or default


def venv_bin(venv: Path | None, name: str) -> str:
    """The path to `name` inside `venv`, or the bare name to resolve on PATH.

    Bare is the fallback, not the intent: a machine hosting more than one arm
    has one virtualenv per vLLM build, and resolving `vllm` on PATH there serves
    whichever one the shell happened to activate.
    """
    if venv is None:
        return name
    return str(Path(venv) / "bin" / name)


def venv_env(venv: Path | None, env: dict[str, str] | None = None) -> dict[str, str]:
    """`VIRTUAL_ENV` and a `PATH` prefix, so child processes stay in the venv.

    The absolute path to the binary is enough to start the right one; this is
    for everything it spawns afterwards, and for anything that shells out by
    name.
    """
    out = dict(env or {})
    if venv is None:
        return out
    bin_dir = str(Path(venv) / "bin")
    out["VIRTUAL_ENV"] = str(venv)
    out["PATH"] = f"{bin_dir}:{out.get('PATH') or os.environ.get('PATH', '')}"
    # A stale PYTHONHOME points the interpreter at another prefix's stdlib.
    out.pop("PYTHONHOME", None)
    return out


@dataclass
class Config:
    env: dict[str, str]
    paths: Paths
    model_name: str
    trace_path: Path
    gpu: int = 0
    min_free_ram_gb: float = 260.0
    # LMCache's L1 pool: CPU memory, grown lazily from a 20 GB initial pool up
    # to this ceiling, where LRU starts discarding. It is the largest single
    # thing a run allocates on the host, so it and `min_free_ram_gb` are one
    # decision -- see `Config.load`.
    lmcache_l1_gb: float = 200.0
    server_ready_timeout_s: float = 1800.0
    port: int = 8000
    lmcache_port: int = 10903
    # Names this stack's tmux sessions and nothing else. Two pitlane processes
    # sharing a box -- one arm per GPU -- would otherwise both own the sessions
    # called "vllm" and "lmcache", and `start_server` kills that session before
    # starting its own, so the second launch would tear down the first arm's
    # server several hours into it.
    instance: str = "default"
    # Node-aware KV eviction, overriding whatever the arm asks for: True forces
    # the full policy on, False forces the upstream LRU free-block queue, None
    # (the default) leaves the arm's own server_env alone. Set from
    # BENCH_NODE_EVICTION so the on/off pair is one variable rather than an
    # edit to arms.toml -- the server's environment is built from the arm
    # alone, so an env-file overlay cannot otherwise reach it.
    node_eviction: bool | None = None
    # Agent prefix prefetch, same three states and the same scoping: True
    # leaves it on, False turns it off, None lets the arm decide. Set from
    # BENCH_PREFETCH. The workflow's own switch is spelled the other way round
    # (KV_EVICTION_DISABLE_PREFETCH), which is exactly why this exists: one
    # variable that reads the way the question is asked.
    prefetch: bool | None = None
    # The system prompt population phase: fill each node's prompt as far as
    # the first runtime-only field, POST it to /v1/agents/prefetch so the
    # server records the prefix, and (with prefill_on_miss) leave it resident
    # before the measured phase. True runs it, False skips it, None lets the
    # arm decide. Set from BENCH_SEED_PREFIXES.
    seed_prefixes: bool | None = None
    # The prompt seeds: rebuild compress_research's and
    # final_report_generation's prompts from requests already sent and prefill
    # them during a gap. True on, False off, None lets the arm decide. Set from
    # BENCH_PROMPT_SEEDS.
    prompt_seeds: bool | None = None
    redis_url: str = "redis://127.0.0.1:6379/0"
    # Below this lead a phantom is counted late -- see `metrics.LEAD_MIN_S`.
    prefetch_lead_min_s: float = 0.1
    # Live ceilings, watched for the length of each cell. The RAM floor is the
    # one that matters: LMCache's L1 pool grows toward `--l1-size-gb` during a
    # run, and the alternative to aborting is the kernel's OOM killer choosing
    # its own victim. The VRAM ceiling is off by default -- vLLM claims its
    # share up front, so a nearly full card is a working one.
    abort_free_ram_gb: float = 32.0
    abort_vram_frac: float = 0.0
    resource_interval_s: float = 5.0

    # Resolved at run time, not from the env file.
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    @classmethod
    def load(cls, env_files: list[Path], overrides: dict[str, str] | None = None,
             overlays: list[Path] | None = None) -> "Config":
        env: dict[str, str] = {}
        for path in env_files:
            env.update(load_env_file(path))
        # A real environment variable wins over the files, so a one-off run can
        # be steered without editing them.
        env.update({k: v for k, v in os.environ.items() if k in env})
        # ...but not over an overlay, which is `--env-extra` and is the more
        # specific statement: it names the handful of keys that make this
        # invocation a different stack, and it is passed per invocation rather
        # than set once for the shell.
        #
        # This ordering is load-bearing on a two-GPU box. `launch-parallel.sh`
        # sources `.env` with `set -a` to resolve BENCH_ROOT for itself, which
        # exports BENCH_PORT, BENCH_LMCACHE_PORT, BENCH_INSTANCE and BENCH_GPU
        # into both children -- so with the overlay applied first, every one of
        # the four keys that distinguishes stack B was overwritten by stack A's
        # value, and both halves of the run became the same stack: one tmux
        # session name, one port, one card. It surfaced as `duplicate session:
        # lmcache` in one arm and, in the other, as its LMCache server being
        # killed by the arm that then failed to replace it.
        for path in overlays or []:
            env.update(load_env_file(path))
        env.update(overrides or {})

        def need(key: str) -> str:
            value = env.get(key, "").strip()
            if not value:
                raise ConfigError(f"{key} is not set (looked in {', '.join(map(str, env_files))})")
            return value

        paths = Paths(
            # `.get` with a default does not help a key that is present but
            # blank: it returns the empty string, and `Path("")` is the current
            # directory. Every artifact would land wherever the tool was run
            # from, which looks like it worked. Same for the two below.
            bench_root=Path(_or(env, "BENCH_ROOT", "/disk2/vibhav/bench")),
            trace_dir=Path(need("TRACE_DIR")),
            lmcache_l2_dir=(
                Path(value) if (value := env.get("LMCACHE_L2_DIR", "").strip())
                else None
            ),
            tavily_cache_dir=Path(_or(env, "TAVILY_CACHE_DIR", "")),
            vllm_log_dir=Path(_or(env, "VLLM_LOG_DIR", "")),
            workflow_repo=Path(need("WORKFLOW_REPO")),
            vllm_repos={
                "baseline": Path(need("VLLM_BASELINE_REPO")),
                "continuum": Path(need("VLLM_CONTINUUM_REPO")),
                "ours": Path(need("VLLM_OURS_REPO")),
            },
            venvs={
                key: Path(value)
                for key, name in (
                    ("baseline", "VLLM_BASELINE_VENV"),
                    ("continuum", "VLLM_CONTINUUM_VENV"),
                    ("ours", "VLLM_OURS_VENV"),
                )
                if (value := env.get(name, "").strip())
            },
            workflow_venv=(
                Path(value) if (value := env.get("WORKFLOW_VENV", "").strip()) else None
            ),
            lmcache_venv=(
                Path(value) if (value := env.get("LMCACHE_VENV", "").strip()) else None
            ),
        )
        # L1 and the RAM preflight are one decision, not two. The floor exists
        # *because* of the pool, so deriving it keeps a lowered L1 from leaving
        # behind a 260 GB requirement that has nothing to do with the run --
        # and a raised one from passing a check it should not. An explicit
        # BENCH_MIN_FREE_RAM_GB still wins, for a box with other tenants.
        l1_gb = float(_or(env, "BENCH_LMCACHE_L1_GB", "200"))
        ram_margin_gb = float(_or(env, "BENCH_RAM_MARGIN_GB", "60"))
        # Ports are per stack, not per box: a second concurrent arm needs its
        # own vLLM and its own LMCache. The instance label defaults to the vLLM
        # port so that changing the port is enough -- forgetting the label
        # cannot silently give two stacks the same tmux session.
        port = int(_or(env, "BENCH_PORT", "8000"))
        lmcache_port = int(_or(env, "BENCH_LMCACHE_PORT", "10903"))

        return cls(
            env=env,
            paths=paths,
            model_name=need("MODEL_NAME"),
            trace_path=Path(need("ODR_TRACE_PATH")),
            gpu=int(env.get("BENCH_GPU", "0")),
            port=port,
            lmcache_port=lmcache_port,
            instance=_or(env, "BENCH_INSTANCE", f"p{port}"),
            node_eviction=_tri_state(env, "BENCH_NODE_EVICTION"),
            prefetch=_tri_state(env, "BENCH_PREFETCH"),
            seed_prefixes=_tri_state(env, "BENCH_SEED_PREFIXES"),
            prompt_seeds=_tri_state(env, "BENCH_PROMPT_SEEDS"),
            lmcache_l1_gb=l1_gb,
            min_free_ram_gb=float(
                _or(env, "BENCH_MIN_FREE_RAM_GB", str(l1_gb + ram_margin_gb))
            ),
            server_ready_timeout_s=float(env.get("BENCH_SERVER_READY_TIMEOUT_S", "1800")),
            redis_url=env.get("KV_FORECAST_REDIS_URL", "redis://127.0.0.1:6379/0"),
            prefetch_lead_min_s=float(env.get("BENCH_PREFETCH_LEAD_MIN_S", "0.1")),
            abort_free_ram_gb=float(_or(env, "BENCH_ABORT_FREE_RAM_GB", "32")),
            abort_vram_frac=float(_or(env, "BENCH_ABORT_VRAM_FRAC", "0")),
            resource_interval_s=float(_or(env, "BENCH_RESOURCE_INTERVAL_S", "5")),
        )

    # -- artifact layout --------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return self.paths.bench_root / self.run_id

    def cell_dir(self, arm: str, question_id: str, rep: int) -> Path:
        return self.run_dir / arm / question_id / f"rep{rep}"

    def base_url(self, suffix: str = "/v1") -> str:
        return f"http://localhost:{self.port}{suffix}"
