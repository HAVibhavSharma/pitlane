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
    server_ready_timeout_s: float = 1800.0
    port: int = 8000
    lmcache_port: int = 10903
    redis_url: str = "redis://127.0.0.1:6379/0"
    # Below this lead a phantom is counted late -- see `metrics.LEAD_MIN_S`.
    prefetch_lead_min_s: float = 0.1

    # Resolved at run time, not from the env file.
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    @classmethod
    def load(cls, env_files: list[Path], overrides: dict[str, str] | None = None) -> "Config":
        env: dict[str, str] = {}
        for path in env_files:
            env.update(load_env_file(path))
        # A real environment variable wins over the files, so a one-off run can
        # be steered without editing them.
        env.update({k: v for k, v in os.environ.items() if k in env})
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
        return cls(
            env=env,
            paths=paths,
            model_name=need("MODEL_NAME"),
            trace_path=Path(need("ODR_TRACE_PATH")),
            gpu=int(env.get("BENCH_GPU", "0")),
            min_free_ram_gb=float(env.get("BENCH_MIN_FREE_RAM_GB", "260")),
            server_ready_timeout_s=float(env.get("BENCH_SERVER_READY_TIMEOUT_S", "1800")),
            redis_url=env.get("KV_FORECAST_REDIS_URL", "redis://127.0.0.1:6379/0"),
            prefetch_lead_min_s=float(env.get("BENCH_PREFETCH_LEAD_MIN_S", "0.1")),
        )

    # -- artifact layout --------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return self.paths.bench_root / self.run_id

    def cell_dir(self, arm: str, question_id: str, rep: int) -> Path:
        return self.run_dir / arm / question_id / f"rep{rep}"

    def base_url(self, suffix: str = "/v1") -> str:
        return f"http://localhost:{self.port}{suffix}"
