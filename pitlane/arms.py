"""Arm definitions -- what makes one serving stack differ from another.

Everything variable lives in arms.toml so adding a stack is a config block, not
a code change. The only thing this module does beyond parsing is expand the
`{repo}` / `{cell}` placeholders, which cannot be written literally in the file
because they depend on the run.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

_ARMS_FILE = Path(__file__).with_name("arms.toml")
_ABLATION_FILE = Path(__file__).with_name("ablation.toml")
_PLAN = Path(__file__).resolve().parents[1] / "plan"

# The arm every ablation mode is derived from, and the switches a mode may set.
_ABLATION_BASE = "ours"
_ABLATION_SWITCHES = ("node_eviction", "prefetch", "prompt_seeds",
                      "pseudo_dynamic", "population")
# Meaningless with the predictor off: the workflow clears the agent env when
# prefetch is disabled, so neither reaches anything.
_NEEDS_PREFETCH = ("prompt_seeds", "pseudo_dynamic")
# The system prompt population phase, as a CLI flag rather than a variable --
# and spelled as a negative, so "on" is the absence of an argument.
_SKIP_POPULATION_FLAG = "--skip-system-prompt-population"


@dataclass(frozen=True)
class Arm:
    name: str
    description: str
    repo: str                      # key into Paths.vllm_repos
    lmcache_server: bool
    server_env: dict[str, str]
    server_args: list[str]
    workflow_script: str
    workflow_args: list[str]
    workflow_env: dict[str, str]
    needs_redis: bool = False
    log_name: str = "server.log"
    # Key into `Paths.venvs`. Defaults to `repo` in `_build`, since one
    # virtualenv per vLLM checkout is the normal case; named separately so two
    # arms can share a build with different flags without sharing a venv entry
    # by accident.
    venv: str = ""
    # Derived from another arm by ablation.toml. Opt-in only: these are a study
    # of one arm, not more arms to measure, and a default run that launched all
    # of them would turn an eleven-hour batch into three days.
    ablation: bool = False

    def resolved_env(self, which: str, *, repo: Path, cell: Path,
                     model: str = "", port: int = 8000) -> dict[str, str]:
        source = self.server_env if which == "server" else self.workflow_env
        return {
            key: self.expand(value, repo=repo, cell=cell, model=model, port=port)
            for key, value in source.items()
        }

    @staticmethod
    def expand(value: str, *, repo: Path, cell: Path, model: str = "",
               port: int = 8000) -> str:
        """Substitute the run-dependent placeholders.

        Literal replacement, not str.format: server args carry JSON
        (``{"kv_connector": ...}``) whose braces format() would try to read as
        fields.

        ``{model}`` exists so an arm can name the served model without the name
        being written twice. Two copies drift, and the way this one drifts is
        silent: langgraph's agent worker reports itself disabled when it cannot
        resolve a model, so the predictor stops without an error.

        ``{port}`` is there for the same reason and fails the same way: an arm
        that hard-codes 8000 points the workflow at whichever stack owns that
        port, which on a two-GPU box is the *other* arm's server -- and the run
        looks fine while measuring the wrong one.
        """
        for token, path in (("{repo}", repo), ("{cell}", cell), ("{plan}", _PLAN)):
            value = value.replace(token, str(path))
        return value.replace("{model}", model).replace("{port}", str(port))

    def resolved_server_args(self, *, repo: Path, cell: Path,
                             model: str = "", port: int = 8000) -> list[str]:
        return [self.expand(a, repo=repo, cell=cell, model=model, port=port)
                for a in self.server_args]

    def unset_keys(self, which: str) -> list[str]:
        """Keys the arm explicitly blanks, so they cannot leak in from a shell."""
        source = self.server_env if which == "server" else self.workflow_env
        return [key for key, value in source.items() if value == ""]


@dataclass(frozen=True)
class Registry:
    arms: dict[str, Arm] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Arm:
        try:
            return self.arms[name]
        except KeyError:
            raise KeyError(
                f"unknown arm {name!r}; known: {', '.join(sorted(self.arms))}"
            ) from None

    def __iter__(self):
        return iter(self.arms.values())

    @property
    def measurement_arms(self) -> list[str]:
        """The default matrix: everything but the recorder and the ablations.

        An ablation mode has to be asked for by name. They are seven ways of
        running one arm, so including them here would mean a bare `pitlane run`
        booting ten stacks.
        """
        return [name for name, arm in self.arms.items()
                if name != "record" and not arm.ablation]

    @property
    def ablation_arms(self) -> list[str]:
        return [name for name, arm in self.arms.items() if arm.ablation]


def _ablation_env(mode: str, switches: dict[str, bool]) -> dict[str, str]:
    """The env overlay one mode puts on top of `ours`.

    Written straight into the arm's own `server_env`/`workflow_env` rather than
    applied later, so `pitlane arms` shows exactly what a mode will run and a
    cell's `command.txt` does too. "" is the file's idiom for "unset before
    launch", which for the eviction policy is what upstream means.
    """
    if any(k in switches for k in _NEEDS_PREFETCH) and switches.get("prefetch") is False:
        raise ValueError(
            f"ablation mode {mode!r} sets "
            f"{[k for k in _NEEDS_PREFETCH if k in switches]} with prefetch off; "
            "the workflow clears the agent env in that case, so the run would "
            "be `no_prefetch` under another name"
        )
    # The one combination that produces a run which looks fine and measures
    # nothing. Only `/v1/agent_chat` and `/v1/agents/*` write the prefix
    # registry, so an unseeded registry makes every lookup empty and every
    # warm a silent no-op -- prefetch on paper, absent in fact, and reported
    # as whatever the mode called itself. With prefetch off there is no lookup
    # to leave empty, which is the only reason the switch exists.
    if switches.get("population") is False and switches.get("prefetch") is not False:
        raise ValueError(
            f"ablation mode {mode!r} skips the system prompt population phase "
            "without turning prefetch off; the registry would be empty and "
            "every prefetch a silent no-op, which is an invalid run rather "
            "than an ablation"
        )
    server: dict[str, str] = {}
    workflow: dict[str, str] = {}
    if "node_eviction" in switches:
        server["VLLM_NODE_EVICTION_POLICY"] = "1" if switches["node_eviction"] else ""
    if "prefetch" in switches:
        # Inverted: the switch says prefetch, the variable says disable.
        workflow["KV_EVICTION_DISABLE_PREFETCH"] = (
            "0" if switches["prefetch"] else "1"
        )
    if "prompt_seeds" in switches:
        workflow["ODR_PROMPT_SEEDS"] = "1" if switches["prompt_seeds"] else "0"
    if "pseudo_dynamic" in switches:
        workflow["LANGGRAPH_PROMPT_PSEUDO_DYNAMIC"] = (
            "1" if switches["pseudo_dynamic"] else "0"
        )
    args: list[str] = []
    if switches.get("population") is False:
        args.append(_SKIP_POPULATION_FLAG)
    return {"server_env": server, "workflow_env": workflow, "workflow_args": args}


def _ablation_arms(arms: dict[str, Arm],
                   path: Path = _ABLATION_FILE) -> dict[str, Arm]:
    """`ours_<mode>` for every mode in ablation.toml.

    Derived rather than written out so a change to the `ours` stack -- a server
    arg, the venv, the script -- reaches every mode without seven edits, and a
    mode differs from `ours` in nothing but its switches.
    """
    base = arms.get(_ABLATION_BASE)
    if base is None or not path.exists():
        return {}
    out: dict[str, Arm] = {}
    for mode, body in tomllib.loads(path.read_text()).items():
        switches = {k: bool(body[k]) for k in _ABLATION_SWITCHES if k in body}
        overlay = _ablation_env(mode, switches)
        summary = ", ".join(
            f"{k}={'on' if switches[k] else 'off'}" for k in _ABLATION_SWITCHES
            if k in switches
        )
        out[f"{_ABLATION_BASE}_{mode}"] = replace(
            base,
            name=f"{_ABLATION_BASE}_{mode}",
            description=body.get("description", "") or f"ours: {summary}",
            ablation=True,
            server_env={**base.server_env, **overlay["server_env"]},
            workflow_env={**base.workflow_env, **overlay["workflow_env"]},
            workflow_args=[*base.workflow_args, *overlay["workflow_args"]],
        )
    return out


def load(path: Path = _ARMS_FILE) -> Registry:
    raw = tomllib.loads(path.read_text())
    arms = {}
    for name, body in raw.items():
        arms[name] = Arm(
            name=name,
            description=body.get("description", ""),
            repo=body["repo"],
            lmcache_server=bool(body.get("lmcache_server", False)),
            server_env=dict(body.get("server_env", {})),
            server_args=list(body.get("server_args", [])),
            workflow_script=body["workflow_script"],
            workflow_args=list(body.get("workflow_args", [])),
            workflow_env=dict(body.get("workflow_env", {})),
            needs_redis=bool(body.get("needs_redis", False)),
            venv=body.get("venv", body["repo"]),
        )
    arms.update(_ablation_arms(arms))
    return Registry(arms)
