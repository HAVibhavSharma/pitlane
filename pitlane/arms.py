"""Arm definitions -- what makes one serving stack differ from another.

Everything variable lives in arms.toml so adding a stack is a config block, not
a code change. The only thing this module does beyond parsing is expand the
`{repo}` / `{cell}` placeholders, which cannot be written literally in the file
because they depend on the run.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_ARMS_FILE = Path(__file__).with_name("arms.toml")
_PLAN = Path(__file__).resolve().parents[1] / "plan"


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

    def resolved_env(self, which: str, *, repo: Path, cell: Path,
                     model: str = "") -> dict[str, str]:
        source = self.server_env if which == "server" else self.workflow_env
        return {
            key: self.expand(value, repo=repo, cell=cell, model=model)
            for key, value in source.items()
        }

    @staticmethod
    def expand(value: str, *, repo: Path, cell: Path, model: str = "") -> str:
        """Substitute the run-dependent placeholders.

        Literal replacement, not str.format: server args carry JSON
        (``{"kv_connector": ...}``) whose braces format() would try to read as
        fields.

        ``{model}`` exists so an arm can name the served model without the name
        being written twice. Two copies drift, and the way this one drifts is
        silent: langgraph's agent worker reports itself disabled when it cannot
        resolve a model, so the predictor stops without an error.
        """
        for token, path in (("{repo}", repo), ("{cell}", cell), ("{plan}", _PLAN)):
            value = value.replace(token, str(path))
        return value.replace("{model}", model)

    def resolved_server_args(self, *, repo: Path, cell: Path,
                             model: str = "") -> list[str]:
        return [self.expand(a, repo=repo, cell=cell, model=model)
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
        """Arms that measure, i.e. everything but the recording one."""
        return [name for name in self.arms if name != "record"]


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
    return Registry(arms)
