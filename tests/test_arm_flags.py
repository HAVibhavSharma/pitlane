"""BENCH_NODE_EVICTION, PREFETCH, SEED_PREFIXES, PROMPT_SEEDS: the `ours` switches.

Each turns one thing on or off for the arm that has it, and none of them can
reach any other arm -- baseline and continuum are the control, and a variable
that could change them would be a way to invalidate the comparison without
leaving a trace in the results. The three are scoped by three different
tests, because the three things live in three different places: the server's
environment, the workflow's environment, and the workflow's command line.

The server's environment is built from the arm alone -- no env file, no
os.environ -- so before this flag existed, turning the eviction policy off
meant editing a tracked arms.toml. The test is what `start_server` would
export, without starting one.

Half of what it asserts is about the arms the flag must NOT reach. baseline
and continuum are the control; a variable that could switch the policy on for
them would not be a convenience, it would be a way to invalidate the
comparison without leaving a trace in the results.

    python3 tests/test_arm_flags.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitlane import stack, workflow
from pitlane import arms as arms_mod
from pitlane.config import Config, ConfigError

FAILURES = 0


def check(label: str, condition: bool) -> None:
    global FAILURES
    print(("   ok   " if condition else "   FAIL ") + label)
    if not condition:
        FAILURES += 1


BASE_ENV = """
TRACE_DIR=/tmp/traces
ODR_TRACE_PATH=/tmp/traces/odr.jsonl
MODEL_NAME=stub-model
WORKFLOW_REPO=/tmp/workflow
VLLM_BASELINE_REPO=/tmp/baseline
VLLM_CONTINUUM_REPO=/tmp/continuum
VLLM_OURS_REPO=/tmp/ours
"""


def server_env(config: Config, arm_name: str, cell: Path) -> dict[str, str]:
    """The real thing `start_server` exports, without starting one."""
    return stack.server_environment(config, arms_mod.load()[arm_name], cell)


def workflow_env(config: Config, arm_name: str, cell: Path) -> dict[str, str]:
    """The workflow env as `workflow.run` builds it, via the real functions."""
    arm = arms_mod.load()[arm_name]
    env = dict(config.env)
    env.update(arm.resolved_env("workflow", repo=Path("/tmp/repo"), cell=cell,
                                model=config.model_name, port=config.port))
    # Both overrides, in the order workflow.run applies them.
    workflow._apply_prefetch_override(env, config, arm)
    workflow._apply_prompt_seeds_override(env, config, arm)
    return env


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = root / "base.env"
        base.write_text(BASE_ENV)

        def config_with(value: str) -> Config:
            overlay = root / f"evict-{value or 'empty'}.env"
            overlay.write_text(f"BENCH_NODE_EVICTION={value}\n")
            return Config.load([base, overlay])

        print("\n== empty: the arm decides, exactly as before the flag existed")
        unset = config_with("")
        check("config reads it as None", unset.node_eviction is None)
        check("ours keeps its policy",
              server_env(unset, "ours", root).get("VLLM_NODE_EVICTION_POLICY") == "1")
        check("baseline still has none",
              "VLLM_NODE_EVICTION_POLICY" not in server_env(unset, "baseline", root))

        print("\n== 0: upstream LRU, even for the arm that asks for the policy")
        off = config_with("0")
        check("config reads it as False", off.node_eviction is False)
        check("ours: the key is gone, not set to 0 -- the fork treats a present "
              "0 and an absent key the same, but the arms.toml idiom is unset",
              "VLLM_NODE_EVICTION_POLICY" not in server_env(off, "ours", root))
        check("ours keeps the rest of its stack (LMCache MP connector)",
              server_env(off, "ours", root).get("LMCACHE_MP_FULL_HIT_ONLY") == "0")
        for control in ("baseline", "continuum", "record"):
            check(f"{control} is untouched",
                  "VLLM_NODE_EVICTION_POLICY" not in server_env(off, control, root))

        print("\n== 1: the full policy, for the eviction arm only")
        on = config_with("1")
        check("config reads it as True", on.node_eviction is True)
        check("ours is on",
              server_env(on, "ours", root).get("VLLM_NODE_EVICTION_POLICY") == "1")
        # The control arms are the reason the flag is scoped: a variable that
        # could switch them on is a variable that can invalidate the A/B.
        for control in ("baseline", "continuum", "record"):
            check(f"{control} is untouched",
                  "VLLM_NODE_EVICTION_POLICY" not in server_env(on, control, root))

        print("\n== BENCH_PREFETCH: the same three states, the same scoping")

        def prefetch_config(value: str) -> Config:
            overlay = root / f"prefetch-{value or 'empty'}.env"
            overlay.write_text(f"BENCH_PREFETCH={value}\n")
            return Config.load([base, overlay])

        empty = prefetch_config("")
        check("empty leaves the arm's own setting",
              workflow_env(empty, "ours", root)["KV_EVICTION_DISABLE_PREFETCH"] == "0")

        # Inverted: the flag asks for prefetch, the variable asks for disable.
        pf_on = prefetch_config("1")
        check("1 -> KV_EVICTION_DISABLE_PREFETCH=0",
              workflow_env(pf_on, "ours", root)["KV_EVICTION_DISABLE_PREFETCH"] == "0")
        pf_off = prefetch_config("0")
        check("0 -> KV_EVICTION_DISABLE_PREFETCH=1",
              workflow_env(pf_off, "ours", root)["KV_EVICTION_DISABLE_PREFETCH"] == "1")

        # The arm's own agent vars are left alone: the harness derives them
        # from the disable flag, and writing them here would be a second
        # source of truth for one decision.
        check("the agent vars are not second-guessed",
              workflow_env(pf_off, "ours", root)["LANGGRAPH_VLLM_AGENT_ENABLE"] == "1")

        for control in ("baseline", "continuum", "record"):
            for cfg, label in ((pf_on, "1"), (pf_off, "0")):
                check(f"{control} untouched by BENCH_PREFETCH={label}",
                      "KV_EVICTION_DISABLE_PREFETCH" not in workflow_env(cfg, control, root))

        print("\n== BENCH_SEED_PREFIXES: a CLI flag, and a negative one")
        # The scope test needs real scripts to sniff, since that is how the
        # flag decides which arms it applies to.
        repo = root / "workflow_repo" / "tests"
        repo.mkdir(parents=True)
        (repo / "run_evaluate_node_eviction.py").write_text(
            "--skip-system-prompt-population --completed-log\n")
        (repo / "run_evaluate.py").write_text("--completed-log\n")
        wf_repo = root / "workflow_repo"

        def seed_config(value: str) -> Config:
            overlay = root / f"seed-{value or 'empty'}.env"
            overlay.write_text(f"BENCH_SEED_PREFIXES={value}\n")
            return Config.load([base, overlay])

        def population_args(config: Config, arm_name: str) -> list:
            return workflow._population_args(config, arms_mod.load()[arm_name], wf_repo)

        check("empty adds nothing", population_args(seed_config(""), "ours") == [])
        check("1 adds nothing -- on is the absence of the skip",
              population_args(seed_config("1"), "ours") == [])
        check("0 adds the skip", population_args(seed_config("0"), "ours")
              == ["--skip-system-prompt-population"])
        # baseline runs run_evaluate.py, which has no population phase; the
        # flag must not put an argument on it that it would die on.
        for control in ("baseline", "continuum", "record"):
            check(f"{control} has no phase to skip, so gets no argument",
                  population_args(seed_config("0"), control) == [])
        # A repo predating the phase: sniffing means an old checkout runs as
        # it always did instead of failing on an unknown flag.
        old = root / "old_repo" / "tests"
        old.mkdir(parents=True)
        (old / "run_evaluate_node_eviction.py").write_text("no such flag here\n")
        check("an older checkout is left alone",
              workflow._population_args(seed_config("0"), arms_mod.load()["ours"],
                                        root / "old_repo") == [])

        print("\n== BENCH_PROMPT_SEEDS: scoped like the prefetch flag")

        def seeds_config(value: str) -> Config:
            overlay = root / f"seeds-{value or 'empty'}.env"
            overlay.write_text(f"BENCH_PROMPT_SEEDS={value}\n")
            return Config.load([base, overlay])

        check("empty writes nothing",
              "ODR_PROMPT_SEEDS" not in workflow_env(seeds_config(""), "ours", root))
        check("1 -> ODR_PROMPT_SEEDS=1",
              workflow_env(seeds_config("1"), "ours", root)["ODR_PROMPT_SEEDS"] == "1")
        check("0 -> ODR_PROMPT_SEEDS=0",
              workflow_env(seeds_config("0"), "ours", root)["ODR_PROMPT_SEEDS"] == "0")
        for control in ("baseline", "continuum", "record"):
            check(f"{control} untouched",
                  "ODR_PROMPT_SEEDS" not in workflow_env(seeds_config("1"), control, root))

        print("\n== the ablation modes")
        registry = arms_mod.load()
        cell = root

        def switches(arm_name: str) -> dict:
            arm = registry[arm_name]
            server = arm.resolved_env("server", repo=root, cell=cell, model="m")
            work = arm.resolved_env("workflow", repo=root, cell=cell, model="m")
            # "" is unset-before-launch; stack/workflow drop those keys.
            return {
                "eviction": server.get("VLLM_NODE_EVICTION_POLICY") or None,
                "prefetch": work.get("KV_EVICTION_DISABLE_PREFETCH"),
                "seeds": work.get("ODR_PROMPT_SEEDS"),
                "pseudo": work.get("LANGGRAPH_PROMPT_PSEUDO_DYNAMIC"),
            }

        check("ablations do not join the default matrix",
              registry.measurement_arms == ["baseline", "continuum", "ours"])
        check("seven modes exist", len(registry.ablation_arms) == 7)

        expected = {
            "ours_full":            {"eviction": "1",  "prefetch": "0", "seeds": "1", "pseudo": "1"},
            "ours_no_seeds":        {"eviction": "1",  "prefetch": "0", "seeds": "0", "pseudo": "1"},
            "ours_no_pseudo":       {"eviction": "1",  "prefetch": "0", "seeds": "1", "pseudo": "0"},
            "ours_predictor_only":  {"eviction": "1",  "prefetch": "0", "seeds": "0", "pseudo": "0"},
            "ours_no_prefetch":     {"eviction": "1",  "prefetch": "1"},
            "ours_no_eviction":     {"eviction": None, "prefetch": "0", "seeds": "1", "pseudo": "1"},
            "ours_none":            {"eviction": None, "prefetch": "1"},
        }
        for name, want in expected.items():
            got = switches(name)
            check(f"{name}: " + " ".join(f"{k}={v}" for k, v in want.items()),
                  all(got[k] == v for k, v in want.items()))

        # A mode is `ours` in everything but its switches, so a change to the
        # stack reaches all seven without seven edits.
        ours = registry["ours"]
        for name in registry.ablation_arms:
            arm = registry[name]
            check_quiet = (arm.server_args == ours.server_args
                           and arm.workflow_script == ours.workflow_script
                           and arm.repo == ours.repo and arm.venv == ours.venv
                           and arm.needs_redis == ours.needs_redis)
            if not check_quiet:
                check(f"{name} inherits the ours stack", False)
        check("every mode inherits the ours stack", True)

        # prompt_seeds / pseudo_dynamic with prefetch off would be no_prefetch
        # wearing another name, and the numbers would be attributed wrongly.
        try:
            arms_mod._ablation_env("bogus", {"prefetch": False, "prompt_seeds": True})
            check("a mode that cannot mean what it says is rejected", False)
        except ValueError as exc:
            check("a mode that cannot mean what it says is rejected",
                  "no_prefetch" in str(exc))

        print("\n== a box-wide flag cannot overrule a mode")
        # The overrides are applied after the arm's own env, so without the
        # guard `BENCH_PROMPT_SEEDS=0` would turn ours_full into ours_no_seeds
        # while it still reported itself as ours_full.
        for flag, value, key, want in (
            ("BENCH_PROMPT_SEEDS", "0", "ODR_PROMPT_SEEDS", "1"),
            ("BENCH_PREFETCH", "0", "KV_EVICTION_DISABLE_PREFETCH", "0"),
        ):
            overlay = root / f"clash-{flag}.env"
            overlay.write_text(f"{flag}={value}\n")
            cfg = Config.load([base, overlay])
            check(f"{flag}={value} does not change ours_full",
                  workflow_env(cfg, "ours_full", root).get(key) == want)
            check(f"{flag}={value} still reaches plain ours",
                  workflow_env(cfg, "ours", root).get(key) != want)

        overlay = root / "clash-evict.env"
        overlay.write_text("BENCH_NODE_EVICTION=0\n")
        cfg = Config.load([base, overlay])
        check("BENCH_NODE_EVICTION=0 does not change ours_full",
              server_env(cfg, "ours_full", root).get("VLLM_NODE_EVICTION_POLICY") == "1")
        check("BENCH_NODE_EVICTION=0 still reaches plain ours",
              "VLLM_NODE_EVICTION_POLICY" not in server_env(cfg, "ours", root))

        print("\n== a value that is neither fails at load, not at boot")
        try:
            config_with("maybe")
            check("nonsense is rejected", False)
        except ConfigError as exc:
            check("nonsense is rejected", "neither true nor false" in str(exc))

    print("\n" + ("all arm-flag assertions passed" if FAILURES == 0
                  else f"{FAILURES} FAILURES"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
