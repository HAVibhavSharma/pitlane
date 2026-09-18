"""Two arms, two GPUs, one box -- the four things that collide.

A second concurrent `pitlane run` shares a host with the first, and everything
it takes from the config that used to be a constant is now something the two
stacks must not agree on: the vLLM port, the LMCache port, the tmux session
names, and the card. The fourth collision is the run ledger, which both
processes read-modify-write under a shared run id.

    python3 tests/test_parallel_arms.py

No GPU and no network: the ledger test forks real processes, the rest is
config resolution.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitlane import runner, stack
from pitlane.arms import Arm
from pitlane.config import Config

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


def write_env(dir_path: Path, name: str, extra: str) -> Path:
    path = dir_path / name
    path.write_text(BASE_ENV + extra)
    return path


def check_config(root: Path) -> None:
    print("\n== one config per stack")
    base = write_env(root, "base.env", "")
    gpu1 = write_env(root, "gpu1.env", "BENCH_GPU=1\nBENCH_PORT=8001\n"
                                       "BENCH_LMCACHE_PORT=10904\n")

    default = Config.load([base])
    check("defaults are unchanged: 8000 / 10903 / gpu 0",
          (default.port, default.lmcache_port, default.gpu) == (8000, 10903, 0))
    check("default instance label is derived from the port",
          default.instance == "p8000")

    second = Config.load([base, gpu1])
    check("overlay moves the vLLM port", second.port == 8001)
    check("overlay moves the LMCache port", second.lmcache_port == 10904)
    check("overlay moves the card", second.gpu == 1)
    check("instance follows the port without being named",
          second.instance == "p8001")
    check("base_url follows the port", second.base_url("/v1")
          == "http://localhost:8001/v1")

    named = Config.load([base, write_env(root, "named.env",
                                         "BENCH_PORT=8002\nBENCH_INSTANCE=left\n")])
    check("an explicit label wins", named.instance == "left")

    print("\n== tmux sessions cannot be shared")
    check("the two stacks name different vLLM sessions",
          stack.vllm_session(default) != stack.vllm_session(second))
    check("the two stacks name different LMCache sessions",
          stack.lmcache_session(default) != stack.lmcache_session(second))
    # A box running one stack keeps the names every runbook documents.
    solo = Config.load([base, write_env(root, "solo.env", "BENCH_INSTANCE=default\n")])
    check("a lone stack is still `vllm` / `lmcache`",
          (stack.vllm_session(solo), stack.lmcache_session(solo))
          == ("vllm", "lmcache"))

    print("\n== the LMCache L2 store cannot be shared either")
    # start_lmcache wipes this path with rm -rf before starting the server, so
    # two stacks sharing it means the second one's wipe deletes the first one's
    # live backing files. Only ever bit once both arms ran an LMCache server,
    # which the baseline + continuum pair never did and two ours-derived
    # ablation arms always do.
    l2 = write_env(root, "l2.env", "LMCACHE_L2_DIR=/disk2/lmcache\n")
    a = Config.load([base, l2, write_env(root, "a.env", "BENCH_INSTANCE=default\n")])
    b = Config.load([base, l2, write_env(root, "b.env", "BENCH_INSTANCE=b\n")])
    check("the two stacks wipe different L2 directories",
          stack.lmcache_l2_dir(a) != stack.lmcache_l2_dir(b))
    check("a lone stack keeps the documented path",
          str(stack.lmcache_l2_dir(a)) == "/disk2/lmcache")
    check("stack B's path is the base plus its label",
          str(stack.lmcache_l2_dir(b)) == "/disk2/lmcache-b")
    # L1-only is a valid configuration and stays one: no disk tier to scope.
    l1_only = Config.load([base, write_env(root, "l1.env", "BENCH_INSTANCE=b\n")])
    check("no L2 dir stays no L2 dir", stack.lmcache_l2_dir(l1_only) is None)

    print("\n== the workflow is pointed at its own server")
    arm = Arm(
        name="stub", description="", repo="baseline", lmcache_server=False,
        server_env={}, server_args=[], workflow_script="x.py", workflow_args=[],
        workflow_env={"LANGGRAPH_VLLM_ECHO_BASE_URL": "http://localhost:{port}"},
    )
    resolved = arm.resolved_env("workflow", repo=Path("/tmp"), cell=Path("/tmp"),
                                model="m", port=8001)
    check("{port} resolves to this stack's port",
          resolved["LANGGRAPH_VLLM_ECHO_BASE_URL"] == "http://localhost:8001")


def check_ledger(root: Path) -> None:
    """Fork writers at the same cell count and demand every entry survives.

    Unlocked this loses entries essentially every time at this width: each
    child reads the ledger before the others have written, so the last rename
    wins and the rest of the run's cells vanish from it.
    """
    print("\n== the ledger survives concurrent arms")
    run_dir = root / "run"
    run_dir.mkdir()
    writers = 8
    per_writer = 6

    children = []
    for writer in range(writers):
        pid = os.fork()
        if pid == 0:
            try:
                for index in range(per_writer):
                    runner._record_cell(
                        run_dir, f"arm{writer}", f"q{index}", 1,
                        exit_code=0, aborted=False,
                    )
            finally:
                os._exit(0)
        children.append(pid)
    for pid in children:
        os.waitpid(pid, 0)

    cells = runner._read_ledger(run_dir)
    check(f"all {writers * per_writer} entries present (got {len(cells)})",
          len(cells) == writers * per_writer)
    check("no writer's temp file was left behind",
          not list(run_dir.glob(".pitlane-progress.json.tmp*")))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        check_config(root)
        check_ledger(root)
    print("\n" + ("all parallel-arm assertions passed" if FAILURES == 0
                  else f"{FAILURES} FAILURES"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
