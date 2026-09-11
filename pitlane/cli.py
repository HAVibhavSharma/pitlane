"""pitlane command line.

    pitlane preflight [--arm ours]
    pitlane arms
    pitlane record --questions 10
    pitlane run --arms baseline,continuum,ours --questions q1,q2 --reps 3
    pitlane collect <cell-dir>
    pitlane report <run-dir>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pitlane import arms as arms_mod
from pitlane import metrics as metrics_mod
from pitlane import preflight, report, runner, stack
from pitlane.config import Config, ConfigError

_ROOT = Path(__file__).resolve().parents[1]
_PLAN = _ROOT / "plan"
# Read in increasing order of precedence -- `Config.load` applies each file over
# the last. Secrets, then the shared config that ships with the runbooks so a
# by-hand run and an automated one read the same defaults, then this box's own
# `.env`, which is the file the operator actually edits and therefore wins.
# A missing file is skipped, so none of the three is required.
DEFAULT_ENV_FILES = [
    Path.home() / ".bench.env",
    _PLAN / "runbooks" / "common.env",
    _ROOT / ".env",
]


def _load_config(args: argparse.Namespace) -> Config:
    files = [Path(p) for p in (args.env or [])] or DEFAULT_ENV_FILES
    config = Config.load([f for f in files])
    if getattr(args, "trace", None):
        config.trace_path = Path(args.trace)
    if getattr(args, "run_id", None):
        config.run_id = args.run_id
    return config


def cmd_arms(args: argparse.Namespace) -> int:
    registry = arms_mod.load()
    for arm in registry:
        marker = "record" if arm.name == "record" else "measure"
        lmcache = "lmcache" if arm.lmcache_server else "no-lmcache"
        print(f"{arm.name:10s} [{marker}] [{lmcache}] {arm.description}")
        print(f"{'':10s} repo={arm.repo} script={arm.workflow_script} "
              f"{' '.join(arm.workflow_args)}")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    config = _load_config(args)
    checks = preflight.run(config, arm=args.arm)
    for check in checks:
        print(check)
    blockers = preflight.blockers(checks)
    if blockers:
        print(f"\n{len(blockers)} blocker(s); refusing to run.", file=sys.stderr)
    return 1 if blockers else 0


def cmd_record(args: argparse.Namespace) -> int:
    config = _load_config(args)
    registry = arms_mod.load()
    if config.trace_path.exists() and config.trace_path.stat().st_size and not args.force:
        print(f"{config.trace_path} already exists; pass --force to record over it",
              file=sys.stderr)
        return 1
    with _teardown_on_exit(config, args.keep_stack):
        result = runner.run_cell(
            config, registry["record"], question_id="record", rep=1,
            trace_mode="record", count=args.questions, keep_stack=args.keep_stack,
            dry_run=args.dry_run,
        )
    print(f"trace: {config.trace_path}")
    return result.exit_code


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args)
    registry = arms_mod.load()
    arm_names = args.arms.split(",") if args.arms else registry.measurement_arms
    for name in arm_names:
        registry[name]  # fail early on a typo, before any GPU work
    questions = args.questions.split(",")

    if not args.dry_run:
        checks = preflight.run(config, arm=arm_names[0])
        blockers = preflight.blockers(checks)
        if blockers and not args.skip_preflight:
            for check in blockers:
                print(check, file=sys.stderr)
            return 1

    with _teardown_on_exit(config, args.keep_stack):
        results = runner.run_matrix(
            config, registry,
            arms=arm_names, questions=questions, reps=args.reps, count=args.count,
            dry_run=args.dry_run, keep_stack=args.keep_stack,
        )
    path = report.write_summary(config.run_dir)
    print(f"\n{len(results)} cell(s) -> {config.run_dir}")
    print(path.read_text())
    return 0 if all(r.exit_code == 0 for r in results) else 1


def cmd_collect(args: argparse.Namespace) -> int:
    cell = Path(args.cell)
    collected = metrics_mod.collect(cell, arm=args.arm or cell.parts[-3],
                                    question_id=args.question or cell.parts[-2])
    report.write_metrics(cell, collected)
    print((cell / "metrics.json").read_text())
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    print(report.summary(run_dir / "results.csv"))
    report.write_summary(run_dir)
    return 0


def _teardown_on_exit(config: Config, keep_stack: bool):
    """Context manager: leave no server running unless asked to.

    An interrupt is where this matters. `stop_server` kills the tmux session
    first and only then waits, so a Ctrl-C anywhere after that point loses the
    only handle anything had on the process -- pitlane exits, vLLM and LMCache
    keep running, and the card stays occupied with nothing on screen to say so.
    The next run then fails preflight on a port it started itself.

    `--keep-stack` is honoured: leaving the stack up is a deliberate debugging
    choice, and an interrupt should not silently reverse it.
    """
    from contextlib import contextmanager

    @contextmanager
    def guard():
        try:
            yield
        except KeyboardInterrupt:
            if keep_stack:
                print("\ninterrupted; leaving the stack up (--keep-stack)",
                      file=sys.stderr)
                raise
            print("\ninterrupted; tearing down the stack", file=sys.stderr)
            try:
                stack.down(config)
            except Exception as exc:  # noqa: BLE001 - already on the way out
                print(f"teardown failed: {exc}; run `pitlane down`",
                      file=sys.stderr)
            raise
    return guard()


def cmd_down(args: argparse.Namespace) -> int:
    """Tear down whatever a previous run left behind."""
    stack.down(_load_config(args))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pitlane", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", action="append", help="env file (repeatable)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("arms", help="list configured stacks").set_defaults(func=cmd_arms)
    sub.add_parser(
        "down", help="kill any vLLM/LMCache pitlane left running",
    ).set_defaults(func=cmd_down)

    pre = sub.add_parser("preflight", help="check the machine can host a run")
    pre.add_argument("--arm", help="include arm-specific checks (redis, lmcache port)")
    pre.set_defaults(func=cmd_preflight)

    rec = sub.add_parser("record", help="record the trace on the baseline stack")
    rec.add_argument("--questions", type=int, default=10)
    rec.add_argument("--trace", help="override ODR_TRACE_PATH")
    rec.add_argument("--force", action="store_true", help="record over an existing trace")
    rec.add_argument("--keep-stack", action="store_true")
    rec.add_argument("--dry-run", action="store_true")
    rec.add_argument("--run-id")
    rec.set_defaults(func=cmd_record)

    run = sub.add_parser("run", help="measure arms against the recorded trace")
    run.add_argument("--arms", help="comma separated; default every measurement arm")
    run.add_argument("--questions", required=True, help="comma separated question ids")
    run.add_argument("--reps", type=int, default=1)
    run.add_argument(
        "--count", type=int,
        help="questions per cell; defaults to the N in a `batchN` id, else 1. "
             "Must match the N the trace was recorded at -- the workflow picks "
             "its questions by sampling N, so a different N is a different set.",
    )
    run.add_argument("--trace", help="override ODR_TRACE_PATH")
    run.add_argument("--keep-stack", action="store_true", help="leave the last stack up")
    run.add_argument("--skip-preflight", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--run-id")
    run.set_defaults(func=cmd_run)

    col = sub.add_parser("collect", help="recompute metrics.json for one cell")
    col.add_argument("cell")
    col.add_argument("--arm")
    col.add_argument("--question")
    col.set_defaults(func=cmd_collect)

    rep = sub.add_parser("report", help="rebuild summary.md from results.csv")
    rep.add_argument("run_dir")
    rep.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        # The teardown has already run; this is only to exit without dumping a
        # traceback whose top frame is `time.sleep`.
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
