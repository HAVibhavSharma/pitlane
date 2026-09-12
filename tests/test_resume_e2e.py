"""Interrupt and resume, end to end, without a GPU.

The thing being tested is expensive to reproduce honestly: a 50-question batch
against the real stack is eleven hours, and the behaviour only shows up when a
Ctrl-C lands in the middle of a question. So the stack is replaced and nothing
else is -- `workflow.run`, `metrics.collect` and the runner's bookkeeping are
the real functions, driving a real child process over a real signal.

What stands in for vLLM is a stub workflow that sleeps `--latency` seconds per
request instead of generating tokens, and writes the same per-request stats
rows the server's `FileStatLogger` writes. That is all the collector ever reads
from a run, so a cell assembled this way is the shape of a real one.

The question and request counts come from a recorded trace when one is given,
so the run has the shape of the batch it stands for:

    python3 tests/test_resume_e2e.py
    python3 tests/test_resume_e2e.py --trace ~/run_50.jsonl --questions 6
    python3 tests/test_resume_e2e.py --latency 0.2 --interrupt-during 3

Only the job ids and per-job request counts are read from the trace, streamed a
line at a time -- the file is hundreds of megabytes and is never held.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitlane import metrics as metrics_mod  # noqa: E402
from pitlane import report, runner, workflow  # noqa: E402
from pitlane.arms import Arm  # noqa: E402

# -- the stand-in for the workflow -----------------------------------------
#
# Written out rather than imported: it has to be a separate process in its own
# session for the signal path to be the real one, and it has to be startable by
# `workflow.run` the way ODR is.
STUB = '''\
"""Stands in for run_evaluate.py: N questions, each K requests of X seconds."""
import argparse, asyncio, json, os, signal, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--max-queries", type=int, default=1)
parser.add_argument("--completions-per-query", type=int, default=1)
parser.add_argument("--completed-log", default=None)
parser.add_argument("--plan", required=True)
parser.add_argument("--latency", type=float, default=0.05)
parser.add_argument("--concurrency", type=int, default=2)
args = parser.parse_args()

PLAN = json.loads(Path(args.plan).read_text())          # [[job_id, requests], ...]
STATS = Path(os.environ["VLLM_REQUEST_STATS_DIR"])
STATS.mkdir(parents=True, exist_ok=True)
# One file per process, exactly as a server boot produces one -- so a resumed
# cell ends up with several, which is what the collector has to glob.
ROWS = STATS / f"finished_requests_engine0_{int(time.time()*1000)}.jsonl"


def emit(job_id, index):
    """One finished-request row, in the shape FileStatLogger writes."""
    now = time.time()
    with ROWS.open("a") as handle:
        handle.write(json.dumps({
            "request_id": f"chatcmpl-j{job_id}-{index}",
            "job_id": str(job_id),
            "langgraph_node": "researcher" if index else "write_research_brief",
            "arrival_ts": now - args.latency,
            "finish_ts": now,
            "queued_time": 0.0,
            "prefill_time": args.latency / 2,
            "decode_time": args.latency / 2,
            "num_prompt_tokens": 1000 + index,
            "num_cached_tokens": 100,
            "num_local_cached_tokens": 100,
            "num_external_cached_tokens": 0,
            "num_generation_tokens": 10,
        }) + "\\n")
        handle.flush()
        os.fsync(handle.fileno())


async def main():
    log = Path(args.completed_log) if args.completed_log else None
    done = set()
    if log is not None and log.exists():
        done = {int(x) for x in log.read_text().split() if x.strip().isdigit()}
    print(f"stub: resuming, {len(done)} question(s) already done", flush=True)

    jobs = [
        {"index": i + 1, "job_id": job_id, "requests": requests,
         "done": (i + 1) in done}
        for i, (job_id, requests) in enumerate(PLAN[:args.max_queries])
    ]

    stop = False

    def request_stop():
        nonlocal stop
        if stop:
            print("stub: second interrupt, exiting now", flush=True)
            raise KeyboardInterrupt
        stop = True
        print("stub: interrupt received, finishing the question(s) in flight",
              flush=True)

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_stop)
        except (NotImplementedError, RuntimeError, ValueError):
            pass

    async def question(job):
        print(f"stub: question {job['index']} started", flush=True)
        for index in range(job["requests"]):
            await asyncio.sleep(args.latency)      # a vLLM call
            emit(job["job_id"], index)
        print(f"stub: question {job['index']} finished", flush=True)
        return job

    active = {}

    def refill():
        while len(active) < args.concurrency:
            nxt = next((j for j in jobs
                        if not j["done"]
                        and j["index"] not in {a["index"] for a in active.values()}),
                       None)
            if nxt is None:
                return
            nxt["done"] = True          # claimed, not yet finished
            active[asyncio.create_task(question(nxt))] = nxt

    refill()
    while active:
        finished, _ = await asyncio.wait(active.keys(),
                                         return_when=asyncio.FIRST_COMPLETED)
        for task in finished:
            job = active.pop(task)
            task.result()
            if log is not None:
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a") as handle:
                    handle.write(f"{job['index']}\\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        if not stop:
            refill()

    if stop:
        print("stub: stopped at a question boundary", flush=True)
    return 0


sys.exit(asyncio.run(main()))
'''


def read_plan(trace: Path | None, limit: int) -> list[tuple[str, int]]:
    """`[(job_id, request count)]` for the first `limit` questions of a trace.

    Streamed a line at a time: the traces this stands in for are hundreds of
    megabytes, and the only things wanted from them are how many questions
    there were and how many calls each one made.
    """
    if trace is None or not trace.exists():
        # No trace: a plausible shape, so the harness is useful on its own.
        return [(str(i), 8 + (i % 5) * 4) for i in range(1, limit + 1)]
    counts: dict[str, int] = {}
    with trace.open() as handle:
        for line in handle:
            try:
                job = str(json.loads(line).get("job_id"))
            except (json.JSONDecodeError, AttributeError):
                continue
            counts[job] = counts.get(job, 0) + 1
            if len(counts) > limit:
                # Every job after the limit is unread; the last one counted is
                # dropped because its own count is still incomplete.
                counts.pop(job)
                break
    return list(counts.items())[:limit]


class StubConfig:
    """The slice of Config that `workflow.run` touches."""

    def __init__(self, repo: Path, trace: Path) -> None:
        class Paths:
            workflow_repo = repo
            workflow_venv = None
        self.paths = Paths()
        # Left unset so the date comes from the trace, the way a real run
        # resolves it -- which also exercises that path against a real file.
        self.env: dict[str, str] = {}
        self.model_name = "stub-model"
        self.trace_path = trace
        self.run_dir = repo / "runs"

    def base_url(self, suffix: str = "") -> str:
        return f"http://127.0.0.1:8000{suffix}"


def make_arm(script: Path, plan: Path, latency: float) -> Arm:
    return Arm(
        name="stub", description="", repo="baseline", lmcache_server=False,
        server_env={}, server_args={}.get("x", []) or [],
        workflow_script=str(script),
        workflow_args=["--plan", str(plan), "--latency", str(latency)],
        # What `start_server` would have exported. `{cell}` is expanded by the
        # arm the same way it is for a real one, so the stub writes its rows
        # where the collector looks for them.
        workflow_env={"VLLM_REQUEST_STATS_DIR": "{cell}/stats"},
    )


def interrupt_when_running(log: Path, question: int, settle_s: float,
                           timeout_s: float = 120.0) -> threading.Thread:
    """SIGINT this process once `question` is under way, not before.

    Watching the workflow's own output rather than sleeping a computed
    interval: the point of the test is that the signal lands *during* a
    question, and a fixed delay gets that wrong as soon as the latency or the
    question count changes -- silently passing on a run where the interrupt
    arrived between questions and nothing was actually drained.
    """
    needle = f"stub: question {question} started"

    def fire() -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                if needle in log.read_text():
                    break
            except OSError:
                pass
            time.sleep(0.02)
        # Far enough in to be mid-question, not so far as to be past it.
        time.sleep(settle_s)
        os.kill(os.getpid(), signal.SIGINT)

    thread = threading.Thread(target=fire, daemon=True)
    thread.start()
    return thread


def rows_in(cell: Path) -> list[dict]:
    out = []
    for path in sorted((cell / "stats").glob("finished_requests_engine*.jsonl")):
        out += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", type=Path,
                        default=Path.home() / "run_50.jsonl")
    parser.add_argument("--questions", type=int, default=6)
    parser.add_argument("--latency", type=float, default=0.05,
                        help="seconds one vLLM call takes in the stub")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--interrupt-during", type=int, default=3,
                        help="interrupt while roughly this question is running")
    parser.add_argument("--keep", action="store_true",
                        help="leave the scratch directory in place")
    args = parser.parse_args()

    import tempfile
    root = Path(tempfile.mkdtemp(prefix="pitlane-resume-"))
    repo = root / "repo"
    (repo / "tests").mkdir(parents=True)
    # `detect()` sniffs this for the flags it accepts, so the stub has to be
    # discoverable as the repo's evaluation script.
    script = repo / "tests" / "run_evaluate.py"
    script.write_text(STUB)

    plan = read_plan(args.trace, args.questions)
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan))
    requests_total = sum(count for _, count in plan)

    trace_note = args.trace if args.trace.exists() else "(synthetic)"
    print(f"trace      : {trace_note}")
    print(f"questions  : {len(plan)}  requests: {requests_total}")
    print(f"latency    : {args.latency}s per request "
          f"-> ~{requests_total * args.latency / args.concurrency:.1f}s of work")
    print()

    config = StubConfig(repo, args.trace)
    arm = make_arm(script, plan_path, args.latency)
    cell = root / "run" / "stub" / f"batch{len(plan)}" / "rep1"
    run_dir = root / "run"
    log = runner._questions_log(run_dir, "stub", f"batch{len(plan)}", 1)

    # -- attempt 1: interrupted mid-question -------------------------------
    print(f"== attempt 1, interrupting once question "
          f"{args.interrupt_during} is running")
    interrupt_when_running(cell / "workflow.log", args.interrupt_during,
                           settle_s=args.latency * 2)
    first = workflow.run(config, arm, cell, question_id=f"batch{len(plan)}",
                         count=len(plan), trace_mode="pinned",
                         completed_log=log)
    done_first = sorted(int(x) for x in log.read_text().split()) if log.exists() else []
    rows_first = rows_in(cell)
    print(f"   exit={first.exit_code} drained={first.drained} "
          f"questions done={done_first} rows={len(rows_first)}")

    assert first.drained, "the interrupt was not forwarded as a drain"
    assert first.exit_code == 0, f"the stub did not exit cleanly: {first.exit_code}"
    assert done_first, "nothing was recorded; the question in flight was lost"
    assert len(done_first) < len(plan), (
        "everything finished before the interrupt -- raise --latency or "
        "lower --interrupt-during so the signal lands mid-question")

    # Every question recorded as done must have written all of its rows: that
    # is what "finish the question in flight" has to mean.
    by_job = {}
    for row in rows_first:
        by_job[row["job_id"]] = by_job.get(row["job_id"], 0) + 1
    for index in done_first:
        job_id, expected = plan[index - 1]
        assert by_job.get(job_id) == expected, (
            f"question {index} was recorded done with {by_job.get(job_id)} of "
            f"{expected} requests")
    print("   every recorded question wrote all of its requests")

    # A drained cell is not a finished one.
    runner._record_cell(run_dir, "stub", f"batch{len(plan)}", 1,
                        exit_code=first.exit_code, aborted=first.aborted,
                        drained=first.drained)
    assert not runner.completed(run_dir, "stub", f"batch{len(plan)}", 1, cell), \
        "a drained cell was marked complete, so a resume would skip it"
    print("   cell recorded as unfinished")

    stamp_first = (cell / "question_started_ts").read_text().strip()

    # -- attempt 2: resume --------------------------------------------------
    print("\n== attempt 2, resuming")
    second = workflow.run(config, arm, cell, question_id=f"batch{len(plan)}",
                          count=len(plan), trace_mode="pinned",
                          completed_log=log)
    done_second = sorted(int(x) for x in log.read_text().split())
    rows_second = rows_in(cell)
    print(f"   exit={second.exit_code} drained={second.drained} "
          f"questions done={done_second} rows={len(rows_second)}")

    assert second.exit_code == 0
    assert not second.drained
    assert done_second == list(range(1, len(plan) + 1)), done_second

    # The resumed process must not have re-run what was already done.
    # The log holds both attempts -- it is appended, which is the point -- so
    # the second attempt's section is what follows the *last* start banner.
    stub_log = (cell / "workflow.log").read_text()
    resumed_section = stub_log.rsplit("stub: resuming", 1)[-1]
    restarted = [i for i in done_first
                 if f"stub: question {i} started" in resumed_section]
    assert not restarted, f"resume re-ran questions {restarted}"
    print("   resume ran only the questions that were left")

    # -- the cell is one cell ----------------------------------------------
    assert (cell / "question_started_ts").read_text().strip() == stamp_first, \
        "the window start was restamped; the first attempt's rows fall outside it"
    assert len(list((cell / "stats").glob("*.jsonl"))) == 2, \
        "each attempt should have written its own stats file"

    collected = metrics_mod.collect(
        cell, arm="stub", question_id=f"batch{len(plan)}", rep=1,
        t0=second.started_ts, t1=second.finished_ts,
    )
    print(f"\n== collected: requests={collected.requests} "
          f"query_tokens={collected.query_tokens} jobs="
          f"{len({r['job_id'] for r in collected.per_request})}")
    assert collected.requests == requests_total, (
        f"{collected.requests} of {requests_total} requests collected; the "
        f"two attempts did not assemble into one cell")
    assert len({r["job_id"] for r in collected.per_request}) == len(plan)

    # What `run_cell` writes, and what `completed()` requires alongside the
    # ledger entry: an entry without the collection it vouches for is a cell
    # whose artifacts were removed, and must be re-run.
    report.write_metrics(cell, collected)
    report.append_row(run_dir / "results.csv", collected)
    report.append_request_rows(run_dir / "requests.csv", collected)
    report.write_by_question(run_dir)
    questions = report.load_rows(run_dir / "by_question.csv")
    assert len(questions) == len(plan), questions
    print(f"   by_question.csv: {len(questions)} rows, one per question")

    runner._record_cell(run_dir, "stub", f"batch{len(plan)}", 1,
                        exit_code=0, aborted=False, drained=False)
    assert runner.completed(run_dir, "stub", f"batch{len(plan)}", 1, cell)
    print("   cell now recorded as complete")

    # The bookkeeping stays out of the results.
    assert sorted(p.name for p in run_dir.iterdir() if p.name.startswith(".")) == [
        ".pitlane", ".pitlane-progress.json"], sorted(p.name for p in run_dir.iterdir())
    assert not any("status" in p.name for p in cell.iterdir()), sorted(
        p.name for p in cell.iterdir())

    print("\nall resume assertions passed")
    if args.keep:
        print(f"scratch: {root}")
    else:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
