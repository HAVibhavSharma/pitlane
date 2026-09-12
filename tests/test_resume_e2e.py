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
import argparse, asyncio, itertools, json, os, signal, sys, time
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
    done, highest = set(), 0
    if log is not None and log.exists():
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            done.add(entry["question"])
            for jid in entry["job_ids"]:
                highest = max(highest, int(jid))
    print(f"stub: resuming, {len(done)} question(s) already done", flush=True)

    # As run_evaluate does: a process-local counter, continued past whatever a
    # previous attempt used. Modelled because getting this wrong is invisible
    # in the totals -- the rows are all there, grouped under the wrong
    # question.
    counter = itertools.count(highest + 1)

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
        job["assigned"] = str(next(counter))
        print(f"stub: question {job['index']} started", flush=True)
        for index in range(job["requests"]):
            await asyncio.sleep(args.latency)      # a vLLM call
            emit(job["assigned"], index)
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
                    handle.write(json.dumps({"question": job["index"],
                                             "job_ids": [job["assigned"]]}) + "\\n")
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


def read_done(log: Path) -> list[int]:
    """Question indices the workflow has recorded as finished."""
    if not log.exists():
        return []
    return sorted(json.loads(line)["question"]
                  for line in log.read_text().splitlines() if line.strip())


def read_job_ids(log: Path) -> dict[int, list[str]]:
    """`question index -> the job ids it ran under`."""
    out: dict[int, list[str]] = {}
    if not log.exists():
        return out
    for line in log.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            out[entry["question"]] = [str(j) for j in entry["job_ids"]]
    return out


def rows_in(cell: Path) -> list[dict]:
    out = []
    for path in sorted((cell / "stats").glob("finished_requests_engine*.jsonl")):
        out += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


def check_hard_kill(root: Path, latency: float) -> None:
    """A question killed outright is re-run, not double-counted.

    The second Ctrl-C, an OOM abort and a machine that goes away all land here:
    the question in flight is lost with its rows already on disk. It is not
    recorded as finished, so the next attempt redoes it -- and without pruning
    the cell keeps both halves, inflating that one question's requests, tokens
    and latencies while the totals look plausible.
    """
    import subprocess

    repo = root / "hardkill" / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "run_evaluate.py").write_text(STUB)
    plan = [("1", 6), ("2", 6), ("3", 6)]
    plan_path = root / "hardkill" / "plan.json"
    plan_path.write_text(json.dumps(plan))

    config = StubConfig(repo, root / "no-trace")
    arm = make_arm(repo / "tests" / "run_evaluate.py", plan_path, latency)
    arm = type(arm)(**{**arm.__dict__,
                       "workflow_args": arm.workflow_args + ["--concurrency", "1"]})
    cell = root / "hardkill" / "cell"
    log = root / "hardkill" / "questions.log"

    def kill_during(question: int) -> threading.Thread:
        needle = f"stub: question {question} started"

        def fire() -> None:
            deadline = time.time() + 60
            while time.time() < deadline:
                try:
                    if needle in (cell / "workflow.log").read_text():
                        break
                except OSError:
                    pass
                time.sleep(0.02)
            time.sleep(latency * 2)     # partway through, rows already written
            subprocess.run(["pkill", "-KILL", "-f",
                            "run_evaluate.py --max-queries 3"],
                           capture_output=True)

        thread = threading.Thread(target=fire, daemon=True)
        thread.start()
        return thread

    kill_during(2)
    first = workflow.run(config, arm, cell, question_id="b3", count=3,
                         trace_mode="pinned", completed_log=log)
    on_disk = len(rows_in(cell))
    done = read_done(log)
    print(f"\n== hard kill: exit={first.exit_code} done={done} "
          f"rows left on disk={on_disk}")
    assert first.exit_code != 0, "the stub was not actually killed"
    assert on_disk > sum(c for _, c in plan[:len(done)]), (
        "no partial rows were left behind, so this run does not exercise the "
        "case -- raise the latency")

    workflow.run(config, arm, cell, question_id="b3", count=3,
                 trace_mode="pinned", completed_log=log)
    collected = metrics_mod.collect(cell, arm="stub", question_id="b3",
                                    t0=0, t1=1e12)
    counts: dict[str, int] = {}
    for row in collected.per_request:
        counts[row["job_id"]] = counts.get(row["job_id"], 0) + 1
    ids = [row["request_id"] for row in collected.per_request]
    assert not {i for i in ids if ids.count(i) > 1}, "duplicate rows survived"
    assert collected.requests == sum(c for _, c in plan), (
        f"{collected.requests} rows, expected {sum(c for _, c in plan)}: the "
        f"killed question was counted twice")
    assert sorted(counts.values()) == [6, 6, 6], counts
    print(f"   after resume: {collected.requests} rows, per question "
          f"{sorted(counts.values())}, no duplicates")
    print("\nall hard-kill assertions passed")


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
    done_first = read_done(log)
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
    assigned = read_job_ids(log)
    for index in done_first:
        _, expected = plan[index - 1]
        got = sum(by_job.get(j, 0) for j in assigned[index])
        assert got == expected, (
            f"question {index} was recorded done with {got} of {expected} "
            f"requests")
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
    done_second = read_done(log)
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

    # The total alone would pass on a cell that lost one request and counted
    # another twice, which is exactly the shape a bad resume produces: a
    # question re-run from the start duplicates its early rows and the
    # interrupted one is short.
    ids = [row["request_id"] for row in collected.per_request]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicated, f"duplicate request rows: {duplicated[:5]}"
    counts: dict[str, int] = {}
    for row in collected.per_request:
        counts[row["job_id"]] = counts.get(row["job_id"], 0) + 1
    assigned = read_job_ids(log)
    short = {}
    for index, (_, expected) in enumerate(plan, 1):
        got = sum(counts.get(j, 0) for j in assigned[index])
        if got != expected:
            short[index] = (got, expected)
    assert not short, f"questions with the wrong number of requests: {short}"

    # Every question must own its own ids. A resumed process that restarts the
    # counter relabels its questions with ids the first attempt already used,
    # and two different questions are then grouped as one -- with the row
    # totals still perfectly correct, which is what makes it worth asserting.
    seen: dict[str, int] = {}
    for index, question_ids in assigned.items():
        for job_id in question_ids:
            assert job_id not in seen, (
                f"job id {job_id} used by questions {seen[job_id]} and {index}")
            seen[job_id] = index
    assert len(set(counts)) == len(plan), (
        f"{len(set(counts))} distinct job ids for {len(plan)} questions")
    print(f"   {len(ids)} rows, no duplicates, every question complete")

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

    check_hard_kill(root, max(args.latency, 0.1))
    if args.keep:
        print(f"scratch: {root}")
    else:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
