#!/usr/bin/env bash
#
# Run every arm over one batch of N questions, one server boot per arm.
#
#   ./launch-batch.sh            # 10 questions, all measurement arms
#   ./launch-batch.sh 30
#   ARMS="baseline ours" ./launch-batch.sh 10
#   RECORD=1 ./launch-batch.sh 10        # record the trace first
#   RESUME=1 RUN_ID=batch10_20260912_015426 ./launch-batch.sh 10
#
# RESUME=1 continues a run instead of starting one: cells that finished are
# skipped and the rest are re-run, so an eight-minute boot is not paid again
# for work already on disk. Pass the RUN_ID of the run being continued -- the
# default id is a fresh timestamp, which would start a new one.
#
# The one thing this exists for is RUN_ID. `pitlane run` stamps a fresh run id
# from the clock on every invocation, so three separate calls write three
# separate run directories and `summary.md` never pivots the arms against each
# other. Passing one id to all of them is what makes the comparison a
# comparison.
#
# Batch mode is not isolated: all N questions run in one process per arm, so
# questions 2..N see a warm LMCache. The collector marks those cells
# `cache_state: warm` and refuses to average them against cold ones. For
# isolated numbers use one question and several reps instead:
#
#   pitlane run --arms baseline,continuum,ours --questions q1 --reps 3

set -euo pipefail

cd "$(dirname "$0")"

N="${1:-10}"
ARMS="${ARMS:-baseline continuum ours}"
RECORD="${RECORD:-0}"
RESUME="${RESUME:-0}"
RUN_ID="${RUN_ID:-batch${N}_$(date +%Y%m%d_%H%M%S)}"

# TRACE_DIR and BENCH_ROOT live in .env, which is shell-syntax and is what
# pitlane itself reads. Sourcing it here keeps the derived trace path in step
# with the tool rather than repeating a default that can drift.
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
# ODR_TRACE_PATH if it is set, else one named after N. Naming by N matters:
# the workflow picks its questions with `random.Random(0).sample(examples, N)`,
# and the sample for one N is not a subset of another's, so a trace can only be
# replayed at the N it was recorded at.
TRACE="${TRACE:-${ODR_TRACE_PATH:-${TRACE_DIR:?set TRACE_DIR or ODR_TRACE_PATH in .env, or pass TRACE=}/odr_n${N}.jsonl}}"

if [ "$RESUME" = "1" ]; then
  RESUME_FLAG="--resume"
else
  RESUME_FLAG=""
fi

echo "run id : $RUN_ID${RESUME_FLAG:+ (resuming)}"
echo "arms   : $ARMS"
echo "trace  : $TRACE"
echo

# Fail before the first model load rather than after it. `pitlane run` checks
# again per invocation, but a boot is ~8 minutes and a missing venv is not.
first_arm="${ARMS%% *}"
pitlane preflight --arm "$first_arm"

if [ "$RECORD" = "1" ]; then
  echo
  echo "== recording $N question(s) -> $TRACE"
  pitlane record --questions "$N" --trace "$TRACE"
elif [ ! -f "$TRACE" ]; then
  echo "no trace at $TRACE; record it first with RECORD=1 $0 $N" >&2
  exit 1
fi

# One boot per arm. Not `--arms baseline,continuum,ours`, which would interleave
# them cell by cell -- right for isolated runs, wrong here, because a batch cell
# is the whole question set and interleaving would mean three boots either way
# with no benefit.
status=0
for arm in $ARMS; do
  echo
  echo "== $arm"
  # `|| status=1` rather than letting `set -e` abort: a failed arm should not
  # take the arms after it down with it, and the run directory is still worth
  # having for the ones that finished.
  pitlane run \
    --arms "$arm" \
    --questions "batch${N}" \
    --count "$N" \
    --reps 1 \
    --trace "$TRACE" \
    --run-id "$RUN_ID" $RESUME_FLAG \
    || { echo "arm $arm exited non-zero" >&2; status=1; }
done

# Required rather than defaulted: this path has to be the one pitlane itself
# resolved, and a default here can silently disagree with the tool's -- which
# reports on an empty directory while the real results sit somewhere else.
echo
pitlane report "${BENCH_ROOT:?set BENCH_ROOT in .env}/$RUN_ID"
exit "$status"
