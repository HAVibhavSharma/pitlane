#!/usr/bin/env bash
#
# Run every arm over one batch of N questions, one server boot per arm.
#
#   ./launch-batch.sh            # 10 questions, all measurement arms
#   ./launch-batch.sh 30
#   ARMS="baseline ours" ./launch-batch.sh 10
#   RECORD=1 ./launch-batch.sh 10        # record the trace first
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
RUN_ID="${RUN_ID:-batch${N}_$(date +%Y%m%d_%H%M%S)}"

# TRACE_DIR and BENCH_ROOT live in .env, which is shell-syntax and is what
# pitlane itself reads. Sourcing it here keeps the derived trace path in step
# with the tool rather than repeating a default that can drift.
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
TRACE="${TRACE:-${TRACE_DIR:?set TRACE_DIR in .env or pass TRACE=}/odr_n${N}.jsonl}"

echo "run id : $RUN_ID"
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
    --reps 1 \
    --trace "$TRACE" \
    --run-id "$RUN_ID" || { echo "arm $arm exited non-zero" >&2; status=1; }
done

echo
pitlane report "${BENCH_ROOT:-/disk2/vibhav/bench}/$RUN_ID"
exit "$status"
