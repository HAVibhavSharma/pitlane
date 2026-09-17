#!/usr/bin/env bash
#
# Three arms over two GPUs: two run side by side, the third starts on whichever
# card frees up first.
#
#   ./launch-parallel.sh            # 10 questions
#   ./launch-parallel.sh 50
#   PAIR="baseline continuum" THIRD=ours ./launch-parallel.sh 50
#   RESUME=1 RUN_ID=batch50_20260917_101500 ./launch-parallel.sh 50
#
# READ THIS BEFORE USING IT FOR A PUBLISHED NUMBER.
#
# The two arms in the pair share a host: CPU, memory bandwidth, the disk under
# LMCache's L2, and the network to Tavily and LangSmith. Their GPUs are
# separate, so token counts, cache-hit rates and prefetch usefulness are as
# clean as they were before -- but wall clock is not. Every latency an arm
# reports here was measured with a neighbour, and the third arm runs with the
# box increasingly to itself, which flatters it. `queued_s` is the number to
# watch: it is what took continuum's question 35 over the summarization timeout
# in the n50 batch, and a co-tenant pushes it up.
#
# So: use this to get token-level results and functional coverage in half the
# wall time, and to find failures early. Re-run the three arms sequentially
# with ./launch-batch.sh for any latency comparison you intend to publish.
#
# If the box is two-socket, bind each stack to the socket its GPU hangs off --
# `nvidia-smi topo -m` and `numactl -H` will tell you which -- by prefixing the
# pitlane call with `numactl --cpunodebind=N --membind=N`. That removes most of
# what is left of the interference.

set -euo pipefail

cd "$(dirname "$0")"

N="${1:-10}"
PAIR="${PAIR:-baseline continuum}"
THIRD="${THIRD:-ours}"
RESUME="${RESUME:-0}"
RUN_ID="${RUN_ID:-batch${N}_$(date +%Y%m%d_%H%M%S)}"
POLL_S="${POLL_S:-20}"

STACK_A="${STACK_A:-plan/runbooks/stack-a.env}"
STACK_B="${STACK_B:-plan/runbooks/stack-b.env}"

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
TRACE="${TRACE:-${ODR_TRACE_PATH:-${TRACE_DIR:?set TRACE_DIR or ODR_TRACE_PATH in .env, or pass TRACE=}/odr_n${N}.jsonl}}"

read -r ARM_A ARM_B <<<"$PAIR"
if [ -z "${ARM_B:-}" ]; then
  echo "PAIR needs exactly two arms, got: $PAIR" >&2
  exit 2
fi
if [ ! -f "$TRACE" ]; then
  echo "no trace at $TRACE; record it first with RECORD=1 ./launch-batch.sh $N" >&2
  exit 1
fi

RESUME_FLAG=""
[ "$RESUME" = "1" ] && RESUME_FLAG="--resume"

echo "run id : $RUN_ID${RESUME_FLAG:+ (resuming)}"
echo "pair   : $ARM_A on stack A, $ARM_B on stack B"
echo "third  : $THIRD, on whichever stack finishes first"
echo "trace  : $TRACE"
echo

# Both stacks, before either model loads. Preflight is per stack -- it checks
# that stack's card and that stack's ports -- so it has to run twice.
# `--env-extra` is a pitlane-level option, so it goes before the subcommand.
pitlane --env-extra "$STACK_A" preflight --arm "$ARM_A"
pitlane --env-extra "$STACK_B" preflight --arm "$ARM_B"

log_dir="${BENCH_ROOT:?set BENCH_ROOT in .env}/$RUN_ID/.pitlane"
mkdir -p "$log_dir"

run_arm() {  # run_arm <arm> <stack env> <log name>
  pitlane --env-extra "$2" run \
    --arms "$1" \
    --questions "batch${N}" \
    --count "$N" \
    --reps 1 \
    --trace "$TRACE" \
    --run-id "$RUN_ID" $RESUME_FLAG \
    >> "$log_dir/$3.log" 2>&1
}

echo "== $ARM_A (stack A, gpu 0) and $ARM_B (stack B, gpu 1); logs in $log_dir"
run_arm "$ARM_A" "$STACK_A" "launch-$ARM_A" & pid_a=$!
run_arm "$ARM_B" "$STACK_B" "launch-$ARM_B" & pid_b=$!

# Poll rather than `wait -n`: this needs to know *which* one finished, so the
# third arm inherits that stack's card and ports. `wait -n -p` would do it in
# one call but only on bash 5.1, and the box is not guaranteed to have it.
free_stack=""
while :; do
  if ! kill -0 "$pid_a" 2>/dev/null; then free_stack=A; break; fi
  if ! kill -0 "$pid_b" 2>/dev/null; then free_stack=B; break; fi
  sleep "$POLL_S"
done

status=0
if [ "$free_stack" = A ]; then
  wait "$pid_a" || { echo "arm $ARM_A exited non-zero" >&2; status=1; }
  free_env="$STACK_A"; other_pid="$pid_b"; other_arm="$ARM_B"
  echo "== $ARM_A done; starting $THIRD on stack A"
else
  wait "$pid_b" || { echo "arm $ARM_B exited non-zero" >&2; status=1; }
  free_env="$STACK_B"; other_pid="$pid_a"; other_arm="$ARM_A"
  echo "== $ARM_B done; starting $THIRD on stack B"
fi

run_arm "$THIRD" "$free_env" "launch-$THIRD" & pid_c=$!

wait "$other_pid" || { echo "arm $other_arm exited non-zero" >&2; status=1; }
wait "$pid_c" || { echo "arm $THIRD exited non-zero" >&2; status=1; }

echo
pitlane report "${BENCH_ROOT}/$RUN_ID"
exit "$status"
