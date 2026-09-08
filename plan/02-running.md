# Running an evaluation — 1, 10 and 30 questions

Same three steps every time: **preflight → record → run**. What changes with
size is whether each question gets its own server boot.

```bash
export PITLANE=/home/vibhav/pitlane          # this repo
cd $PITLANE && pitlane preflight --arm ours  # must be clean before anything
```

Set the workflow under test in `plan/runbooks/common.env`:

```bash
WORKFLOW_REPO=/home/vibhav/open_deep_research   # or .../swe-agent
ODR_TRACE_PATH=/disk2/vibhav/traces/<name>.jsonl
```

**One trace per (workflow, question count).** Never reuse a trace across
either — see "Why N is part of the identity" below.

---

## ODR

### 1 question

```bash
pitlane record --questions 1 --trace /disk2/vibhav/traces/odr_n1.jsonl
pitlane run --arms baseline,continuum,ours --questions q1 --reps 3 \
            --trace /disk2/vibhav/traces/odr_n1.jsonl
```

9 boots (3 arms × 3 reps), each fully isolated: LMCache wiped, server
restarted, metrics epoch reset. This is the mode the numbers in the write-up
come from.

### 10 questions

```bash
pitlane record --questions 10 --trace /disk2/vibhav/traces/odr_n10.jsonl
pitlane run --arms baseline,continuum,ours --questions q1..q10 --reps 1 \
            --trace /disk2/vibhav/traces/odr_n10.jsonl
```

⚠️ **Needs the ODR selection flag** (below). Without it `--max-queries 10`
runs all ten in one process, so the cells are not isolated: questions 2–10 see
a warm LMCache. Until then, run 10 questions as a **batch** and accept
`cache_state: warm` on all but the first — the reporter keeps warm and cold
cells apart and will not average them.

### 30 questions

Batch, not isolated — 30 questions × 3 arms × one boot each is ~90 model loads
(≈8 min apiece, so ~12 h of pure loading before any inference). Use one boot
per arm:

```bash
pitlane record --questions 30 --trace /disk2/vibhav/traces/odr_n30.jsonl
for arm in baseline continuum ours; do
  pitlane run --arms $arm --questions batch30 --reps 1 \
              --trace /disk2/vibhav/traces/odr_n30.jsonl
done
```

Treat 30 as a throughput/aggregate number, not as 30 isolated measurements.
Report it separately from the 1-question results.

---

## swe-agent

Same commands; only the repo and the ids change. swe-agent's
`tests/run_evaluate.py` accepts `--instance-ids`, so pitlane can select a
single instance and **every size stays isolated**:

```bash
# in common.env: WORKFLOW_REPO=/home/vibhav/swe-agent
pitlane record --questions 10 --trace /disk2/vibhav/traces/swe_n10.jsonl
pitlane run --arms baseline,continuum,ours \
            --questions django__django-11099,astropy__astropy-12907 --reps 1 \
            --trace /disk2/vibhav/traces/swe_n10.jsonl
```

Get the instance ids the record run actually used from the trace:

```bash
python3 -c "
import json,collections,sys
print(sorted({json.loads(l)['job_id'] for l in open(sys.argv[1])}))
" /disk2/vibhav/traces/swe_n10.jsonl
```

For 30, pass 30 ids (or split across several `pitlane run` invocations —
results append to the same `results.csv` when you pass the same `--run-id`).

---

## Why N is part of the trace identity

ODR picks its questions with `random.Random(0).sample(english_examples, N)`
(`tests/run_evaluate.py:388`) — a **sample of size N**, not the first N. The
set for N=1 is not a subset of the set for N=10. A trace recorded at N=10 and
replayed at N=1 therefore misses on every request, and `ODR_TRACE_ON_MISS=strict`
aborts the run. Name traces after the workflow and N, as above.

## Prerequisite for isolated ODR runs at N > 1

ODR has no way to run one specific question — only "sample N". Add one flag to
`tests/run_evaluate.py` and every size becomes isolated:

```
--query-ids ID[,ID...]   filter _select_examples() output by benchmark id
```

pitlane already knows how to use it: `workflow.ODR` just needs its
`select_flag` set to `--query-ids`, exactly as the swe-agent adapter does with
`--instance-ids`.

## Reading the results

```bash
pitlane report /disk2/vibhav/bench/<run_id>     # prints and rewrites summary.md
```

A cell is invalid — do not report it — if `summary.md` marks it `trace miss`,
or if `metrics.json` warns about zero prefetches on `ours`. Re-run that cell.
