# Ablation smoke test — one question per mode

**For an agent running on the benchmark box.** Verify that every ablation arm
actually switches what it claims to switch, before anyone spends eleven hours
on a real batch. One question per arm.

You need only this repo. Do not edit `open_deep_research` or the vLLM forks —
if something is wrong there, report it and stop.

**What "working" means here:** not that the numbers are good, but that each
mode's flags *reached the process that reads them*, and that the evidence for
each mechanism is present when it should be and absent when it should not.
A mode that silently runs as another mode is the failure this is looking for.

---

## 0. Before booting anything

```bash
cd ~/pitlane                      # or wherever this repo lives
git pull
set -a; . ~/.bench.env; . plan/runbooks/common.env; set +a
```

Pull the other two repos too. A stale `open_deep_research` has no
`prompt_seeds.py` and a stale langgraph fork resolves the wrong date — both
would make modes indistinguishable:

```bash
git -C "$WORKFLOW_REPO" pull
test -f "$WORKFLOW_REPO/src/open_deep_research/prompt_seeds.py" || echo "STALE WORKFLOW"
```

The langgraph fork must be the one in `$WORKFLOW_VENV`. Confirm it reads the
frozen date:

```bash
"$WORKFLOW_VENV/bin/python" - <<'EOF'
from langgraph.pregel._prediction import _default_prefix_resolvers
import os
os.environ["ODR_FROZEN_DATE"] = "2026-07-20"
got = _default_prefix_resolvers()["get_today_str"]()
print(got, "OK" if got == "Mon Jul 20, 2026" else "STALE LANGGRAPH FORK")
EOF
```

Then the flag plumbing, which costs nothing and catches most mistakes:

```bash
python3 tests/test_arm_flags.py     # expect: all arm-flag assertions passed
pitlane arms                        # expect: 5 ours_* arms listed
```

`pitlane arms` must show exactly `ours_full ours_no_seeds
ours_predictor_only ours_no_prefetch ours_none`. If any are missing,
`pitlane/ablation.toml` did not load — stop and report.

Preflight the stack:

```bash
pitlane preflight --arm ours
```

---

## 1. Record a one-question trace

The modes replay a pinned trace, and the question set is drawn with
`random.Random(62).sample(examples, N)` — the sample for N=1 is **not** a
subset of the sample for N=50, so a 50-question trace cannot be replayed at
N=1. Record its own:

```bash
RECORD=1 ARMS="" ./launch-batch.sh 1
ls -la "$TRACE_DIR/odr_n1.jsonl"
```

If that file already exists from a previous smoke test, skip this step and
reuse it. Recording twice gives two different worlds and the modes stop being
comparable.

---

## 2. Run the modes

Every mode, one question, one run id. Roughly eight minutes of boot plus one
question each — budget two hours for all five.

```bash
export RUN_ID="ablation_smoke_$(date +%Y%m%d_%H%M%S)"
ARMS="ours_full ours_no_seeds ours_predictor_only ours_no_prefetch ours_none" \
  RUN_ID="$RUN_ID" ./launch-batch.sh 1
```

The five are a ladder — each drops exactly one switch from the one before it —
so all five are needed to tell which switch a difference came from. Run them
all.

Each arm that exits non-zero is reported at the end and does not stop the
others. Note which failed; a failed arm's checks below will be missing rather
than wrong.

---

## 3. Check each mode

```bash
CELL_ROOT="$BENCH_ROOT/$RUN_ID"
```

Every arm's cell is `$CELL_ROOT/<arm>/batch1/rep1`. Run this once — it answers
most of the table in one pass:

```bash
for arm in ours_full ours_no_seeds ours_predictor_only ours_no_prefetch \
           ours_none; do
  c="$CELL_ROOT/$arm/batch1/rep1"
  [ -d "$c" ] || { printf '%-22s MISSING CELL\n' "$arm"; continue; }
  evict=$(grep -c "Node-aware KV eviction enabled" "$c"/*.log 2>/dev/null | paste -sd+ | bc)
  lg=$(grep -c '"issuer": *"langgraph"' "$c/agent_prefetch.jsonl" 2>/dev/null || echo 0)
  pop=$(grep -c '"issuer": *"population"' "$c/agent_prefetch.jsonl" 2>/dev/null || echo 0)
  seeds=$(wc -l < "$c/prompt_seeds.jsonl" 2>/dev/null || echo 0)
  blocked=$(grep -c '"blocked_on_segment_ordinal": *[0-9]' "$c/prediction.prefix.jsonl" 2>/dev/null || echo 0)
  miss=$(wc -l < "$c/divergence.jsonl" 2>/dev/null || echo 0)
  printf '%-22s evict_line=%s langgraph=%s population=%s seeds=%s blocked_prefixes=%s divergence=%s\n' \
    "$arm" "${evict:-0}" "$lg" "$pop" "$seeds" "$blocked" "$miss"
done
```

Expected:

| arm | evict_line | langgraph | population | seeds | blocked_prefixes |
|---|---|---|---|---|---|
| `ours_full` | ≥1 | >0 | >0 | >0 | 0 or few |
| `ours_no_seeds` | ≥1 | >0 | >0 | **0** | 0 or few |
| `ours_predictor_only` | ≥1 | >0 | >0 | **0** | **many** |
| `ours_no_prefetch` | ≥1 | **0** | **0** | **0** | 0 |
| `ours_none` | **0** | **0** | **0** | **0** | 0 |

`ours_no_seeds` and `ours_predictor_only` differ in one column only —
`blocked_prefixes`. Identical numbers across that pair mean the pseudo-dynamic
switch never reached langgraph, which is the failure that row exists to catch.

`ours_none` is the only row where `evict_line` is 0, and it is the only row
allowed to be: it is the floor, below the ladder rather than on it.

`evict_line` is `≥1` in every row but the last: eviction is on throughout the
ladder and off only in the floor. A **0** anywhere above `ours_none` is the
eviction switch failing to reach the server, in an arm that never asked for it
to be off — and a **non**-zero in `ours_none` is the same failure the other way
round.

Read the columns as:

- **`evict_line`** — the server logs `Node-aware KV eviction enabled` at boot.
  Every mode but `ours_none` runs the policy, so absent anywhere else is the
  switch not reaching the server (check the tmux pane's `export` lines).
- **`langgraph`** — prefetches issued by the live predictor. Zero with
  `prefetch=on` means the predictor is disabled, most likely a stale langgraph
  fork or an empty registry.
- **`population`** — the system prompt population phase. **Must be >0 in every
  row where prefetch is on, and 0 in `ours_no_prefetch`.** With prefetch on it
  is not an ablation switch: the registry is written only by `/v1/agent_chat`
  and `/v1/agents/*`, so skipping it leaves every lookup empty and every
  prefetch a silent no-op — a zero there invalidates that whole run. With
  prefetch off there is no lookup to leave empty, and the phase would only
  prefill every node's system prompt into HBM, which is warming ahead of a
  request and the one thing those rungs are defined by not doing. So
  `ours_no_prefetch` and `ours_none` pass `--skip-system-prompt-population`,
  and a **non**-zero in either is the failure.
- **`seeds`** — rows in `prompt_seeds.jsonl`. `>0` needs both `prefetch` and
  `prompt_seeds` on. Note a one-question run may produce only a
  `final_report_seed` row: `compress_research_seed` fires only on a researcher
  turn that hits the iteration cap, which one short question may never reach.
  **One row is a pass; zero where the table says `>0` is a failure.**
- **`blocked_prefixes`** — prefixes that stopped at a manufactured segment.
  This is the pseudo-dynamic switch's signature: `pseudo_dynamic=false` blocks
  at the first one, so the count should be high and the built prefixes short.
- **`divergence`** — trace misses. **Should be 0 everywhere.** Non-zero means
  the replay left the recording; see §5.

---

## 4. Two checks the loop cannot make

**`ours_full` must reproduce `ours`.** If it does not, `arms.toml` and
`ablation.toml` disagree and every other row is suspect. Run `ours` as a
sixth arm into the same run id and compare token counts, which are
deterministic under pinned replay:

```bash
ARMS="ours" RUN_ID="$RUN_ID" ./launch-batch.sh 1
python3 - <<'EOF'
import csv, os
rows = {r["arm"]: r for r in csv.DictReader(
    open(os.path.join(os.environ["BENCH_ROOT"], os.environ["RUN_ID"], "results.csv")))}
a, b = rows.get("ours"), rows.get("ours_full")
if not a or not b:
    print("MISSING an arm"); raise SystemExit
for field in ("query_tokens", "token_hits", "requests"):
    print(f"{field:15} ours={a[field]:>10} ours_full={b[field]:>10}",
          "OK" if a[field] == b[field] else "MISMATCH")
EOF
```

**The switches must be visible in the mode's own environment**, not merely
intended. For any mode, the workflow log records what the harness resolved:

```bash
grep -iE "prefetch|pseudo|seed" "$CELL_ROOT/ours_predictor_only/batch1/rep1/workflow.log" | head
```

The harness prints its real state at startup — trust that line over any mode
label.

---

## 5. If something fails

| symptom | most likely cause |
|---|---|
| a mode's cell is missing | that arm exited non-zero; read `$CELL_ROOT/<arm>/batch1/rep1/workflow.log` |
| `population=0` in an arm that prefetches | the seeding phase was skipped; that run is invalid, not an ablation |
| `population>0` in `ours_no_prefetch` | the skip flag did not reach the script; that arm warmed its system prompts and is not the floor it claims to be |
| `langgraph=0` with prefetch on | stale langgraph fork in `$WORKFLOW_VENV`, or the registry was never seeded |
| `seeds=0` with seeds on | stale `open_deep_research` (no `prompt_seeds.py`), or `ODR_PROMPT_SEEDS_LOG_PATH` not reaching the process |
| `evict_line=0` with eviction on | `VLLM_NODE_EVICTION_POLICY` not reaching the server; check the tmux pane's `export` lines: `tmux capture-pane -p -t vllm \| grep NODE_EVICTION` |
| `divergence>0` | a summarization exceeded `SUMMARIZATION_TIMEOUT_S` and fell back to raw page content, which no longer matches the recording. Check the value is 600, not the 240 default |
| two modes give identical numbers | the switch did not reach anything — the failure this test exists to find. Report which pair, with both `command.txt` files |

Report results as the table from §3 with the actual numbers, plus the
`ours` vs `ours_full` comparison. Do not tune any flag to make a check pass.
