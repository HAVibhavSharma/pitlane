"""Per-question Gantt timelines: one Mermaid file and one table per cell.

What a cell's numbers cannot show is *shape* -- whether a warm left before the
call it serves, whether three researcher tool calls really ran in parallel,
whether a 19 ms prefetch landed inside a 200 ms gap or on top of the prefill it
was meant to precede. A row of aggregates hides all of it; a timeline does not.

Model calls come from the request rows, never from the server log.
`agent_prefetch_start` / `agent_prefetch_end` take their instants from the
engine, and the engine already writes them as `arrival_ts` / `finish_ts` on the
phantom's own `prefetch_only=true` row -- so parsing the log back would be a
lossier route to the same two numbers, and would need the log to have been kept
at the right level.

Leaf tool spans are the exception, and have to be: a tool call is not a request,
so no row records it. Those come from the `/v1/echo` markers, which is what puts
them on the same clock as everything else here rather than on the workflow
process's own.

Two consequences of the request-row choice, both deliberate:

* The population phase never appears. Its phantoms are `langgraph:*:**:...` and
  they run before `t0`, so the question window has already excluded them; the
  agent-id filter below is belt and braces for a run whose window is loose.
* Parallel calls stay separate, because each is its own row. Nothing here
  merges by agent id -- a node that takes three concurrent turns gets three
  numbered tasks, which is the whole point of looking.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from pitlane.metrics import Metrics

# Mermaid's own colours, set once at the top of every file so a diagram pasted
# anywhere reads the same: orange for prefetch (`crit`), blue for chat
# (`active`), green for a leaf tool (`done`).
#
# Three is the limit. Mermaid has exactly four task styles -- `crit`, `active`,
# `done`, `milestone` -- and `milestone` renders as a point rather than a bar,
# so a fourth event kind would need its own diagram rather than another colour.
_INIT = """%%{init: {
  "theme": "base",
  "themeVariables": {
    "activeTaskBkgColor": "#4C78A8",
    "activeTaskBorderColor": "#4C78A8",
    "critBkgColor": "#F28E2B",
    "critBorderColor": "#F28E2B",
    "doneTaskBkgColor": "#59A14F",
    "doneTaskBorderColor": "#59A14F",
    "gridColor": "#cccccc"
  }
}}%%"""

_STYLE = {"prefetch": ("crit", "pf"), "chat": ("active", "c"), "tool": ("done", "t")}

# Concrete workflow agents are `langgraph:<unit>:...`; the population phase uses
# `langgraph:*:**:...`. Anything with a `*` in it is warmup.
_WARMUP_MARK = "*"


@dataclass(frozen=True)
class Event:
    """One bar. `kind` is "prefetch", "chat" or "tool"."""

    agent_id: str      # the lane: the graph node, shared by both arms
    kind: str
    index: int          # 1-based, chronological within (arm, agent_id, kind)
    start: float
    end: float
    name: str = ""      # tool name, for a tool span
    arm: str = ""
    job_id: str = ""
    question_id: str = ""
    # Phase split, chat completions only. A bar says how long a call took; this
    # says where the time went, which is the difference between "ours is slower
    # here" and "ours waited longer to start here".
    queued_s: float | None = None
    prefill_s: float | None = None
    decode_s: float | None = None

    @property
    def duration_s(self) -> float:
        return max(self.end - self.start, 0.0)

    @property
    def label(self) -> str:
        if self.kind == "prefetch":
            return f"Prefetch #{self.index} ({self.duration_s * 1000:.1f} ms)"
        if self.kind == "tool":
            return (
                f"Tool #{self.index} {self.name} ({self.duration_s * 1000:.1f} ms)"
            )
        return f"Chat #{self.index}{self.phases}"

    @property
    def phases(self) -> str:
        """` (q 0.10s · p 0.30s · d 7.50s)`, or empty when nothing is known.

        Abbreviated because it rides in a Gantt task name, where the bar is
        already carrying the total. Any field that is missing is omitted rather
        than shown as zero -- a phantom has no decode phase at all, and writing
        `d 0.00s` would make that look like a measurement.
        """
        parts = [
            (label, value) for label, value in
            (("q", self.queued_s), ("p", self.prefill_s), ("d", self.decode_s))
            if value is not None
        ]
        if not parts:
            return ""
        return " (" + " · ".join(f"{k} {v:.2f}s" for k, v in parts) + ")"


def _concrete(agent_id: str | None, node: str | None = None) -> bool:
    """Whether this span belongs on a question's chart.

    Not "has an agent id". Only `/v1/agents/*` stamps one, so every `baseline`
    row has an empty one -- and a filter that required it dropped the entire
    arm the chart exists to compare against, leaving a comparison with one side
    in it and nothing to say so. A row with a node name is a real call whatever
    endpoint served it; what is excluded is the population phase, whose agent
    ids carry a `*`.
    """
    if agent_id and _WARMUP_MARK in agent_id:
        return False
    return bool(agent_id or node)


def _lane(agent_id: str | None, node: str | None) -> str:
    """The row a span is drawn on: the graph node, for both arms alike.

    `langgraph_node` rather than `agent_id`, and not only because baseline has
    no agent id. Within one question's chart the id's `langgraph:<job>:` prefix
    is redundant -- the file is that job -- and using the node name puts the two
    arms on identically named lanes, which is what makes a section-by-section
    read possible at all. Parallel calls on one node stay distinct as numbered
    tasks, exactly as three concurrent `researcher_tools` turns already did.
    """
    if node:
        return node
    return (agent_id or "").split(":")[-1] or "(unknown)"


def _short(agent_id: str) -> str:
    """Drop the `langgraph:` prefix every label would otherwise carry."""
    return agent_id[len("langgraph:"):] if agent_id.startswith("langgraph:") else agent_id


def _stamp(ts: float) -> str:
    """`YYYY-MM-DD HH:mm:ss.SSS`, truncated rather than rounded.

    Rounding a millisecond here would let a task's rendered start drift past an
    end it really preceded, which is exactly the ordering these diagrams exist
    to show.
    """
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _number(raw: list[Event]) -> list[Event]:
    """Sort by time and number within `(arm, agent_id, kind)`.

    Per arm, so a combined chart numbers `baseline` and `ours` independently --
    `Chat #2` has to mean the same call in both or the comparison is unreadable.
    """
    raw.sort(key=lambda e: (e.start, e.end))
    seen: dict[tuple[str, str, str], int] = {}
    out: list[Event] = []
    for event in raw:
        key = (event.arm, event.agent_id, event.kind)
        seen[key] = seen.get(key, 0) + 1
        out.append(replace(event, index=seen[key]))
    return out


def events(metrics: Metrics) -> list[Event]:
    """Every concrete-agent bar for one cell, in time order."""
    raw: list[Event] = []
    for entry in metrics.per_prefetch:
        if (_concrete(entry.get("agent_id"), entry.get("langgraph_node"))
                and entry.get("arrival_ts") is not None):
            raw.append(Event(
                _lane(entry.get("agent_id"), entry.get("langgraph_node")),
                "prefetch", 0, entry["arrival_ts"],
                entry.get("finish_ts") or entry["arrival_ts"], "",
                metrics.arm, str(entry.get("job_id") or ""), metrics.question_id,
                entry.get("queued_s"), entry.get("prefill_s"), entry.get("decode_s"),
            ))
    for entry in metrics.per_request:
        if (_concrete(entry.get("agent_id"), entry.get("langgraph_node"))
                and entry.get("arrival_ts") is not None):
            raw.append(Event(
                _lane(entry.get("agent_id"), entry.get("langgraph_node")),
                "chat", 0, entry["arrival_ts"],
                entry.get("finish_ts") or entry["arrival_ts"], "",
                metrics.arm, str(entry.get("job_id") or ""), metrics.question_id,
                entry.get("queued_s"), entry.get("prefill_s"), entry.get("decode_s"),
            ))
    for entry in metrics.per_tool:
        if (_concrete(entry.get("agent_id"), entry.get("langgraph_node"))
                and entry.get("start_ts") is not None):
            raw.append(Event(
                _lane(entry.get("agent_id"), entry.get("langgraph_node")),
                "tool", 0, entry["start_ts"],
                entry.get("end_ts") or entry["start_ts"], entry.get("tool") or "tool",
                metrics.arm, str(entry.get("job_id") or ""), metrics.question_id,
            ))
    return _number(raw)


def mermaid(metrics: Metrics, *, title: str | None = None) -> str:
    """A Gantt chart of the cell, one section per agent id.

    Short bars are left at their true length. A 19 ms prefetch inside a 200 s
    workflow renders as a sliver Mermaid may not show at all, so the duration
    goes in the task name instead -- the information survives even when the
    geometry does not. Stretching the bar would make the picture legible and
    the data wrong.
    """
    heading = title or f"{metrics.arm} — {metrics.question_id} (rep {metrics.rep})"
    lines = [
        _INIT,
        "gantt",
        f"    title Agent Timeline: {heading}",
        "    dateFormat YYYY-MM-DD HH:mm:ss.SSS",
        "    axisFormat %H:%M:%S",
        "    todayMarker off",
    ]
    ordered = events(metrics)
    if not ordered:
        lines.append("    section (no events in window)")
        return "\n".join(lines) + "\n"

    # Sections in first-appearance order, so the chart reads top-to-bottom the
    # way the workflow ran.
    agents: list[str] = []
    for event in ordered:
        if event.agent_id not in agents:
            agents.append(event.agent_id)

    task_id = 0
    for agent_id in agents:
        lines += ["", f"    section {_short(agent_id)}"]
        for event in (e for e in ordered if e.agent_id == agent_id):
            task_id += 1
            tag, prefix = _STYLE[event.kind]
            lines.append(
                f"    {event.label} :{tag}, {prefix}{task_id}, "
                f"{_stamp(event.start)}, {_stamp(event.end)}"
            )
    return "\n".join(lines) + "\n"


def table(metrics: Metrics) -> str:
    """Per-call timings, with the counts the diagram has to agree with.

    Offsets are relative to the first concrete event, which is what makes two
    arms comparable; the absolute stamps live in the Mermaid file, and both are
    rendered from the same `events()` list so they cannot disagree.
    """
    ordered = events(metrics)
    chats = sum(1 for e in ordered if e.kind == "chat")
    prefetches = sum(1 for e in ordered if e.kind == "prefetch")
    tools = sum(1 for e in ordered if e.kind == "tool")
    origin = ordered[0].start if ordered else 0.0

    lines = [
        f"# Timeline — {metrics.arm} / {metrics.question_id} / rep {metrics.rep}",
        "",
        f"- chat-completion calls: **{chats}**",
        f"- concrete-agent prefetches: **{prefetches}**",
        f"- leaf tool calls: **{tools}**",
        f"- t=0 is {_stamp(origin)}" if ordered else "- no events in window",
        "",
        "| Agent ID | Type | Call # | Start (+s) | End (+s) | Duration |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for event in ordered:
        duration = (
            f"{event.duration_s:.3f} s" if event.kind == "chat"
            else f"{event.duration_s * 1000:.1f} ms"
        )
        kind = f"{event.kind} {event.name}".strip()
        lines.append(
            f"| {_short(event.agent_id)} | {kind} | {event.index} | "
            f"{event.start - origin:.3f} | {event.end - origin:.3f} | {duration} |"
        )
    return "\n".join(lines) + "\n"


def write(timelines_dir: Path, metrics: Metrics) -> list[Path]:
    """Write `<arm>__<question>__rep<k>.{mmd,md}` and return what was written."""
    ordered = events(metrics)
    if not ordered:
        return []
    timelines_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{metrics.arm}__{metrics.question_id}__rep{metrics.rep}"
    written = []
    for suffix, body in ((".mmd", mermaid(metrics)), (".md", table(metrics))):
        path = timelines_dir / f"{stem}{suffix}"
        path.write_text(body)
        written.append(path)
    return written


# -- run level --------------------------------------------------------------
#
# One cell's chart answers "what did this arm do". The question the study asks
# is "what did the arms do differently", and that needs them on one axis --
# which per-cell files cannot give, because each is written before the next arm
# has run.
#
# The run-root CSVs already carry every span with its `arm`, `question_id` and
# `job_id`, so the combined charts are a second read of those rather than a
# second bookkeeping path. Regenerated after every cell, so a matrix that is
# still running, or one that failed partway, still has whatever it produced.

_CSV_KINDS = (
    ("prefetches.csv", "prefetch", "arrival_ts", "finish_ts", None),
    ("requests.csv", "chat", "arrival_ts", "finish_ts", None),
    ("tools.csv", "tool", "start_ts", "end_ts", "tool"),
)


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def run_events(run_dir: Path) -> list[Event]:
    """Every span the run has produced so far, across arms, from the CSVs."""
    raw: list[Event] = []
    for name, kind, start_key, end_key, name_key in _CSV_KINDS:
        for row in _rows(run_dir / name):
            agent_id = row.get("agent_id") or ""
            node = row.get("langgraph_node") or ""
            start = _float(row.get(start_key))
            if not _concrete(agent_id, node) or start is None:
                continue
            raw.append(Event(
                _lane(agent_id, node), kind, 0, start,
                _float(row.get(end_key)) or start,
                (row.get(name_key) or "tool") if name_key else "",
                row.get("arm") or "", str(row.get("job_id") or ""),
                row.get("question_id") or "",
                _float(row.get("queued_s")), _float(row.get("prefill_s")),
                _float(row.get("decode_s")),
            ))
    return _number(raw)


def _rebased(events_: list[Event]) -> tuple[list[Event], dict[str, float]]:
    """Shift every arm to a common origin, and say by how much.

    Arms run one after another, minutes or hours apart, so plotting them on
    real clock time puts them side by side as distant blocks and compares
    nothing. Each arm is rebased to its own first event, which is the only way
    "the warm fires 88 ms before the call in one arm and not at all in the
    other" is a thing you can see.

    The shift is returned so the table can state it: these are not wall-clock
    times any more, and a chart that silently pretends otherwise is worse than
    one that does not exist.
    """
    origins: dict[str, float] = {}
    for event in events_:
        if event.arm not in origins or event.start < origins[event.arm]:
            origins[event.arm] = event.start
    base = min(origins.values()) if origins else 0.0
    shifted = [
        replace(event, start=event.start - origins[event.arm] + base,
                end=event.end - origins[event.arm] + base)
        for event in events_
    ]
    shifted.sort(key=lambda e: (e.arm, e.start, e.end))
    return shifted, origins


def combined_mermaid(events_: list[Event], *, title: str) -> str:
    """One Gantt, sectioned `<arm> · <node>`.

    Used for both the per-question chart and the per-arm one. Rebasing is a
    no-op on a single arm -- its origin is the base -- so a per-arm file keeps
    true wall-clock times without needing a second code path, and the title only
    claims a rebase when there was one to do.
    """
    shifted, origins = _rebased(events_)
    note = " (each arm rebased to its own start)" if len(origins) > 1 else ""
    lines = [
        _INIT,
        "gantt",
        f"    title {title}{note}",
        "    dateFormat YYYY-MM-DD HH:mm:ss.SSS",
        "    axisFormat %H:%M:%S",
        "    todayMarker off",
    ]
    sections: list[tuple[str, str]] = []
    for event in shifted:
        key = (event.arm, event.agent_id)
        if key not in sections:
            sections.append(key)

    task_id = 0
    for arm, agent_id in sections:
        lines += ["", f"    section {arm} · {_short(agent_id)}"]
        for event in (e for e in shifted if e.arm == arm and e.agent_id == agent_id):
            task_id += 1
            tag, prefix = _STYLE[event.kind]
            lines.append(
                f"    {event.label} :{tag}, {prefix}{task_id}, "
                f"{_stamp(event.start)}, {_stamp(event.end)}"
            )
    return "\n".join(lines) + "\n"


def combined_table(events_: list[Event], *, title: str) -> str:
    shifted, origins = _rebased(events_)
    base = min(origins.values()) if origins else 0.0
    lines = [f"# {title}", ""]
    for arm in sorted({e.arm for e in shifted}):
        chats = sum(1 for e in shifted if e.arm == arm and e.kind == "chat")
        pf = sum(1 for e in shifted if e.arm == arm and e.kind == "prefetch")
        tools = sum(1 for e in shifted if e.arm == arm and e.kind == "tool")
        span = max((e.end for e in shifted if e.arm == arm), default=base) - base
        lines.append(
            f"- **{arm}**: {chats} chat, {pf} prefetch, {tools} tool "
            f"— {span:.1f}s wall (t=0 is this arm's first event, "
            f"{_stamp(origins[arm])})"
        )
    lines += [
        "",
        "| Arm | Agent ID | Type | Call # | Start (+s) | End (+s) | Duration | "
        "Queued (s) | Prefill (s) | Decode (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for event in shifted:
        duration = (
            f"{event.duration_s:.3f} s" if event.kind == "chat"
            else f"{event.duration_s * 1000:.1f} ms"
        )
        kind = f"{event.kind} {event.name}".strip()
        phase = [
            "—" if value is None else f"{value:.3f}"
            for value in (event.queued_s, event.prefill_s, event.decode_s)
        ]
        lines.append(
            f"| {event.arm} | {_short(event.agent_id)} | {kind} | {event.index} | "
            f"{event.start - base:.3f} | {event.end - base:.3f} | {duration} | "
            + " | ".join(phase) + " |"
        )
    return "\n".join(lines) + "\n"


def write_run(run_dir: Path) -> list[Path]:
    """One `.mmd` + `.md` per question, every arm together.

    Split by `(question_id, job_id)` rather than by cell. A batch cell is N
    questions in one process, so a per-cell file would lump them into a single
    unreadable chart -- and the question, not the cell, is the unit anything is
    compared at.
    """
    all_events = run_events(run_dir)
    if not all_events:
        return []
    out = run_dir / "timelines"
    out.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str], list[Event]] = {}
    for event in all_events:
        groups.setdefault((event.question_id, event.job_id), []).append(event)

    written: list[Path] = []
    for (question, job), group in sorted(groups.items()):
        stem = f"{question}__job{job}" if job else str(question)
        title = f"{question} / job {job}" if job else str(question)

        # The comparison chart, and then one per arm. Two questions, two
        # answers: "what differs" needs the arms on one axis, "what did this
        # arm do" needs the other arm out of the way and the real clock back --
        # a per-arm file is not rebased, since there is nothing to rebase
        # against.
        variants: list[tuple[str, str, list[Event]]] = [(stem, title, group)]
        for arm in sorted({e.arm for e in group if e.arm}):
            variants.append((
                f"{stem}__{arm}", f"{title} — {arm}",
                [e for e in group if e.arm == arm],
            ))

        for variant_stem, variant_title, events_ in variants:
            for suffix, body in (
                (".mmd", combined_mermaid(events_, title=variant_title)),
                (".md", combined_table(events_, title=variant_title)),
            ):
                path = out / f"{variant_stem}{suffix}"
                path.write_text(body)
                written.append(path)
    return written
