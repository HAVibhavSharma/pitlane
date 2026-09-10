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

from dataclasses import dataclass
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

    agent_id: str
    kind: str
    index: int          # 1-based, chronological within (agent_id, kind)
    start: float
    end: float
    name: str = ""      # tool name, for a tool span

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
        return f"Chat #{self.index}"


def _concrete(agent_id: str | None) -> bool:
    return bool(agent_id) and _WARMUP_MARK not in agent_id


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


def events(metrics: Metrics) -> list[Event]:
    """Every concrete-agent bar, in time order, numbered per agent and kind."""
    rows: list[tuple[float, float, str, str, str]] = []
    for entry in metrics.per_prefetch:
        if _concrete(entry.get("agent_id")) and entry.get("arrival_ts") is not None:
            rows.append((entry["arrival_ts"], entry.get("finish_ts") or entry["arrival_ts"],
                         entry["agent_id"], "prefetch", ""))
    for entry in metrics.per_request:
        if _concrete(entry.get("agent_id")) and entry.get("arrival_ts") is not None:
            rows.append((entry["arrival_ts"], entry.get("finish_ts") or entry["arrival_ts"],
                         entry["agent_id"], "chat", ""))
    for entry in metrics.per_tool:
        if _concrete(entry.get("agent_id")) and entry.get("start_ts") is not None:
            rows.append((entry["start_ts"], entry.get("end_ts") or entry["start_ts"],
                         entry["agent_id"], "tool", entry.get("tool") or "tool"))
    rows.sort(key=lambda r: (r[0], r[1]))

    seen: dict[tuple[str, str], int] = {}
    out: list[Event] = []
    for start, end, agent_id, kind, name in rows:
        key = (agent_id, kind)
        seen[key] = seen.get(key, 0) + 1
        out.append(Event(agent_id, kind, seen[key], start, end, name))
    return out


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
