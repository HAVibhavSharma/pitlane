"""What a recorded trace says about itself.

A replay has to reproduce the recording's prompts byte for byte, and two of the
inputs to those prompts are not in the trace file's keys: the date the prompts
interpolate, and which questions were asked. Both are recoverable from the
recording, and recovering them beats asking the operator to keep a second copy
of the answer in an env file -- a trace that has been moved, renamed or
re-recorded then carries its own truth.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

# What `get_today_str()` renders into every prompt that interpolates a date:
# `f"{now:%a} {now:%b} {now.day}, {now:%Y}"`, e.g. `Tue Aug 4, 2026`. Note the
# un-padded day, which is why this is matched rather than parsed with a fixed
# width.
_PROMPT_DATE = re.compile(
    r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) "
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{4}\b"
)

logger = logging.getLogger("pitlane.trace")


def _first_entry(path: Path) -> dict | None:
    try:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    return json.loads(line)
    except (OSError, json.JSONDecodeError):
        return None
    return None


def prompt_date(path: Path) -> str | None:
    """`YYYY-MM-DD` as it appears *inside* the recording's prompts.

    The authoritative answer, and not the same as when the recording ran. A
    recording made under its own `ODR_FROZEN_DATE` carries that date in its
    prompts while its wall stamps say something else entirely -- measured on a
    real trace: recorded Sep 8, prompts reading `Tue Aug 4, 2026`. Replaying
    against the wall stamp would then miss every request, so the prompt text is
    what has to be matched.
    """
    entry = _first_entry(path)
    if entry is None:
        return None
    try:
        body = json.dumps(entry.get("request_body") or "")
    except (TypeError, ValueError):
        return None
    found = _PROMPT_DATE.search(body)
    if not found:
        return None
    try:
        return datetime.strptime(found.group(0), "%a %b %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def started_date(path: Path) -> str | None:
    """`YYYY-MM-DD` the recording *ran* on, from its first wall stamp.

    A fallback, and only correct for a recording that was not itself frozen --
    see `prompt_date`, which is what the prompts actually say.
    """
    entry = _first_entry(path)
    if entry is None:
        return None
    started = entry.get("record_started_s")
    if not started:
        return None
    try:
        return datetime.fromtimestamp(float(started)).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return None


def recorded_date(path: Path) -> tuple[str | None, str]:
    """`(date, source)` for the replay to pin to.

    Several ODR prompts interpolate `get_today_str()`, so replaying on a
    different date changes every prompt prefix and the first request misses.
    The prompts themselves are the truth; the wall stamp is a guess that is
    right only when the recording ran unfrozen.
    """
    date = prompt_date(path)
    if date:
        return date, "prompt"
    date = started_date(path)
    if date:
        return date, "record_started_s"
    return None, "none"


def recorded_jobs(path: Path) -> list[str]:
    """The distinct `job_id`s in the trace -- one per question recorded.

    The count is the N the trace was recorded at, and the workflow has to be
    asked for the same N: ODR selects with `random.Random(0).sample(examples,
    N)`, so a different N is a different set of questions, not a subset.
    """
    jobs: list[str] = []
    try:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    job = json.loads(line).get("job_id")
                except json.JSONDecodeError:
                    continue
                if job is not None and str(job) not in jobs:
                    jobs.append(str(job))
    except OSError:
        return []
    return jobs
