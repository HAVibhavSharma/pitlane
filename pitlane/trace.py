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
from datetime import datetime
from pathlib import Path

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


def recorded_date(path: Path) -> str | None:
    """`YYYY-MM-DD` of the recording, from its first request's wall stamp.

    Several ODR prompts interpolate `get_today_str()`, so an unpinned clock
    changes every prompt prefix from one day to the next and the replay's
    request hashes stop matching. `ODR_FROZEN_DATE` exists to pin it, but
    setting it by hand means keeping a date in an env file in step with a file
    on disk -- and the failure when they drift is a trace miss on the very
    first call, which reads like a broken trace rather than a stale variable.

    The trace already knows. `record_started_s` is the wall clock the first
    recorded request went out on, in the same local timezone `datetime.now()`
    reads, so the date derived here is the date the recording's prompts carry.
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
