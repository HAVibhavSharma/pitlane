"""pitlane -- stop the car, swap the engine, run a timed lap on a fixed circuit.

Brings up a serving stack, waits for it, runs one LangGraph workflow question
against it under a pinned trace, tears it down, and collects the metrics. The
workload is identical across stacks, so what differs in the numbers is the
serving layer.
"""

__version__ = "0.1.0"
