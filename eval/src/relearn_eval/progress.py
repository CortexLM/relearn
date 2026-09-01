"""What the run is doing, and how long it has been doing it.

The harvest runs the scorer under `timeout --kill-after=60 <secs>` and keeps
only the last 8 KiB of the log, so two things matter for a run that does not
finish: the image has to name the phase it died in, and it has to be the
loudest thing in that log tail.

`RELEARN_EVAL_OK` is never printed from here. This module only narrates.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

from .contract import ContractError

log = logging.getLogger(__name__)


class Terminated(ContractError):
    """The run was signalled — almost always the harvest's timeout."""


class BudgetExceeded(ContractError):
    """The run could not finish inside the operator's budget."""


@dataclass
class Phase:
    """The current phase of a run, and the clock it started on."""

    name: str = "starting"
    started: float = field(default_factory=time.monotonic)
    phase_started: float = field(default_factory=time.monotonic)

    def enter(self, name: str) -> None:
        now = time.monotonic()
        if self.name != "starting":
            log.info("%s done in %.0fs", self.name, now - self.phase_started)
        self.name = name
        self.phase_started = now
        log.info("phase: %s (%.0fs elapsed)", name, now - self.started)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


@dataclass
class Budget:
    """Optional wall-clock ceiling for one run.

    Unset by default: the harvest already imposes one, and guessing lower than
    the operator's timeout would refuse runs that would have finished. When it
    is set, the run stops between phases with a reason rather than being killed
    mid-phase.
    """

    seconds: float = 0.0
    phase: Phase = field(default_factory=Phase)

    @property
    def bounded(self) -> bool:
        return self.seconds > 0

    @property
    def remaining(self) -> float:
        return self.seconds - self.phase.elapsed

    def check(self, next_phase: str) -> None:
        """Refuse to start `next_phase` when the budget is already spent."""
        if self.bounded and self.remaining <= 0:
            raise BudgetExceeded(
                f"run budget of {self.seconds:.0f}s spent during {self.phase.name}; "
                f"raise RELEARN_RUN_BUDGET_SECS, prime the pod's weights, or "
                f"reduce RELEARN_MAX_NEW_TOKENS before {next_phase}"
            )


def budget_seconds() -> float:
    """`RELEARN_RUN_BUDGET_SECS`, or 0 for unbounded."""
    raw = os.environ.get("RELEARN_RUN_BUDGET_SECS", "").strip()
    if not raw:
        return 0.0
    try:
        seconds = float(raw)
    except ValueError as exc:
        raise ContractError(f"RELEARN_RUN_BUDGET_SECS {raw!r} is not a number") from exc
    if seconds < 0:
        raise ContractError("RELEARN_RUN_BUDGET_SECS must not be negative")
    return seconds
