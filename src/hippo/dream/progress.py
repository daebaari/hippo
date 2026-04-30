"""Throttled progress reporter for dream-heavy phases.

Writes phase progress to the dream_runs row and to stderr at most once per
PROGRESS_THROTTLE_SECONDS or every PROGRESS_THROTTLE_TICKS callback invocations,
whichever comes first.
"""
from __future__ import annotations

PROGRESS_THROTTLE_SECONDS = 5.0
PROGRESS_THROTTLE_TICKS = 100


def rolling_rate(
    *, now_done: int, then_done: int, now_time: float, then_time: float
) -> float:
    """Items per second between two snapshots. Returns 0.0 on degenerate input."""
    elapsed = now_time - then_time
    delta = now_done - then_done
    if elapsed <= 0 or delta <= 0:
        return 0.0
    return delta / elapsed


def format_eta(*, remaining: int, rate: float) -> str:
    """Render remaining/rate as `<1m`, `Nm`, `>99m`, or `?` (rate==0)."""
    if rate <= 0:
        return "?"
    seconds = remaining / rate
    minutes = round(seconds / 60)
    if seconds < 60:
        return "<1m"
    if minutes > 99:
        return ">99m"
    return f"{minutes}m"


from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ProgressReporter:
    """Throttled progress callback.

    Calls `emit(done, total)` at most once per PROGRESS_THROTTLE_SECONDS or every
    PROGRESS_THROTTLE_TICKS calls to tick(). Always emits exactly once on finish().
    """

    emit: Callable[[int, int], None]
    clock: Callable[[], float]
    total: int
    _ticks_since_emit: int = 0
    _last_emit_time: float = 0.0
    _last_done: int = 0
    _started: bool = False

    def tick(self, done: int) -> None:
        if not self._started:
            self._started = True
        self._last_done = done
        self._ticks_since_emit += 1
        now = self.clock()
        time_due = now - self._last_emit_time >= PROGRESS_THROTTLE_SECONDS
        ticks_due = self._ticks_since_emit >= PROGRESS_THROTTLE_TICKS
        if time_due or ticks_due:
            self.emit(done, self.total)
            self._last_emit_time = now
            self._ticks_since_emit = 0

    def finish(self) -> None:
        self.emit(self._last_done, self.total)


PHASE_COL_WIDTH = 16


def _phase_col(phase: str) -> str:
    return f"phase={phase:<{PHASE_COL_WIDTH}}"


def format_phase_start_line(*, phase: str, total: int) -> str:
    return f"{_phase_col(phase)} total={total}"


def format_phase_complete_line(*, phase: str, total: int, elapsed_s: int) -> str:
    return f"{_phase_col(phase)} done={total}/{total} (100%) elapsed={elapsed_s}s"


def format_progress_line(
    *,
    phase: str,
    done: int,
    total: int,
    elapsed_s: int,
    rate: float,
    eta: str,
) -> str:
    pct = (100 * done / total) if total > 0 else 0.0
    return (
        f"{_phase_col(phase)} done={done}/{total} ({pct:.1f}%) "
        f"rate={rate:.1f}/s eta={eta}"
    )
