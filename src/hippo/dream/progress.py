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
