"""Tests for dream progress reporting (ETA math + throttled reporter)."""
from __future__ import annotations

import pytest

from hippo.dream.progress import (
    ProgressReporter,
    format_eta,
    format_phase_complete_line,
    format_phase_start_line,
    format_progress_line,
    rolling_rate,
)


def test_rolling_rate_simple_case():
    rate = rolling_rate(now_done=200, then_done=100, now_time=10.0, then_time=0.0)
    assert rate == pytest.approx(10.0)


def test_rolling_rate_zero_elapsed_returns_zero():
    rate = rolling_rate(now_done=200, then_done=100, now_time=5.0, then_time=5.0)
    assert rate == 0.0


def test_rolling_rate_zero_progress_returns_zero():
    rate = rolling_rate(now_done=100, then_done=100, now_time=10.0, then_time=0.0)
    assert rate == 0.0


def test_format_eta_minutes():
    # 100 remaining at 1.0/s = 100s → "2m"
    assert format_eta(remaining=100, rate=1.0) == "2m"


def test_format_eta_under_one_minute():
    # 30 remaining at 1.0/s = 30s
    assert format_eta(remaining=30, rate=1.0) == "<1m"


def test_format_eta_zero_rate():
    assert format_eta(remaining=100, rate=0.0) == "?"


def test_format_eta_capped_at_99m():
    assert format_eta(remaining=10000, rate=1.0) == ">99m"


def test_progress_reporter_throttles_by_ticks():
    emits: list[tuple[int, int]] = []

    def emit(done: int, total: int) -> None:
        emits.append((done, total))

    clock = [0.0]
    reporter = ProgressReporter(
        emit=emit,
        clock=lambda: clock[0],
        total=500,
    )
    for i in range(1, 501):
        reporter.tick(i)
    reporter.finish()

    # 500 ticks, throttle every 100, so 5 emits during ticks + 1 finish
    assert len(emits) == 6
    assert emits[-1] == (500, 500)


def test_progress_reporter_throttles_by_seconds():
    emits: list[tuple[int, int]] = []

    def emit(done: int, total: int) -> None:
        emits.append((done, total))

    clock = [0.0]
    reporter = ProgressReporter(
        emit=emit,
        clock=lambda: clock[0],
        total=10,
    )
    # 10 ticks below tick-threshold; advance clock past PROGRESS_THROTTLE_SECONDS
    # between each tick to force time-based emits.
    for i in range(1, 11):
        clock[0] += 5.5
        reporter.tick(i)
    reporter.finish()

    # 10 time-based emits + 1 finish (finish always emits final state)
    assert len(emits) == 11


def test_progress_reporter_finish_always_emits_final_state():
    emits: list[tuple[int, int]] = []

    def emit(done: int, total: int) -> None:
        emits.append((done, total))

    reporter = ProgressReporter(
        emit=emit, clock=lambda: 0.0, total=3,
    )
    reporter.tick(1)
    reporter.tick(2)
    reporter.tick(3)
    reporter.finish()

    assert emits[-1] == (3, 3)


def test_format_progress_line_running_phase():
    line = format_progress_line(
        phase="edge_proposal",
        done=87,
        total=5765,
        elapsed_s=5,
        rate=9.0,
        eta="10m",
    )
    assert "phase=edge_proposal" in line
    assert "done=87/5765" in line
    assert "(1.5%)" in line
    assert "rate=9.0/s" in line
    assert "eta=10m" in line


def test_format_phase_start_line():
    line = format_phase_start_line(phase="atomize", total=12)
    assert "phase=atomize" in line
    assert "total=12" in line


def test_format_phase_complete_line():
    line = format_phase_complete_line(phase="review", total=16, elapsed_s=27)
    assert "phase=review" in line
    assert "done=16/16" in line
    assert "(100%)" in line
    assert "elapsed=27s" in line
