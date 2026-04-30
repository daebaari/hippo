"""Tests for dream progress reporting (ETA math + throttled reporter)."""
from __future__ import annotations

import pytest

from hippo.dream.progress import format_eta, rolling_rate


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
