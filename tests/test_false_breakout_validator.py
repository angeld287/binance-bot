from __future__ import annotations

import importlib.util
import os
import sys
from datetime import UTC, datetime

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
MODULE_PATH = os.path.join(
    SRC_DIR, "strategies", "breakout", "validators", "false_breakout.py"
)

spec = importlib.util.spec_from_file_location("false_breakout", MODULE_PATH)
assert spec and spec.loader
false_breakout = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = false_breakout
spec.loader.exec_module(false_breakout)

FalseBreakoutSettings = false_breakout.FalseBreakoutSettings
validate_false_breakout = false_breakout.validate_false_breakout


INTERVAL_MS = 60 * 60 * 1000


def make_candles(count: int, base_price: float = 109.0) -> list[list[float]]:
    candles: list[list[float]] = []
    for i in range(count):
        candles.append(
            [
                float(i * INTERVAL_MS),
                base_price,
                base_price + 0.5,
                base_price - 0.5,
                base_price,
                1000.0,
            ]
        )
    return candles


def update_candle(
    candles: list[list[float]],
    idx: int,
    *,
    open_price: float | None = None,
    high: float | None = None,
    low: float | None = None,
    close: float | None = None,
    volume: float | None = None,
) -> None:
    if open_price is not None:
        candles[idx][1] = open_price
    if high is not None:
        candles[idx][2] = high
    if low is not None:
        candles[idx][3] = low
    if close is not None:
        candles[idx][4] = close
    if volume is not None:
        candles[idx][5] = volume


def test_false_breakout_blocks_on_wick_and_low_volume() -> None:
    candles = make_candles(70)
    level = 110.0

    update_candle(candles, 63, open_price=109.2, high=level, low=108.7, close=109.2, volume=1050)
    update_candle(candles, 64, open_price=109.1, high=level, low=108.8, close=109.4, volume=950)
    update_candle(candles, 66, open_price=109.0, high=level, low=108.9, close=109.3, volume=970)
    update_candle(candles, 68, open_price=109.2, high=level, low=108.9, close=109.6, volume=980)
    update_candle(
        candles,
        69,
        open_price=110.0,
        high=111.0,
        low=109.5,
        close=110.25,
        volume=900.0,
    )

    allowed, reason, _ = validate_false_breakout(
        ctx={},
        side="BUY",
        level=level,
        timeframe="1h",
        klines=candles,
        now=datetime.now(UTC),
        params=FalseBreakoutSettings(),
    )
    assert not allowed
    assert reason == "wick_ratio/vol_confirm"


def test_false_breakout_allows_confirmed_breakout_with_volume() -> None:
    candles = make_candles(70)
    level = 110.0

    update_candle(candles, 63, open_price=109.2, high=level, low=108.7, close=109.2, volume=1050)
    update_candle(candles, 64, open_price=109.1, high=level, low=108.8, close=109.4, volume=950)
    update_candle(candles, 66, open_price=109.0, high=level, low=108.9, close=109.3, volume=970)
    update_candle(candles, 68, open_price=109.2, high=level, low=108.9, close=109.6, volume=980)
    update_candle(
        candles,
        69,
        open_price=110.2,
        high=110.7,
        low=109.9,
        close=110.6,
        volume=1500.0,
    )

    allowed, reason, _ = validate_false_breakout(
        ctx={},
        side="BUY",
        level=level,
        timeframe="1h",
        klines=candles,
        now=datetime.now(UTC),
        params=FalseBreakoutSettings(),
    )
    assert allowed
    assert reason == "ok"


def test_false_breakout_blocks_when_time_window_exceeded() -> None:
    candles = make_candles(80)
    level = 110.0

    update_candle(candles, 30, open_price=109.2, high=level, low=108.8, close=109.3, volume=1020)
    update_candle(candles, 31, open_price=109.1, high=level, low=108.9, close=109.4, volume=990)
    update_candle(candles, 50, open_price=109.0, high=level, low=108.8, close=109.2, volume=980)
    update_candle(candles, 78, open_price=109.3, high=level, low=108.9, close=109.6, volume=980)
    update_candle(
        candles,
        79,
        open_price=110.3,
        high=110.8,
        low=109.9,
        close=110.6,
        volume=1500.0,
    )

    allowed, reason, _ = validate_false_breakout(
        ctx={},
        side="BUY",
        level=level,
        timeframe="1h",
        klines=candles,
        now=datetime.now(UTC),
        params=FalseBreakoutSettings(),
    )
    assert not allowed
    assert reason == "time_window"


def test_false_breakout_allows_valid_retest() -> None:
    candles = make_candles(70)
    level = 110.0

    update_candle(candles, 65, open_price=109.3, high=level, low=108.9, close=109.5, volume=1020)
    update_candle(candles, 66, open_price=109.4, high=level, low=109.0, close=109.8, volume=990)
    update_candle(
        candles,
        67,
        open_price=109.8,
        high=110.6,
        low=109.4,
        close=110.5,
        volume=1500.0,
    )
    update_candle(candles, 68, open_price=109.6, high=level, low=109.2, close=109.9, volume=1100)
    update_candle(
        candles,
        69,
        open_price=109.95,
        high=110.16,
        low=109.4,
        close=110.05,
        volume=1400.0,
    )

    allowed, reason, details = validate_false_breakout(
        ctx={},
        side="BUY",
        level=level,
        timeframe="1h",
        klines=candles,
        now=datetime.now(UTC),
        params=FalseBreakoutSettings(),
    )
    assert allowed
    assert reason == "ok"
    assert details["metrics"]["retest_ok"] is True
