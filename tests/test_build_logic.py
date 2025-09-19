import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from strategies.liquidity_sweep.strategy import build_entry_prices, build_bracket
import pytest


def test_build_entry_prices_rounds_to_tick():
    levels = {"S": 100.0, "R": 110.0, "microbuffer": 0.27}
    settings = SimpleNamespace(TICK_SIZE=0.1)
    res = build_entry_prices(levels, settings=settings)
    assert res["buy_px"] == pytest.approx(100.3)
    assert res["sell_px"] == pytest.approx(109.7)


def test_build_bracket_long_structural():
    settings = SimpleNamespace(TICK_SIZE=0.1)
    side = "BUY"
    entry = 100.3
    S, R = 100.0, 102.0
    micro = 0.27
    buffer_sl = 0.2
    res = build_bracket(side, entry, S, R, micro, buffer_sl, atr1m=0.0, tp_policy=None, settings=settings)
    assert res["sl"] == pytest.approx(99.8)
    assert res["tp"] == pytest.approx(101.7)


def test_build_bracket_long_uses_multiple_when_rr_low():
    settings = SimpleNamespace(TICK_SIZE=0.1)
    side = "BUY"
    entry = 100.3
    S, R = 100.0, 100.8
    micro = 0.27
    buffer_sl = 0.3
    res = build_bracket(side, entry, S, R, micro, buffer_sl, atr1m=0.0, tp_policy=None, settings=settings)
    assert res["sl"] == pytest.approx(99.7)
    assert res["tp"] == pytest.approx(101.4)


def test_build_bracket_short_structural():
    settings = SimpleNamespace(TICK_SIZE=0.1)
    side = "SELL"
    entry = 101.7
    S, R = 100.0, 102.0
    micro = 0.27
    buffer_sl = 0.2
    res = build_bracket(side, entry, S, R, micro, buffer_sl, atr1m=0.0, tp_policy=None, settings=settings)
    assert res["sl"] == pytest.approx(102.2)
    assert res["tp"] == pytest.approx(100.3)


def test_build_bracket_short_uses_multiple_when_rr_low():
    settings = SimpleNamespace(TICK_SIZE=0.1)
    side = "SELL"
    entry = 100.5
    S, R = 100.0, 100.8
    micro = 0.27
    buffer_sl = 0.3
    res = build_bracket(side, entry, S, R, micro, buffer_sl, atr1m=0.0, tp_policy=None, settings=settings)
    assert res["sl"] == pytest.approx(101.1)
    assert res["tp"] == pytest.approx(99.4)
