import importlib.util
from pathlib import Path
import sys
from typing import Any
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

# Provide lightweight stubs for config.settings to avoid optional dependencies
stub_config = types.ModuleType("config")
stub_settings = types.ModuleType("config.settings")

def _sl_stub(settings):  # pragma: no cover - trivial stub
    return None


def _tp_stub(settings):  # pragma: no cover - trivial stub
    return None


stub_settings.get_stop_loss_pct = _sl_stub
stub_settings.get_take_profit_pct = _tp_stub
stub_config.settings = stub_settings
sys.modules.setdefault("config", stub_config)
sys.modules.setdefault("config.settings", stub_settings)

MODULE_PATH = ROOT / "src" / "strategies" / "breakout_dual_tf.py"
spec = importlib.util.spec_from_file_location("breakout_dual_tf", MODULE_PATH)
assert spec and spec.loader  # defensive
breakout_dual_tf = importlib.util.module_from_spec(spec)
sys.modules["breakout_dual_tf"] = breakout_dual_tf
spec.loader.exec_module(breakout_dual_tf)

BreakoutDualTFStrategy = breakout_dual_tf.BreakoutDualTFStrategy
Level = breakout_dual_tf.Level


class DummySettings:
    def __init__(self, data: dict[str, Any] | None = None):
        self.data = data or {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __getattr__(self, item):
        return self.data.get(item)


def _make_candle(ts: int, open_px: float, high: float, low: float, close: float, vol: float):
    return [float(ts), float(open_px), float(high), float(low), float(close), float(vol)]


def _build_history(base_ts: int, count: int, start: float = 98.0, step: float = 0.1):
    candles = []
    price = start
    for idx in range(count):
        ts = base_ts + idx * 60_000
        candles.append(_make_candle(ts, price - 0.1, price + 0.5, price - 0.5, price, 100.0))
        price += step
    return candles


def test_breakout_dual_tf_rejects_low_volume(caplog):
    settings = DummySettings({"INTERVAL": "1h"})
    strategy = BreakoutDualTFStrategy(None, None, settings)
    level = Level(price=100.0, level_type="R", timestamp=1, score=1.0)

    candles = _build_history(0, 60, start=97.0, step=0.05)
    atr = 2.0
    strategy._last_level_atr = atr
    breakout_ts = candles[-1][0] + 60_000
    candles.append(_make_candle(breakout_ts, 100.0, 105.0, 99.5, 102.0, 40.0))

    context = {
        "symbol": "BTCUSDT",
        "exec_tf": "15m",
        "exec_candles": candles,
    }

    with caplog.at_level("INFO", logger="bot.strategy.breakout_dual_tf"):
        result = strategy.should_trigger_breakout(candles[-1], [level], context)

    assert result is None
    assert any("vol_rel" in record.message for record in caplog.records)


def test_breakout_dual_tf_retest_flow():
    settings = DummySettings({"INTERVAL": "1h"})
    strategy = BreakoutDualTFStrategy(
        None,
        None,
        settings,
        config={"USE_RETEST": True, "VOL_REL_MIN": 0.5, "K_ATR": 0.1, "RR_MIN": 0.5},
    )
    level = Level(price=100.0, level_type="R", timestamp=1, score=1.0)

    candles = _build_history(0, 55, start=97.0, step=0.2)
    strategy._last_level_atr = 1.5

    # Breakout candle (should set pending retest)
    breakout_ts = candles[-1][0] + 60_000
    breakout = _make_candle(breakout_ts, 100.0, 104.0, 99.8, 102.5, 250.0)
    candles.append(breakout)
    context = {"symbol": "BTCUSDT", "exec_tf": "15m", "exec_candles": candles}
    assert strategy.should_trigger_breakout(breakout, [level], context) is None
    assert strategy._pending_breakouts  # type: ignore[attr-defined]

    # Retest candle (touches level)
    retest_ts = breakout_ts + 60_000
    retest = _make_candle(retest_ts, 101.0, 102.2, 100.1, 101.8, 220.0)
    candles.append(retest)
    context = {"symbol": "BTCUSDT", "exec_tf": "15m", "exec_candles": candles}
    assert strategy.should_trigger_breakout(retest, [level], context) is None

    # Confirmation candle (new high, should trigger)
    confirm_ts = retest_ts + 60_000
    confirm = _make_candle(confirm_ts, 102.0, 105.0, 101.5, 103.8, 260.0)
    candles.append(confirm)
    context = {"symbol": "BTCUSDT", "exec_tf": "15m", "exec_candles": candles}
    signal_payload = strategy.should_trigger_breakout(confirm, [level], context)
    assert signal_payload is not None
    assert signal_payload.action == "BUY"

    # Without retest requirement we should trigger immediately
    strategy_no_retest = BreakoutDualTFStrategy(
        None,
        None,
        settings,
        config={"USE_RETEST": False, "VOL_REL_MIN": 0.5, "K_ATR": 0.1, "RR_MIN": 0.5},
    )
    strategy_no_retest._last_level_atr = 1.5
    candles_short = candles[:-2]  # remove retest/confirm
    candles_short.append(breakout)
    context = {"symbol": "BTCUSDT", "exec_tf": "15m", "exec_candles": candles_short}
    signal_payload_direct = strategy_no_retest.should_trigger_breakout(breakout, [level], context)
    assert signal_payload_direct is not None
    assert signal_payload_direct.action == "BUY"


def test_breakout_dual_tf_cooldown_blocks_reentry():
    settings = DummySettings({"INTERVAL": "1h"})
    strategy = BreakoutDualTFStrategy(
        None,
        None,
        settings,
        config={
            "USE_RETEST": False,
            "MAX_RETRIES": 2,
            "COOLDOWN_BARS": 3,
            "VOL_REL_MIN": 0.5,
            "K_ATR": 0.1,
            "RR_MIN": 0.5,
        },
    )
    level = Level(price=100.0, level_type="R", timestamp=1, score=1.0)

    candles = _build_history(0, 55, start=97.0, step=0.2)
    strategy._last_level_atr = 1.5
    breakout_ts = candles[-1][0] + 60_000
    breakout = _make_candle(breakout_ts, 100.0, 104.0, 99.5, 103.0, 220.0)
    candles.append(breakout)
    context = {"symbol": "BTCUSDT", "exec_tf": "15m", "exec_candles": candles}
    payload = strategy.should_trigger_breakout(breakout, [level], context)
    assert payload is not None

    strategy.register_trade_exit(level_price=level.price, direction="LONG", time_ms=int(breakout_ts), reason="sl")

    # Next candle inside cooldown -> should reject
    next_ts = breakout_ts + 60_000
    next_candle = _make_candle(next_ts, 100.2, 100.8, 99.6, 100.5, 180.0)
    candles.append(next_candle)
    context = {"symbol": "BTCUSDT", "exec_tf": "15m", "exec_candles": candles}
    assert strategy.should_trigger_breakout(next_candle, [level], context) is None

    # Advance beyond cooldown bars -> signal allowed again
    far_ts = breakout_ts + 3 * 60_000
    far_candle = _make_candle(far_ts, 102.0, 106.0, 101.5, 104.0, 250.0)
    candles.append(far_candle)
    context = {"symbol": "BTCUSDT", "exec_tf": "15m", "exec_candles": candles}
    result = strategy.should_trigger_breakout(far_candle, [level], context)
    assert result is not None

