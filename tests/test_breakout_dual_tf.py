import importlib
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

breakout_dual_tf = importlib.import_module("strategies.breakout_dual_tf")

from core.domain.models.Signal import Signal

BreakoutDualTFStrategy = breakout_dual_tf.BreakoutDualTFStrategy
Level = breakout_dual_tf.Level
BreakoutSignalPayload = breakout_dual_tf.BreakoutSignalPayload


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


def _make_payload(action: str = "BUY") -> BreakoutSignalPayload:
    level_type = "R" if action == "BUY" else "S"
    level = Level(price=100.0, level_type=level_type, timestamp=1, score=1.0)
    direction = "LONG" if action == "BUY" else "SHORT"
    return BreakoutSignalPayload(
        symbol="BTCUSDT",
        action=action,
        direction=direction,
        level=level,
        entry_price=101.0,
        sl=99.0,
        tp1=102.0,
        tp2=103.0,
        rr=2.0,
        atr=1.0,
        volume_rel=1.5,
        ema_fast=100.5,
        ema_slow=100.0,
        exec_tf="15m",
        candle=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        swing=99.5,
    )


def test_run_skips_when_position_is_open(monkeypatch, caplog):
    settings = DummySettings({"INTERVAL": "1h", "SYMBOL": "BTCUSDT"})

    class BrokerStub:
        def __init__(self):
            self.open_orders_calls = 0

        def get_position(self, symbol: str):
            return {"symbol": symbol, "positionAmt": 0.75}

        def open_orders(self, symbol: str):  # pragma: no cover - guarded by skip
            self.open_orders_calls += 1
            return []

    broker = BrokerStub()
    strategy = BreakoutDualTFStrategy(None, broker, settings)
    payload = _make_payload("BUY")
    strategy._last_payload = payload

    def fake_generate_signal(self, now):
        return Signal(action="BUY", price=payload.entry_price, time=now)

    strategy.generate_signal = types.MethodType(fake_generate_signal, strategy)

    def fail_compute_orders(self, payload, *, context=None):  # pragma: no cover - defensive
        raise AssertionError("compute_orders should not run when skipping")

    strategy.compute_orders = types.MethodType(fail_compute_orders, strategy)

    place_called = {"count": 0}

    def fail_place_orders(self, orders, *, exchange=None):  # pragma: no cover - defensive
        place_called["count"] += 1
        raise AssertionError("place_orders should not run when skipping")

    strategy.place_orders = types.MethodType(fail_place_orders, strategy)

    with caplog.at_level("INFO", logger="bot.strategy.breakout_dual_tf"):
        result = strategy.run()

    assert result["status"] == "skipped_existing_position"
    assert result["strategy"] == "breakout_dual_tf"
    assert result["symbol"] == "BTCUSDT"
    assert result["side"] == "BUY"
    assert any("skipped_existing_position" in record.message for record in caplog.records)
    assert place_called["count"] == 0


def test_run_skips_when_entry_order_exists(caplog):
    settings = DummySettings({"INTERVAL": "1h", "SYMBOL": "BTCUSDT"})

    class BrokerStub:
        def __init__(self):
            self.open_orders_calls = 0

        def get_position(self, symbol: str):
            return {"symbol": symbol, "positionAmt": "0"}

        def open_orders(self, symbol: str):
            self.open_orders_calls += 1
            return [
                {"status": "NEW", "side": "BUY", "reduceOnly": False},
                {"status": "NEW", "side": "SELL", "reduceOnly": False},
            ]

    broker = BrokerStub()
    strategy = BreakoutDualTFStrategy(None, broker, settings)
    payload = _make_payload("BUY")
    strategy._last_payload = payload

    def fake_generate_signal(self, now):
        return Signal(action="BUY", price=payload.entry_price, time=now)

    strategy.generate_signal = types.MethodType(fake_generate_signal, strategy)

    def fail_compute_orders(self, payload, *, context=None):  # pragma: no cover - defensive
        raise AssertionError("compute_orders should not run when skipping")

    strategy.compute_orders = types.MethodType(fail_compute_orders, strategy)

    place_called = {"count": 0}

    def fail_place_orders(self, orders, *, exchange=None):  # pragma: no cover - defensive
        place_called["count"] += 1
        raise AssertionError("place_orders should not run when skipping")

    strategy.place_orders = types.MethodType(fail_place_orders, strategy)

    with caplog.at_level("INFO", logger="bot.strategy.breakout_dual_tf"):
        result = strategy.run()

    assert broker.open_orders_calls == 1
    assert result["status"] == "skipped_existing_entry_orders"
    assert result["strategy"] == "breakout_dual_tf"
    assert result["symbol"] == "BTCUSDT"
    assert result["side"] == "BUY"
    assert any("skipped_existing_entry_orders" in record.message for record in caplog.records)
    assert place_called["count"] == 0


def test_run_allows_signal_when_no_duplicates(caplog):
    settings = DummySettings({"INTERVAL": "1h", "SYMBOL": "BTCUSDT"})

    class BrokerStub:
        def __init__(self):
            self.open_orders_calls = 0

        def get_position(self, symbol: str):
            return {"symbol": symbol, "positionAmt": 0}

        def open_orders(self, symbol: str):
            self.open_orders_calls += 1
            return [
                {"status": "CANCELED", "side": "BUY", "reduceOnly": False},
                {"status": "NEW", "side": "BUY", "reduceOnly": True},
            ]

    broker = BrokerStub()
    strategy = BreakoutDualTFStrategy(None, broker, settings)
    payload = _make_payload("BUY")
    strategy._last_payload = payload

    def fake_generate_signal(self, now):
        return Signal(action="BUY", price=payload.entry_price, time=now)

    strategy.generate_signal = types.MethodType(fake_generate_signal, strategy)

    def fake_compute_orders(self, payload, *, context=None):
        return {
            "symbol": payload.symbol,
            "side": payload.action,
            "entry": payload.entry_price,
            "stop_loss": payload.sl,
            "take_profit_1": payload.tp1,
            "take_profit_2": payload.tp2,
            "qty": 1.0,
            "qty_tp1": 0.5,
            "qty_tp2": 0.5,
        }

    strategy.compute_orders = types.MethodType(fake_compute_orders, strategy)

    place_calls: list[dict[str, Any]] = []

    def fake_place_orders(self, orders, *, exchange=None):
        place_calls.append({"orders": orders, "exchange": exchange})
        return {"status": "orders_placed"}

    strategy.place_orders = types.MethodType(fake_place_orders, strategy)

    with caplog.at_level("INFO", logger="bot.strategy.breakout_dual_tf"):
        result = strategy.run()

    assert broker.open_orders_calls == 2
    assert result["status"] == "signal"
    assert result["strategy"] == "breakout_dual_tf"
    assert result["symbol"] == "BTCUSDT"
    assert result["side"] == "BUY"
    assert result["orders"]["entry"] == payload.entry_price
    assert result["orders"]["qty_tp1"] == 0.5
    assert result["orders"]["qty_tp2"] == 0.5
    assert result["placement"] == {"status": "orders_placed"}
    assert len(place_calls) == 1
    assert place_calls[0]["exchange"] is broker
    assert place_calls[0]["orders"] == result["orders"]
    assert not any("skipped_existing_" in record.message for record in caplog.records)

