from __future__ import annotations

import json
import math
import pytest
from dataclasses import dataclass, replace
from decimal import Decimal
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_SRC = Path(__file__).resolve().parents[3]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

if "boto3" not in sys.modules:
    boto3_stub = types.ModuleType("boto3")

    def _stub_client(_name: str):  # pragma: no cover - safety net
        raise RuntimeError("boto3 client should be mocked in tests")

    boto3_stub.client = _stub_client
    sys.modules["boto3"] = boto3_stub

if "botocore" not in sys.modules:
    botocore_pkg = types.ModuleType("botocore")
    exceptions_pkg = types.ModuleType("botocore.exceptions")

    class ClientError(Exception):
        def __init__(self, error_response=None, operation_name=""):
            super().__init__("client error")
            self.response = error_response or {}
            self.operation_name = operation_name

    class BotoCoreError(Exception):
        pass

    exceptions_pkg.ClientError = ClientError
    exceptions_pkg.BotoCoreError = BotoCoreError
    botocore_pkg.exceptions = exceptions_pkg
    sys.modules["botocore"] = botocore_pkg
    sys.modules["botocore.exceptions"] = exceptions_pkg

strategies_pkg = types.ModuleType("strategies")
strategies_pkg.__path__ = [str(PROJECT_SRC / "strategies")]
sys.modules.setdefault("strategies", strategies_pkg)

if "strategies.wedge_formation" not in sys.modules:
    wedge_pkg = types.ModuleType("strategies.wedge_formation")
    wedge_pkg.__path__ = []
    sys.modules["strategies.wedge_formation"] = wedge_pkg

if "strategies.wedge_formation.strategy" not in sys.modules:
    wedge_strategy = types.ModuleType("strategies.wedge_formation.strategy")

    @dataclass
    class SymbolFilters:
        tick_size: Decimal
        step_size: Decimal
        min_notional: Decimal
        min_qty: Decimal

    @dataclass
    class PrecisionResult:
        price_requested: Decimal | None
        qty_requested: Decimal | None
        stop_requested: Decimal | None
        side: str
        order_type: str
        filters: SymbolFilters
        exchange: object
        symbol: str
        price_adjusted: Decimal | None
        qty_adjusted: Decimal | None
        stop_adjusted: Decimal | None = None
        price_payload: str | None = None
        stop_payload: str | None = None
        price_decimals: int | None = None
        stop_decimals: int | None = None

    @dataclass
    class QtyGuardResult:
        success: bool
        qty: Decimal | None
        adjusted: bool
        reason: str | None

    class OrderPrecisionError(Exception):
        def __init__(self, tag: str, reason: str) -> None:
            super().__init__(reason)
            self.tag = tag
            self.reason = reason

    def compute_order_precision(**kwargs):
        price = kwargs.get("price_requested")
        qty = kwargs.get("qty_requested")
        stop = kwargs.get("stop_requested")
        return PrecisionResult(
            price_requested=price,
            qty_requested=qty,
            stop_requested=stop,
            side=kwargs.get("side", ""),
            order_type=kwargs.get("order_type", ""),
            filters=kwargs.get("filters"),
            exchange=kwargs.get("exchange"),
            symbol=kwargs.get("symbol", ""),
            price_adjusted=price,
            qty_adjusted=qty,
            stop_adjusted=stop,
        )

    def apply_qty_guards(**kwargs):
        return QtyGuardResult(True, kwargs.get("qty_dec"), False, None)

    def get_symbol_filters(exchange, symbol):
        raw = exchange.get_symbol_filters(symbol)
        price = Decimal(str(raw["PRICE_FILTER"]["tickSize"]))
        step = Decimal(str(raw["LOT_SIZE"]["stepSize"]))
        min_notional = Decimal(str(raw["MIN_NOTIONAL"]["minNotional"]))
        min_qty = Decimal(str(raw["LOT_SIZE"].get("minQty", "0")))
        return SymbolFilters(price, step, min_notional, min_qty)

    class WedgeFormationStrategy:
        @staticmethod
        def _compute_atr(candles, period: int = 14):
            if len(candles) < 2:
                return 0.0
            trs = []
            prev_close = float(candles[0][4])
            for candle in candles[1:]:
                high = float(candle[2])
                low = float(candle[3])
                close = float(candle[4])
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                trs.append(tr)
                prev_close = close
            if not trs:
                return 0.0
            window = trs[-period:] if len(trs) >= period else trs
            return sum(window) / len(window)

    wedge_strategy.SymbolFilters = SymbolFilters
    wedge_strategy.apply_qty_guards = apply_qty_guards
    wedge_strategy.compute_order_precision = compute_order_precision
    wedge_strategy.get_symbol_filters = get_symbol_filters
    wedge_strategy.OrderPrecisionError = OrderPrecisionError
    wedge_strategy.WedgeFormationStrategy = WedgeFormationStrategy
    sys.modules["strategies.wedge_formation.strategy"] = wedge_strategy

if "strategies.breakout_dual_tf" not in sys.modules:
    dual_pkg = types.ModuleType("strategies.breakout_dual_tf")
    dual_pkg.__path__ = []
    sys.modules["strategies.breakout_dual_tf"] = dual_pkg

if "strategies.breakout_dual_tf.filters" not in sys.modules:
    filters_pkg = types.ModuleType("strategies.breakout_dual_tf.filters")
    filters_pkg.__path__ = []
    sys.modules["strategies.breakout_dual_tf.filters"] = filters_pkg

if "strategies.breakout_dual_tf.filters.ema_distance" not in sys.modules:
    ema_module = types.ModuleType("strategies.breakout_dual_tf.filters.ema_distance")

    class DummyResult:
        def __init__(self, ema_fast, ema_slow):
            self.ok = True
            self.reason = ""
            self.ema7 = float(ema_fast or 0.0)
            self.ema25 = float(ema_slow or 0.0)
            self.nearest_ema_label = "ema7"
            self.dist_to_ema7_pct = 0.0
            self.dist_to_ema25_pct = 0.0

    def compute_ema_distance(ohlc, ema_fast, ema_slow, side="", **_kwargs):
        return DummyResult(ema_fast, ema_slow)

    ema_module.compute_ema_distance = compute_ema_distance
    sys.modules["strategies.breakout_dual_tf.filters.ema_distance"] = ema_module

from strategies.ParallelChannelFormation import channel_detector, stale_pending_orders
from strategies.ParallelChannelFormation.channel_detector import ChannelEnv, MarketSnapshot, STRATEGY_NAME, run
from strategies.wedge_formation.strategy import OrderPrecisionError


def test_select_candles_for_ema_disabled(monkeypatch):
    monkeypatch.setenv("EMA_HIGHER_TF_ENABLED", "0")

    class DummyMarketData:
        def fetch_ohlcv(self, *args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("fetch_ohlcv should not be called when disabled")

    base_candles = [[0, 0, 0, 0, 1.0, 1.0]]
    selected, tf_used = channel_detector._select_candles_for_ema(
        DummyMarketData(),
        symbol="BTCUSDT",
        base_timeframe="1m",
        limit=200,
        base_candles=base_candles,
    )

    assert selected == base_candles
    assert tf_used == "1m"


def test_select_candles_for_ema_enabled(monkeypatch):
    monkeypatch.setenv("EMA_HIGHER_TF_ENABLED", "1")

    class DummyMarketData:
        def __init__(self) -> None:
            self.requested: list[str] = []

        def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int):
            self.requested.append(timeframe)
            if timeframe == "15m":
                return [[0, 0, 0, 0, 2.0, 1.0]]
            raise AssertionError(f"unexpected timeframe requested: {timeframe}")

    base_candles = [[0, 0, 0, 0, 1.0, 1.0]]
    dummy_md = DummyMarketData()
    selected, tf_used = channel_detector._select_candles_for_ema(
        dummy_md,
        symbol="BTCUSDT",
        base_timeframe="1m",
        limit=200,
        base_candles=base_candles,
    )

    assert dummy_md.requested == ["15m"]
    assert tf_used == "15m"
    assert selected != base_candles
    assert selected[0][4] == 2.0


def test_select_candles_for_ema_fallback_when_empty(monkeypatch):
    monkeypatch.setenv("EMA_HIGHER_TF_ENABLED", "1")

    class DummyMarketData:
        def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int):
            return []

    base_candles = [[0, 0, 0, 0, 1.0, 1.0]]
    selected, tf_used = channel_detector._select_candles_for_ema(
        DummyMarketData(),
        symbol="BTCUSDT",
        base_timeframe="1m",
        limit=200,
        base_candles=base_candles,
    )

    assert selected == base_candles
    assert tf_used == "1m"


class FakeExchange:
    def __init__(self) -> None:
        self._open_orders: list[dict] = []
        self.position: dict | None = None
        self.tp_orders: list[dict] = []
        self.entry_orders: list[dict] = []
        self.stop_orders: list[dict] = []
        self.market_orders: list[dict] = []
        self.cancelled_orders: list[dict] = []
        self._order_seq = 1

    def open_orders(self, symbol: str):
        return list(self._open_orders)

    def get_position(self, symbol: str):
        return self.position

    def place_tp_reduce_only(self, *, symbol: str, side: str, tpPrice: float, qty: float, clientOrderId: str):
        self.tp_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "tpPrice": tpPrice,
                "qty": qty,
                "clientOrderId": clientOrderId,
            }
        )
        return {"status": "NEW"}

    def place_entry_limit(self, *, symbol: str, side: str, price: float, qty: float, clientOrderId: str, timeInForce: str = "GTC"):
        order_id = self._order_seq
        self._order_seq += 1
        self.entry_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "price": price,
                "qty": qty,
                "clientOrderId": clientOrderId,
                "orderId": order_id,
            }
        )
        self._open_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "price": price,
                "origQty": qty,
                "clientOrderId": clientOrderId,
                "orderId": order_id,
                "status": "NEW",
            }
        )
        return {"status": "NEW"}

    def place_stop_reduce_only(self, *, symbol: str, side: str, stopPrice: float, qty: float, clientOrderId: str):
        self.stop_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "stopPrice": stopPrice,
                "qty": qty,
                "clientOrderId": clientOrderId,
            }
        )
        return {"status": "NEW"}

    def place_entry_market(self, *, symbol: str, side: str, qty: float, clientOrderId: str, reduceOnly: bool = False):
        self.market_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "clientOrderId": clientOrderId,
                "reduceOnly": reduceOnly,
            }
        )
        return {"status": "FILLED"}

    def cancel_order(self, *, symbol: str, orderId: int | None = None, clientOrderId: str | None = None):
        self.cancelled_orders.append(
            {
                "symbol": symbol,
                "orderId": orderId,
                "clientOrderId": clientOrderId,
            }
        )
        remaining: list[dict] = []
        for order in self._open_orders:
            matches_id = orderId is not None and order.get("orderId") == orderId
            matches_client = clientOrderId and order.get("clientOrderId") == clientOrderId
            if matches_id or matches_client:
                continue
            remaining.append(order)
        self._open_orders = remaining

    def round_price_to_tick(self, symbol: str, price: float) -> float:
        return price

    def round_qty_to_step(self, symbol: str, qty: float) -> float:
        return qty

    def get_symbol_filters(self, symbol: str):
        return {
            "PRICE_FILTER": {"tickSize": "0.1"},
            "LOT_SIZE": {"stepSize": "0.01", "minQty": "0.01"},
            "MIN_NOTIONAL": {"minNotional": "5"},
        }


def _candles() -> list[list[float]]:
    candles: list[list[float]] = []
    for idx in range(60):
        trend = 100 + idx * 0.2
        oscillation = math.sin(idx / 3.0) * 1.5
        close = trend + oscillation
        high = close + 2.0
        low = close - 2.0
        candles.append([idx, trend, high, low, close, 10.0])
    return candles


def _mock_tp_store(monkeypatch):
    storage: dict[str, dict[str, Any]] = {}
    channel_storage: dict[str, dict[str, Any]] = {}

    def fake_load(symbol: str):
        entry = storage.get(symbol)
        if not entry:
            return None
        return entry.get("tp_value")

    def fake_load_entry(symbol: str):
        entry = storage.get(symbol)
        if not entry:
            return None
        return dict(entry)

    def fake_persist(symbol: str, tp_value: float, timestamp: float, extra: dict | None = None):
        payload: dict[str, Any] = {
            "symbol": symbol,
            "tp_value": float(tp_value),
            "timestamp": float(timestamp),
        }
        if extra:
            payload.update(extra)
        storage[symbol] = payload
        return True

    def fake_load_symbol_channel(symbol: str):
        record = channel_storage.get(symbol)
        if not record:
            return None
        return dict(record)

    def fake_persist_symbol_channel(symbol: str, payload: Mapping[str, Any]) -> None:
        channel_storage[str(symbol)] = dict(payload)

    monkeypatch.setattr(channel_detector, "load_tp_value", fake_load, raising=False)
    monkeypatch.setattr(channel_detector, "load_tp_entry", fake_load_entry)
    monkeypatch.setattr(channel_detector, "persist_tp_value", fake_persist)
    monkeypatch.setattr(channel_detector, "load_symbol_channel", fake_load_symbol_channel)
    monkeypatch.setattr(channel_detector, "persist_symbol_channel", fake_persist_symbol_channel)
    monkeypatch.setattr(stale_pending_orders, "load_tp_entry", fake_load_entry)
    monkeypatch.setattr(stale_pending_orders, "persist_tp_value", fake_persist)

    storage["__channel_records__"] = channel_storage
    return storage


def _env() -> ChannelEnv:
    return ChannelEnv(
        tolerance_slope=0.2,
        min_touches=1,
        min_vertical_gap_pct=0.0,
        min_duration_bars=0,
        confidence_threshold=0.0,
        tp_mode="opuesto_inmediato",
        sl_enabled=True,
        fixed_sl_pct=1.0,
        price_tick_override=None,
        qty_step_override=None,
        min_notional_buffer_pct=0.0,
        max_trades_per_channel=1,
    )


def test_open_orders_skip(monkeypatch):
    _mock_tp_store(monkeypatch)
    exchange = FakeExchange()
    exchange._open_orders = [{"status": "NEW", "clientOrderId": "OTHER"}]
    snapshot = MarketSnapshot(
        candles=_candles(),
        timeframe="15m",
        atr=1.0,
        ema_fast=100.0,
        ema_slow=100.0,
        volume_avg=10.0,
    )
    env = _env()
    result = run(
        "BTCUSDT",
        snapshot,
        {"qty": 1.0},
        env,
        exchange=exchange,
    )
    assert result["action"] == "reject"
    assert result["reason"] == "open_order_exists"


def test_place_tp_for_existing_position(monkeypatch):
    store = _mock_tp_store(monkeypatch)
    now_ts = datetime.utcnow().timestamp()
    store["BTCUSDT"] = {"tp_value": 110.0, "timestamp": now_ts}
    exchange = FakeExchange()
    exchange.position = {"positionAmt": "1", "entryPrice": "100"}
    exchange._open_orders = [{"status": "NEW", "clientOrderId": "PCF_TP_EXISTING"}]
    env = _env()
    snapshot = MarketSnapshot(
        candles=_candles(),
        timeframe="15m",
        atr=1.0,
        ema_fast=100.0,
        ema_slow=100.0,
        volume_avg=10.0,
    )
    result = run(
        "BTCUSDT",
        snapshot,
        {"qty": 1.0},
        env,
        exchange=exchange,
    )
    assert result["action"] == "monitor_tp"
    assert len(exchange.tp_orders) == 1


def test_detection_places_entry_and_persists_tp(monkeypatch):
    store = _mock_tp_store(monkeypatch)
    store["ADAUSDT"] = {"tp_value": 1.0, "timestamp": datetime.utcnow().timestamp()}
    exchange = FakeExchange()
    env = _env()
    snapshot = MarketSnapshot(
        candles=_candles(),
        timeframe="15m",
        atr=1.0,
        ema_fast=100.0,
        ema_slow=100.0,
        volume_avg=10.0,
    )
    result = run(
        "BTCUSDT",
        snapshot,
        {"qty": 1.0},
        env,
        exchange=exchange,
    )
    assert result["action"] == "place_order"
    assert exchange.entry_orders
    assert exchange.stop_orders
    stored = store.get("BTCUSDT")
    assert stored is not None
    assert math.isclose(stored["tp_value"], result["tp1"], rel_tol=1e-9)
    assert stored.get("status") == "OPEN"
    assert stored.get("opened_at")
    channel_records = store["__channel_records__"]
    state = channel_records.get("BTCUSDT")
    assert state is not None
    assert state.get("symbol") == "BTCUSDT"
    assert state.get("lifetime_trades_opened") == 1
    assert "channel_id" not in state


def test_channel_meta_persisted(monkeypatch):
    store = _mock_tp_store(monkeypatch)
    channel_records = store["__channel_records__"]
    exchange = FakeExchange()
    env = _env()
    snapshot = MarketSnapshot(
        candles=_candles(),
        timeframe="15m",
        atr=1.0,
        ema_fast=100.0,
        ema_slow=100.0,
        volume_avg=10.0,
    )

    result = run(
        "BTCUSDT",
        snapshot,
        {"qty": 1.0},
        env,
        exchange=exchange,
    )
    assert result["action"] == "place_order"
    stored = store.get("BTCUSDT")
    assert stored is not None
    meta = stored.get("channel_meta")
    assert meta is not None
    assert meta.get("break_logged") is False
    assert meta.get("lower_at_entry") < meta.get("upper_at_entry")
    assert meta.get("anchor_start_hm")
    assert meta.get("anchor_end_hm")
    fixed_sl_meta = meta.get("fixed_sl")
    assert isinstance(fixed_sl_meta, dict)
    assert fixed_sl_meta.get("price")
    assert fixed_sl_meta.get("pct") == pytest.approx(1.0)

    utc_minus_four = timezone(timedelta(hours=-4))
    start_seconds = float(meta.get("anchor_start_ts", 0))
    end_seconds = float(meta.get("anchor_end_ts", 0))
    if start_seconds > 10**10:
        start_seconds /= 1000.0
    if end_seconds > 10**10:
        end_seconds /= 1000.0
    expected_start = datetime.fromtimestamp(start_seconds, tz=utc_minus_four).strftime("%H:%M")
    expected_end = datetime.fromtimestamp(end_seconds, tz=utc_minus_four).strftime("%H:%M")
    assert meta.get("anchor_start_hm") == expected_start
    assert meta.get("anchor_end_hm") == expected_end

    assert "channel_id" not in stored
    assert "channel_id" not in meta
    state = channel_records.get("BTCUSDT")
    assert state is not None
    assert state.get("lifetime_trades_opened") == 1

    pending_order = stored.get("pending_order")
    assert isinstance(pending_order, dict)
    assert pending_order.get("client_order_id")
    assert pending_order.get("timeframe") == "15m"
    assert pending_order.get("side") in {"LONG", "SHORT"}
    assert pending_order.get("candle_index_created") == len(_candles()) - 1
    assert stored.get("timeframe") == "15m"
    assert stored.get("entry_price")


def test_sweep_stale_pending_order_cancels(monkeypatch):
    store = _mock_tp_store(monkeypatch)
    exchange = FakeExchange()
    env = _env()
    snapshot = MarketSnapshot(
        candles=_candles(),
        timeframe="15m",
        atr=1.0,
        ema_fast=100.0,
        ema_slow=100.0,
        volume_avg=10.0,
    )

    result = run(
        "BTCUSDT",
        snapshot,
        {"qty": 1.0},
        env,
        exchange=exchange,
    )
    assert result["action"] == "place_order"

    stored = dict(store.get("BTCUSDT") or {})
    pending = dict(stored.get("pending_order") or {})
    assert pending

    pending["candle_index_created"] = int(pending.get("candle_index_created", 0)) - 5
    if pending["candle_index_created"] < 0:
        pending["candle_index_created"] = 0
    limit_price = float(pending.get("limit_price", 0.0)) or 100.0
    pending["limit_price"] = limit_price
    stored["pending_order"] = pending
    store["BTCUSDT"] = stored

    monkeypatch.setenv("MAX_WAIT_CANDLES", "3")
    monkeypatch.setenv("MAX_DRIFT_PCT", "0.001")

    current_index = pending["candle_index_created"] + 10
    current_price = limit_price * 1.01

    stale_pending_orders.sweep_stale_pending_orders(
        exchange=exchange,
        symbol="BTCUSDT",
        current_price=current_price,
        current_candle_index=current_index,
        timeframe="15m",
    )

    assert exchange.cancelled_orders
    updated = store.get("BTCUSDT")
    assert updated is not None
    assert updated.get("status") == "CANCELLED"
    assert updated.get("cancel_reason") == "expired"
    assert "pending_order" not in updated


def test_channel_trade_record_logged(monkeypatch):
    store = _mock_tp_store(monkeypatch)
    channel_records = store["__channel_records__"]
    exchange = FakeExchange()
    env = _env()
    snapshot = MarketSnapshot(
        candles=_candles(),
        timeframe="15m",
        atr=1.0,
        ema_fast=100.0,
        ema_slow=100.0,
        volume_avg=10.0,
    )

    logged: list[dict[str, Any]] = []

    def fake_log(payload: Mapping[str, Any]) -> None:
        logged.append(dict(payload))

    monkeypatch.setattr(channel_detector, "_log", fake_log)

    result = run(
        "BTCUSDT",
        snapshot,
        {"qty": 1.0},
        env,
        exchange=exchange,
    )

    assert result["action"] == "place_order"
    state = channel_records.get("BTCUSDT")
    assert state is not None
    assert state.get("lifetime_trades_opened") == 1

    trade_logs = [
        log
        for log in logged
        if log.get("action") == "open" and log.get("reason") == "channel_entry"
    ]
    assert trade_logs
    trade_log = trade_logs[0]
    assert trade_log.get("symbol") == "BTCUSDT"
    assert trade_log.get("side") == state.get("side")
    assert trade_log.get("lifetime_after") == 1
    assert trade_log.get("max") == env.max_trades_per_channel
    assert "channel_id" not in trade_log


def test_rejects_when_channel_trade_limit_reached(monkeypatch):
    store = _mock_tp_store(monkeypatch)
    channel_records = store["__channel_records__"]
    exchange = FakeExchange()
    env = replace(_env(), max_trades_per_channel=1)
    snapshot = MarketSnapshot(
        candles=_candles(),
        timeframe="15m",
        atr=1.0,
        ema_fast=100.0,
        ema_slow=100.0,
        volume_avg=10.0,
    )

    first_result = run(
        "BTCUSDT",
        snapshot,
        {"qty": 1.0},
        env,
        exchange=exchange,
    )

    assert first_result["action"] == "place_order"
    stored_entry = store.get("BTCUSDT")
    assert stored_entry is not None
    state = channel_records.get("BTCUSDT")
    assert state is not None
    assert state.get("lifetime_trades_opened") == 1

    exchange.entry_orders.clear()
    exchange.stop_orders.clear()
    exchange.tp_orders.clear()
    exchange._open_orders.clear()

    logged: list[dict[str, Any]] = []

    def fake_log(payload: Mapping[str, Any]) -> None:
        logged.append(dict(payload))

    monkeypatch.setattr(channel_detector, "_log", fake_log)

    second_result = run(
        "BTCUSDT",
        snapshot,
        {"qty": 1.0},
        env,
        exchange=exchange,
    )

    assert second_result["action"] == "reject"
    assert second_result["reason"] == "channel_trade_limit"
    assert channel_records["BTCUSDT"]["lifetime_trades_opened"] == 1
    assert any(log.get("reason") == "channel_trade_limit" for log in logged)


def test_channel_break_logged_once(monkeypatch, caplog):
    store = _mock_tp_store(monkeypatch)
    exchange = FakeExchange()
    env = _env()
    base_snapshot = MarketSnapshot(
        candles=_candles(),
        timeframe="15m",
        atr=1.0,
        ema_fast=100.0,
        ema_slow=100.0,
        volume_avg=10.0,
    )

    run(
        "BTCUSDT",
        base_snapshot,
        {"qty": 1.0},
        env,
        exchange=exchange,
    )

    exchange._open_orders.clear()
    stored = store.get("BTCUSDT")
    assert stored is not None
    channel_meta = stored.get("channel_meta")
    assert channel_meta is not None
    tolerance = float(channel_meta.get("break_tolerance", 0.0005))

    lower_entry = float(channel_meta["lower_at_entry"])
    upper_entry = float(channel_meta["upper_at_entry"])
    side = str(channel_meta.get("side", "LONG"))

    entry_price = float(channel_meta.get("entry_price", lower_entry))
    position_amt = "1" if side.upper() == "LONG" else "-1"
    exchange.position = {"positionAmt": position_amt, "entryPrice": str(entry_price)}

    candles_break = _candles()
    if side.upper() == "LONG":
        break_price = lower_entry * (1 - tolerance * 5)
        candles_break[-1][4] = break_price
        candles_break[-1][3] = min(candles_break[-1][3], break_price)
        candles_break[-1][2] = max(candles_break[-1][2], break_price + 1.0)
    else:
        break_price = upper_entry * (1 + tolerance * 5)
        candles_break[-1][4] = break_price
        candles_break[-1][2] = max(candles_break[-1][2], break_price)
        candles_break[-1][3] = min(candles_break[-1][3], break_price - 1.0)

    snapshot_break = MarketSnapshot(
        candles=candles_break,
        timeframe="15m",
        atr=1.0,
        ema_fast=100.0,
        ema_slow=100.0,
        volume_avg=10.0,
    )

    caplog.clear()
    caplog.set_level("INFO")
    result_monitor = run(
        "BTCUSDT",
        snapshot_break,
        {"qty": 1.0},
        env,
        exchange=exchange,
    )
    assert result_monitor["action"] in {"monitor_tp", "reject", "monitoring"}

    close_logs = []
    
    break_logs = []
    for record in caplog.records:
        try:
            payload = json.loads(record.message)
        except Exception:
            continue
        action = payload.get("action")
        if action == "channel_break":
            break_logs.append(payload)
        if action == "sl_structure_close":
            close_logs.append(payload)

    assert len(break_logs) == 1
    assert break_logs[0]["now"]["price"] == break_price
    assert store["BTCUSDT"]["channel_meta"]["break_logged"] is True
    assert close_logs
    assert close_logs[0]["status"] == "success"
    assert exchange.market_orders

    caplog.clear()
    caplog.set_level("INFO")
    run(
        "BTCUSDT",
        snapshot_break,
        {"qty": 1.0},
        env,
        exchange=exchange,
    )

    repeat_logs = []
    repeat_close = []
    for record in caplog.records:
        try:
            payload = json.loads(record.message)
        except Exception:
            continue
        action = payload.get("action")
        if action == "channel_break":
            repeat_logs.append(payload)
        if action == "sl_structure_close":
            repeat_close.append(payload)

    assert not repeat_logs
    assert not repeat_close
    assert len(exchange.market_orders) == 1


def test_channel_break_requires_close_outside(monkeypatch):
    captured_logs: list[dict[str, Any]] = []

    def _capture(payload: Mapping[str, Any]):
        captured_logs.append(dict(payload))

    monkeypatch.setattr(channel_detector, "_log", _capture)

    persisted: list[tuple[Any, ...]] = []

    def _persist(*args: Any, **kwargs: Any):
        persisted.append(args)

    monkeypatch.setattr(channel_detector, "persist_tp_value", _persist)

    timeframe_sec = 60.0
    entry_ts = datetime.utcnow().timestamp() - timeframe_sec * 2
    channel_meta = {
        "break_logged": False,
        "slope": 0.0,
        "intercept_mid": 100.0,
        "width": 5.0,
        "entry_index": 0.0,
        "entry_ts": entry_ts,
        "timeframe_sec": timeframe_sec,
        "break_tolerance": channel_detector.CHANNEL_BREAK_TOLERANCE,
        "lower_at_entry": 95.0,
        "upper_at_entry": 105.0,
        "entry_price": 100.0,
    }

    store_payload: dict[str, Any] = {"channel_meta": channel_meta}

    channel_detector._maybe_log_channel_break(
        symbol="BTCUSDT",
        side="LONG",
        current_price=94.0,
        candle_close=96.0,
        store_payload=store_payload,
        position=None,
    )

    assert not captured_logs
    assert not persisted
    assert channel_meta.get("break_logged") is False

    channel_detector._maybe_log_channel_break(
        symbol="BTCUSDT",
        side="LONG",
        current_price=94.0,
        candle_close=94.0,
        store_payload=store_payload,
        position=None,
    )

    assert captured_logs
    assert captured_logs[0]["action"] == "channel_break"
    assert captured_logs[0]["now"]["price"] == 94.0
    assert persisted


def test_precision_failure_bubbles_reason(monkeypatch):
    _mock_tp_store(monkeypatch)
    exchange = FakeExchange()
    env = _env()
    snapshot = MarketSnapshot(
        candles=_candles(),
        timeframe="15m",
        atr=1.0,
        ema_fast=100.0,
        ema_slow=100.0,
        volume_avg=10.0,
    )

    def _fail_precision(**_kwargs):
        raise OrderPrecisionError("ORDER_FAIL", "precision_failed")

    monkeypatch.setattr(channel_detector, "_precision_with_retry", _fail_precision)
    result = run(
        "BTCUSDT",
        snapshot,
        {"qty": 1.0},
        env,
        exchange=exchange,
    )
    assert result["action"] == "reject"
    assert result["reason"] == "precision_error_on_entry"
