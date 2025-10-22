from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
import sys
import types
from datetime import datetime
from pathlib import Path

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
        return PrecisionResult(
            price_requested=price,
            qty_requested=qty,
            stop_requested=None,
            side=kwargs.get("side", ""),
            order_type=kwargs.get("order_type", ""),
            filters=kwargs.get("filters"),
            exchange=kwargs.get("exchange"),
            symbol=kwargs.get("symbol", ""),
            price_adjusted=price,
            qty_adjusted=qty,
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

from strategies.ParallelChannelFormation import channel_detector
from strategies.ParallelChannelFormation.channel_detector import ChannelEnv, MarketSnapshot, STRATEGY_NAME, run
from strategies.wedge_formation.strategy import OrderPrecisionError


class FakeExchange:
    def __init__(self) -> None:
        self._open_orders: list[dict] = []
        self.position: dict | None = None
        self.tp_orders: list[dict] = []
        self.entry_orders: list[dict] = []

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
        self.entry_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "price": price,
                "qty": qty,
                "clientOrderId": clientOrderId,
            }
        )
        return {"status": "NEW"}

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
    storage: dict[str, dict[str, float]] = {}

    def fake_load(symbol: str):
        entry = storage.get(symbol)
        if not entry:
            return None
        return entry.get("tp_value")

    def fake_persist(symbol: str, tp_value: float, timestamp: float):
        storage[symbol] = {"tp_value": float(tp_value), "timestamp": float(timestamp)}
        return True

    monkeypatch.setattr(channel_detector, "load_tp_value", fake_load)
    monkeypatch.setattr(channel_detector, "persist_tp_value", fake_persist)
    return storage


def _env() -> ChannelEnv:
    return ChannelEnv(
        tolerance_slope=0.2,
        min_touches=1,
        min_vertical_gap_pct=0.0,
        min_duration_bars=0,
        confidence_threshold=0.0,
        tp_mode="opuesto_inmediato",
        sl_enabled=False,
        price_tick_override=None,
        qty_step_override=None,
        min_notional_buffer_pct=0.0,
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
    stored = store.get("BTCUSDT")
    assert stored is not None
    assert math.isclose(stored["tp_value"], result["tp1"], rel_tol=1e-9)


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
