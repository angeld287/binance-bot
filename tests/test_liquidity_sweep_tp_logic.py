import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.append("src")
MODULE_PATH = Path("src/strategies/liquidity_sweep/strategy.py")
SPEC = importlib.util.spec_from_file_location("liquidity_sweep_strategy", MODULE_PATH)
strategy = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader  # pragma: no cover - sanity check
SPEC.loader.exec_module(strategy)  # type: ignore[attr-defined]
do_tick = strategy.do_tick


class DummySettings(dict):
    def get(self, key, default=None):  # noqa: A003
        return super().get(key, default)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - default path
            raise AttributeError(item) from exc


class StubExchange:
    def __init__(self, symbol, *, open_orders=None, orders=None, filters=None, balance=1000.0):
        self.symbol = symbol
        self._open_orders = [dict(o) for o in (open_orders or [])]
        self._orders = {cid: dict(data) for cid, data in (orders or {}).items()}
        self._filters = filters or {}
        self._balance = balance
        self.tp_orders: list[dict] = []
        self.sl_orders: list[dict] = []
        self.cancelled: list[str] = []

    def open_orders(self, symbol):
        assert symbol == self.symbol
        return [dict(o) for o in self._open_orders]

    def get_order(self, symbol, clientOrderId):
        assert symbol == self.symbol
        info = self._orders.get(clientOrderId)
        return dict(info) if info is not None else None

    def cancel_order(self, symbol, clientOrderId):
        assert symbol == self.symbol
        self.cancelled.append(clientOrderId)
        self._open_orders = [o for o in self._open_orders if o.get("clientOrderId") != clientOrderId]
        if clientOrderId in self._orders:
            self._orders[clientOrderId]["status"] = "CANCELED"

    def get_symbol_filters(self, symbol):
        assert symbol == self.symbol
        return self._filters

    def get_available_balance_usdt(self):
        return self._balance

    def place_stop_market(self, symbol, side, *, stopPrice, closePosition, workingType, clientOrderId):  # noqa: N802
        assert symbol == self.symbol
        self.sl_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "stopPrice": stopPrice,
                "clientOrderId": clientOrderId,
            }
        )
        self._open_orders.append(
            {"clientOrderId": clientOrderId, "stopPrice": stopPrice, "side": side}
        )
        self._orders[clientOrderId] = {"status": "NEW", "stopPrice": stopPrice, "side": side}

    def place_tp_reduce_only(self, symbol, side, price, qty, clientOrderId):  # noqa: N802
        assert symbol == self.symbol
        self.tp_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "price": price,
                "qty": qty,
                "clientOrderId": clientOrderId,
            }
        )
        if not any(o.get("clientOrderId") == clientOrderId for o in self._open_orders):
            self._open_orders.append({"clientOrderId": clientOrderId, "price": price, "side": side})
        self._orders[clientOrderId] = {"status": "NEW", "price": price, "side": side}


class StubMarketData:
    def __init__(self, candles):
        self.candles = candles
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, limit):  # noqa: D401
        self.calls += 1
        return self.candles


def _setup(symbol="TESTUSDT"):
    tick_filters = {
        "PRICE_FILTER": {"tickSize": "0.0001"},
        "LOT_SIZE": {"minQty": "0.001", "stepSize": "0.001"},
        "MIN_NOTIONAL": {"notional": "5"},
    }
    settings = DummySettings(
        {
            "SYMBOL": symbol,
            "TP_USE_OPPOSITE_CANCELED_FIRST": True,
            "MAX_LOOKBACK_MIN": 10,
            "RISK_PCT": 0.01,
        }
    )
    return tick_filters, settings


def _trade_ids(symbol):
    open_at_ny = datetime(2024, 1, 2, 9, 30, tzinfo=ZoneInfo("America/New_York"))
    open_at_epoch_ms = int(open_at_ny.astimezone(ZoneInfo("UTC")).timestamp() * 1000)
    trade_id = f"{symbol}-{open_at_ny.strftime('%Y%m%d')}-NY"
    return trade_id, open_at_epoch_ms


def test_long_tp_uses_opposite_cancelled_price():
    symbol = "TESTUSDT"
    trade_id, open_at_epoch_ms = _trade_ids(symbol)
    cid_buy = f"{trade_id}:pre:buy"
    cid_sell = f"{trade_id}:pre:sell"
    cid_tp = f"{trade_id}:tp"
    filters, settings = _setup(symbol)
    exchange = StubExchange(
        symbol,
        open_orders=[{"clientOrderId": cid_sell, "price": "2.8815"}],
        orders={
            cid_buy: {"status": "FILLED", "avgPrice": "2.8757"},
            cid_sell: {"status": "NEW", "price": "2.8815"},
        },
        filters=filters,
    )
    event = {
        "open_at_epoch_ms": open_at_epoch_ms,
        "S": 2.87,
        "R": 2.89,
        "microbuffer": 0.0002,
        "buffer_sl": 0.0004,
        "atr1m": 0.001,
    }

    result = do_tick(exchange, symbol, settings, None, event)

    assert exchange.tp_orders
    assert exchange.tp_orders[0]["price"] == pytest.approx(2.8815)
    assert result["tp"] == pytest.approx(2.8815)
    assert result["state"]["opposite_canceled"]["price"] == pytest.approx(2.8815)
    assert result["state"]["opposite_canceled"]["side"] == "SHORT"
    assert exchange.tp_orders[0]["clientOrderId"] == cid_tp


def test_short_tp_uses_opposite_cancelled_price():
    symbol = "TESTUSDT"
    trade_id, open_at_epoch_ms = _trade_ids(symbol)
    cid_buy = f"{trade_id}:pre:buy"
    cid_sell = f"{trade_id}:pre:sell"
    cid_tp = f"{trade_id}:tp"
    filters, settings = _setup(symbol)
    exchange = StubExchange(
        symbol,
        open_orders=[{"clientOrderId": cid_buy, "price": "2.8668"}],
        orders={
            cid_sell: {"status": "FILLED", "avgPrice": "2.8757"},
            cid_buy: {"status": "NEW", "price": "2.8668"},
        },
        filters=filters,
    )
    event = {
        "open_at_epoch_ms": open_at_epoch_ms,
        "S": 2.8668,
        "R": 2.89,
        "microbuffer": 0.0002,
        "buffer_sl": 0.0004,
        "atr1m": 0.001,
    }

    result = do_tick(exchange, symbol, settings, None, event)

    assert exchange.tp_orders
    assert exchange.tp_orders[0]["price"] == pytest.approx(2.8668)
    assert result["tp"] == pytest.approx(2.8668)
    assert result["state"]["opposite_canceled"]["price"] == pytest.approx(2.8668)
    assert result["state"]["opposite_canceled"]["side"] == "LONG"
    assert exchange.tp_orders[0]["clientOrderId"] == cid_tp


def test_long_tp_uses_resistance_when_no_opposite():
    symbol = "TESTUSDT"
    trade_id, open_at_epoch_ms = _trade_ids(symbol)
    cid_buy = f"{trade_id}:pre:buy"
    filters, settings = _setup(symbol)
    exchange = StubExchange(
        symbol,
        open_orders=[],
        orders={cid_buy: {"status": "FILLED", "avgPrice": "2.8757"}},
        filters=filters,
    )
    event = {
        "open_at_epoch_ms": open_at_epoch_ms,
        "S": 2.87,
        "R": 2.90,
        "microbuffer": 0.0002,
        "buffer_sl": 0.0004,
        "atr1m": 0.001,
    }

    result = do_tick(exchange, symbol, settings, None, event)

    assert result["tp"] == pytest.approx(2.90)
    assert exchange.tp_orders[0]["price"] == pytest.approx(2.90)


def test_short_tp_uses_support_when_no_opposite():
    symbol = "TESTUSDT"
    trade_id, open_at_epoch_ms = _trade_ids(symbol)
    cid_sell = f"{trade_id}:pre:sell"
    filters, settings = _setup(symbol)
    exchange = StubExchange(
        symbol,
        open_orders=[],
        orders={cid_sell: {"status": "FILLED", "avgPrice": "2.8757"}},
        filters=filters,
    )
    event = {
        "open_at_epoch_ms": open_at_epoch_ms,
        "S": 2.866,
        "R": 2.89,
        "microbuffer": 0.0002,
        "buffer_sl": 0.0004,
        "atr1m": 0.001,
    }

    result = do_tick(exchange, symbol, settings, None, event)

    assert result["tp"] == pytest.approx(2.866)
    assert exchange.tp_orders[0]["price"] == pytest.approx(2.866)


def test_tp_order_is_idempotent():
    symbol = "TESTUSDT"
    trade_id, open_at_epoch_ms = _trade_ids(symbol)
    cid_buy = f"{trade_id}:pre:buy"
    cid_sell = f"{trade_id}:pre:sell"
    filters, settings = _setup(symbol)
    exchange = StubExchange(
        symbol,
        open_orders=[{"clientOrderId": cid_sell, "price": "2.8815"}],
        orders={
            cid_buy: {"status": "FILLED", "avgPrice": "2.8757"},
            cid_sell: {"status": "NEW", "price": "2.8815"},
        },
        filters=filters,
    )
    event = {
        "open_at_epoch_ms": open_at_epoch_ms,
        "S": 2.87,
        "R": 2.89,
        "microbuffer": 0.0002,
        "buffer_sl": 0.0004,
        "atr1m": 0.001,
    }

    do_tick(exchange, symbol, settings, None, event)
    assert len(exchange.tp_orders) == 1

    # Re-run with the same state and existing TP order
    result = do_tick(exchange, symbol, settings, None, event)
    assert len(exchange.tp_orders) == 1
    assert result["tp"] == pytest.approx(2.8815)


def test_tp_respects_tick_rounding():
    symbol = "TESTUSDT"
    trade_id, open_at_epoch_ms = _trade_ids(symbol)
    cid_buy = f"{trade_id}:pre:buy"
    filters, settings = _setup(symbol)
    exchange = StubExchange(
        symbol,
        open_orders=[],
        orders={cid_buy: {"status": "FILLED", "avgPrice": "2.8757"}},
        filters=filters,
    )
    event = {
        "open_at_epoch_ms": open_at_epoch_ms,
        "S": 2.87,
        "R": 2.88156,
        "microbuffer": 0.0002,
        "buffer_sl": 0.0004,
        "atr1m": 0.001,
    }

    result = do_tick(exchange, symbol, settings, None, event)

    assert result["tp"] == pytest.approx(2.8815)
    assert exchange.tp_orders[0]["price"] == pytest.approx(2.8815)


def test_directional_fetches_nearest_from_market_data():
    symbol = "TESTUSDT"
    trade_id, open_at_epoch_ms = _trade_ids(symbol)
    cid_buy = f"{trade_id}:pre:buy"
    filters, settings = _setup(symbol)
    settings["MAX_LOOKBACK_MIN"] = 5
    exchange = StubExchange(
        symbol,
        open_orders=[],
        orders={cid_buy: {"status": "FILLED", "avgPrice": "2.8757"}},
        filters=filters,
    )
    # Candles arranged so compute_levels fallback returns price_now +- microbuffer
    candles = [
        [0, 0, 2.88, 2.86, 2.875, 0],
        [1, 0, 2.88, 2.86, 2.875, 0],
        [2, 0, 2.88, 2.86, 2.875, 0],
        [3, 0, 2.88, 2.86, 2.875, 0],
        [4, 0, 2.88, 2.86, 2.875, 0],
    ]
    market_data = StubMarketData(candles)
    event = {
        "open_at_epoch_ms": open_at_epoch_ms,
        "S": None,
        "R": None,
        "microbuffer": 0.0002,
        "buffer_sl": 0.0004,
        "atr1m": 0.001,
    }

    result = do_tick(exchange, symbol, settings, market_data, event)

    assert market_data.calls == 1
    assert result["tp"] > result["entry"]
    assert exchange.tp_orders[0]["price"] > result["entry"]

