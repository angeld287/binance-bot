from __future__ import annotations

import os
import sys
import types
from decimal import Decimal

import pytest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, "src"))

_config_stub = types.ModuleType("config")
_settings_stub = types.ModuleType("config.settings")


class _Settings:
    def __init__(self, **kwargs):
        self.BINANCE_API_KEY = kwargs.get("BINANCE_API_KEY", "")
        self.BINANCE_API_SECRET = kwargs.get("BINANCE_API_SECRET", "")
        self.BINANCE_TESTNET = kwargs.get("BINANCE_TESTNET", False)
        self.HTTP_TIMEOUT = kwargs.get("HTTP_TIMEOUT", 30)
        self.FILTERS_CACHE_TTL_MIN = kwargs.get("FILTERS_CACHE_TTL_MIN", 5)
        self.RISK_NOTIONAL_USDT = kwargs.get("RISK_NOTIONAL_USDT", 0.0)

    def get(self, key, default=None):  # pragma: no cover - compatibility shim
        return getattr(self, key, default)


_settings_stub.Settings = _Settings
_config_stub.settings = _settings_stub
sys.modules.setdefault("config", _config_stub)
sys.modules.setdefault("config.settings", _settings_stub)

from adapters.brokers.binance import BinanceBroker
from config.settings import Settings


class _DummyClient:
    def __init__(self, *_, **__):
        self.last_order: dict | None = None

    def futures_create_order(self, **kwargs):
        self.last_order = kwargs
        return {"orderId": "1", **kwargs}

    # Stubs required by the broker interface but unused in these tests
    def futures_cancel_order(self, **kwargs):  # pragma: no cover - unused helper
        return {}

    def futures_get_order(self, **kwargs):  # pragma: no cover - unused helper
        return {}

    def futures_exchange_info(self):  # pragma: no cover - unused helper
        return {"symbols": []}


@pytest.fixture
def broker(monkeypatch) -> BinanceBroker:
    monkeypatch.setattr("adapters.brokers.binance.Client", _DummyClient)
    settings = Settings(BINANCE_API_KEY="k", BINANCE_API_SECRET="s")
    return BinanceBroker(settings)


def _filters(**overrides):
    base = {
        "PRICE_FILTER": {"tickSize": "0.05"},
        "LOT_SIZE": {"stepSize": "0.01", "minQty": "0.01"},
        "MARKET_LOT_SIZE": {"stepSize": "0.01", "minQty": "0.01"},
        "MIN_NOTIONAL": {"notional": "0"},
    }
    base.update(overrides)
    return base


def test_limit_order_rounds_price_and_qty(monkeypatch, broker):
    monkeypatch.setattr(broker, "get_symbol_filters", lambda _: _filters())

    broker.place_entry_limit("ADA/USDT", "BUY", 12.3456, 0.1234, "cid-1")

    payload = broker._client.last_order  # type: ignore[attr-defined]
    assert payload["price"] == "12.3"
    assert payload["quantity"] == "0.12"


def test_min_notional_adjusts_quantity(monkeypatch, broker):
    filters = _filters(MIN_NOTIONAL={"notional": "5"})
    monkeypatch.setattr(broker, "get_symbol_filters", lambda _: filters)

    broker.place_entry_limit("ADA/USDT", "BUY", 12.3, 0.1, "cid-2")

    payload = broker._client.last_order  # type: ignore[attr-defined]
    qty_dec = Decimal(payload["quantity"])
    assert qty_dec == Decimal("0.41")
    assert Decimal(payload["price"]) * qty_dec >= Decimal("5")


def test_stop_order_uses_tick_rounding(monkeypatch, broker):
    monkeypatch.setattr(broker, "get_symbol_filters", lambda _: _filters())

    broker.place_stop_reduce_only("ADA/USDT", "SELL", 12.3456, 0.5, "cid-3")

    payload = broker._client.last_order  # type: ignore[attr-defined]
    assert payload["stopPrice"] == "12.35"
    assert payload["quantity"] == "0.5"
