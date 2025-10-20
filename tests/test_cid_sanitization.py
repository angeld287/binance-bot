import os
import sys
import types

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

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

from common.utils import sanitize_client_order_id
from adapters.brokers.binance import BinanceBroker
from config.settings import Settings


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.last_params = None

    def futures_create_order(self, **kwargs):
        self.last_params = kwargs
        return {}

    def futures_cancel_order(self, **kwargs):
        self.last_params = kwargs
        return {}

    def futures_get_order(self, **kwargs):
        self.last_params = kwargs
        return {}

    def futures_exchange_info(self):  # pragma: no cover - simple stub
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {
                            "filterType": "LOT_SIZE",
                            "stepSize": "0.001",
                            "minQty": "0.001",
                        },
                        {"filterType": "MIN_NOTIONAL", "notional": "0"},
                    ],
                }
            ]
        }


RAW = "CID??!!0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # longer than 36 with invalid chars
EXPECTED = sanitize_client_order_id(RAW)


def test_sanitize_function_basic():
    assert sanitize_client_order_id("ABC$%") == "ABC--"
    assert sanitize_client_order_id("x" * 40) == "x" * 36
    assert sanitize_client_order_id(RAW) == EXPECTED
    # Deterministic
    assert sanitize_client_order_id(RAW) == sanitize_client_order_id(RAW)


def test_broker_uses_sanitized_ids(monkeypatch):
    monkeypatch.setattr("adapters.brokers.binance.Client", DummyClient)
    settings = Settings(BINANCE_API_KEY="k", BINANCE_API_SECRET="s")
    broker = BinanceBroker(settings)

    broker.place_entry_limit("BTC/USDT", "BUY", 1.0, 1.0, RAW)
    assert broker._client.last_params["newClientOrderId"] == EXPECTED

    broker.place_stop_reduce_only("BTC/USDT", "BUY", 1.0, 1.0, RAW)
    assert broker._client.last_params["newClientOrderId"] == EXPECTED

    broker.place_tp_reduce_only("BTC/USDT", "SELL", 1.0, 1.0, RAW)
    assert broker._client.last_params["newClientOrderId"] == EXPECTED

    broker.place_entry_market("BTC/USDT", "BUY", 1.0, RAW)
    assert broker._client.last_params["newClientOrderId"] == EXPECTED

    broker.get_order("BTC/USDT", clientOrderId=RAW)
    assert broker._client.last_params["origClientOrderId"] == EXPECTED

    broker.cancel_order("BTC/USDT", clientOrderId=RAW)
    assert broker._client.last_params["origClientOrderId"] == EXPECTED
