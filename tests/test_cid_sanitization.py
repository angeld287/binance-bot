import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

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
