import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from adapters.brokers.binance import BinanceBroker
from config.settings import Settings


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.created = []

    def futures_create_order(self, **kwargs):
        self.created.append(kwargs)
        return {"orderId": len(self.created)}

    def futures_exchange_info(self):
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5"},
                    ],
                }
            ]
        }


def _make_broker(monkeypatch):
    monkeypatch.setattr("adapters.brokers.binance.Client", DummyClient)
    settings = Settings(BINANCE_API_KEY="k", BINANCE_API_SECRET="s")
    return BinanceBroker(settings)


def test_precheck_rejects_invalid_precision(monkeypatch, caplog):
    broker = _make_broker(monkeypatch)
    caplog.set_level(logging.INFO, logger="adapters.brokers.binance")

    result = broker.place_entry_limit("BTC/USDT", "BUY", "1.05", "1", "cid-precision")

    assert result["status"] == "rejected"
    assert "invalid_precision:price" in result["reason"]
    assert broker._client.created == []
    assert "ORDER_REJECT_DECIMALS" in caplog.text


def test_precheck_rejects_notional(monkeypatch, caplog):
    broker = _make_broker(monkeypatch)
    caplog.set_level(logging.INFO, logger="adapters.brokers.binance")

    result = broker.place_entry_limit("BTC/USDT", "BUY", "1", "0.001", "cid-notional")

    assert result["status"] == "rejected"
    assert result["reason"] == "notional_below_min"
    assert broker._client.created == []
    assert "ORDER_REJECT_NOTIONAL" in caplog.text


def test_precheck_allows_valid_order(monkeypatch, caplog):
    broker = _make_broker(monkeypatch)
    caplog.set_level(logging.INFO, logger="adapters.brokers.binance")

    result = broker.place_entry_limit("BTC/USDT", "BUY", "10", "1", "cid-valid")

    assert result.get("orderId") == 1
    assert len(broker._client.created) == 1
    assert "ORDER_PRECHECK" in caplog.text
