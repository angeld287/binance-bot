import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from core.bot_trading import FuturesBot, config_por_moneda, IDEMPOTENCY_REGISTRY
import core.bot_trading as bot_trading

config_por_moneda["TEST/USDT"] = {"atr_factor": 1.0}


class DummyExchange:
    def __init__(self, open_orders=None):
        self.open_orders = open_orders or []
        self.created = []

    def futures_exchange_info(self):
        return {
            "symbols": [
                {
                    "symbol": "TESTUSDT",
                    "quantityPrecision": 3,
                    "pricePrecision": 1,
                    "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.1"}],
                }
            ]
        }

    def futures_change_leverage(self, symbol, leverage):
        pass

    def futures_get_open_orders(self, symbol):
        return list(self.open_orders)

    def futures_create_order(self, **kwargs):
        self.created.append(kwargs)
        order = {
            "orderId": len(self.created),
            "price": kwargs.get("price"),
            "origQty": str(kwargs.get("quantity")),
            "side": kwargs.get("side"),
            "type": kwargs.get("type"),
        }
        self.open_orders.append(order)
        return {"orderId": len(self.created)}

    def futures_position_information(self, symbol):
        return [{"positionAmt": "0", "entryPrice": "0"}]


def _setup():
    bot_trading.ORDER_META_BY_CID.clear()
    bot_trading.ORDER_META_BY_OID.clear()
    IDEMPOTENCY_REGISTRY.clear()
    bot_trading.get_sr_levels = lambda s, t: {}


def test_precheck_avoids_duplicate():
    _setup()
    existing = [{"orderId": 1, "side": "BUY", "type": "LIMIT", "price": "100", "origQty": "1"}]
    ex = DummyExchange(existing)
    bot = FuturesBot(ex, "TEST/USDT")
    bot.abrir_posicion("BUY", 1, 100, (90, 110))
    assert len(ex.created) == 0


def test_idempotency_prevents_recreation():
    _setup()
    ex = DummyExchange([])
    bot = FuturesBot(ex, "TEST/USDT")
    bot.abrir_posicion("BUY", 1, 100, (90, 110))
    assert len(ex.created) == 1
    ex.open_orders.clear()
    bot.abrir_posicion("BUY", 1, 100, (90, 110))
    assert len(ex.created) == 1

