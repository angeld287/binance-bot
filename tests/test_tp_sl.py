import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Configurar porcentajes de prueba
os.environ['TAKE_PROFIT_PCT'] = '1.5'
os.environ['STOP_LOSS_PCT'] = '1'

from bot_trading import FuturesBot
from bot_trading import config_por_moneda

config_por_moneda["TEST/USDT"] = {"atr_factor": 1.0}


class DummyExchange:
    def __init__(self, position_amt):
        self.position_amt = position_amt
        self.orders = []

    def futures_change_leverage(self, symbol, leverage):
        pass

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

    def futures_position_information(self, symbol):
        return [{"positionAmt": str(self.position_amt), "entryPrice": "100"}]

    def futures_get_open_orders(self, symbol):
        return []

    def futures_create_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"orderId": len(self.orders)}


def test_long_tp_sl():
    ex = DummyExchange(1)
    bot = FuturesBot(ex, "TEST/USDT")
    bot.verificar_y_configurar_tp_sl()
    tp_order, sl_order = ex.orders
    assert tp_order["type"] == "LIMIT"
    assert sl_order["type"] == "STOP_MARKET"
    assert tp_order["price"] == 101.5
    assert sl_order["stopPrice"] == 99.0


def test_short_tp_sl():
    ex = DummyExchange(-1)
    bot = FuturesBot(ex, "TEST/USDT")
    bot.verificar_y_configurar_tp_sl()
    tp_order, sl_order = ex.orders
    assert tp_order["type"] == "LIMIT"
    assert sl_order["type"] == "STOP_MARKET"
    assert tp_order["price"] == 98.5
    assert sl_order["stopPrice"] == 101.0


if __name__ == "__main__":
    test_long_tp_sl()
    test_short_tp_sl()
    print("Tests OK")
