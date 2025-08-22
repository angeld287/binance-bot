import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Configuración común
os.environ['TAKE_PROFIT_PCT'] = '1.5'
os.environ['STOP_LOSS_PCT'] = '1'

from bot_trading import FuturesBot, config_por_moneda
import bot_trading

config_por_moneda["TEST/USDT"] = {"atr_factor": 1.0}


class DummyExchange:
    def __init__(self, orders, price, position_amt=0):
        self.open_orders = orders
        self.price = price
        self.position_amt = position_amt
        self.cancelled = []
        self.created = []

    def futures_get_open_orders(self, symbol):
        return list(self.open_orders)

    def futures_symbol_ticker(self, symbol):
        return {"price": str(self.price)}

    def futures_cancel_order(self, symbol, orderId=None, origClientOrderId=None):
        if orderId is None and origClientOrderId is not None:
            target = next((o for o in self.open_orders if o.get("clientOrderId") == origClientOrderId), None)
            if target:
                orderId = target["orderId"]
            else:
                self.cancelled.append(origClientOrderId)
                return
        self.cancelled.append(orderId)
        self.open_orders = [o for o in self.open_orders if o["orderId"] != orderId]

    def futures_create_order(self, **kwargs):
        self.created.append(kwargs)
        return {"orderId": len(self.created) + 1}

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


def test_cancel_by_ttl():
    bot_trading.ORDER_META_BY_CID.clear()
    bot_trading.ORDER_META_BY_OID.clear()
    now = int(time.time() * 1000)
    order = {
        "orderId": 1,
        "status": "NEW",
        "type": "LIMIT",
        "side": "BUY",
        "price": "100",
        "origQty": "1",
        "executedQty": "0",
        "time": now - 120000,
        "clientOrderId": "test"
    }
    ex = DummyExchange([order], price=100)
    bot = FuturesBot(ex, "TEST/USDT")
    bot_trading.PENDING_TTL_MIN = 1
    bot_trading.PENDING_USE_SR3 = False
    bot_trading.PENDING_CANCEL_CONFIRM_BARS = 1
    bot_trading.PENDING_MAX_GAP_BPS = 80
    bot.revisar_ordenes_pendientes()
    assert 1 in ex.cancelled


def test_cancel_by_distance():
    bot_trading.ORDER_META_BY_CID.clear()
    bot_trading.ORDER_META_BY_OID.clear()
    now = int(time.time() * 1000)
    order = {
        "orderId": 2,
        "status": "NEW",
        "type": "LIMIT",
        "side": "BUY",
        "price": "100",
        "origQty": "1",
        "executedQty": "0",
        "time": now,
        "clientOrderId": "test"
    }
    ex = DummyExchange([order], price=110)
    bot = FuturesBot(ex, "TEST/USDT")
    bot_trading.PENDING_TTL_MIN = 10
    bot_trading.PENDING_USE_SR3 = False
    bot_trading.PENDING_CANCEL_CONFIRM_BARS = 1
    bot_trading.PENDING_MAX_GAP_BPS = 80
    bot.revisar_ordenes_pendientes()
    assert 2 in ex.cancelled


def test_cancel_by_sr3_buy():
    bot_trading.ORDER_META_BY_CID.clear()
    bot_trading.ORDER_META_BY_OID.clear()
    now = int(time.time() * 1000)
    cid = "bot-1-deadbeef"
    order = {
        "orderId": 3,
        "status": "NEW",
        "type": "LIMIT",
        "side": "BUY",
        "price": "100",
        "origQty": "1",
        "executedQty": "0",
        "time": now,
        "clientOrderId": cid
    }
    bot_trading.ORDER_META_BY_CID[cid] = {"sr3S": "90", "sr3R": "105", "srasof": 1, "ttl": 10, "cfm": 0, "base_id": cid}
    ex = DummyExchange([order], price=106)
    bot = FuturesBot(ex, "TEST/USDT")
    bot_trading.PENDING_TTL_MIN = 10
    bot_trading.PENDING_USE_SR3 = True
    bot_trading.PENDING_SR_BUFFER_BPS = 15
    bot_trading.PENDING_CANCEL_CONFIRM_BARS = 1
    bot.revisar_ordenes_pendientes()
    assert 3 in ex.cancelled


def test_cancel_by_sr3_sell():
    bot_trading.ORDER_META_BY_CID.clear()
    bot_trading.ORDER_META_BY_OID.clear()
    now = int(time.time() * 1000)
    cid = "bot-1-cafebabe"
    order = {
        "orderId": 4,
        "status": "NEW",
        "type": "LIMIT",
        "side": "SELL",
        "price": "100",
        "origQty": "1",
        "executedQty": "0",
        "time": now,
        "clientOrderId": cid
    }
    bot_trading.ORDER_META_BY_CID[cid] = {"sr3S": "95", "sr3R": "105", "srasof": 1, "ttl": 10, "cfm": 0, "base_id": cid}
    ex = DummyExchange([order], price=94)
    bot = FuturesBot(ex, "TEST/USDT")
    bot_trading.PENDING_TTL_MIN = 10
    bot_trading.PENDING_USE_SR3 = True
    bot_trading.PENDING_SR_BUFFER_BPS = 15
    bot_trading.PENDING_CANCEL_CONFIRM_BARS = 1
    bot.revisar_ordenes_pendientes()
    assert 4 in ex.cancelled


def test_partial_fill_adjusts_tp_sl():
    bot_trading.ORDER_META_BY_CID.clear()
    bot_trading.ORDER_META_BY_OID.clear()
    now = int(time.time() * 1000)
    order = {
        "orderId": 5,
        "status": "NEW",
        "type": "LIMIT",
        "side": "BUY",
        "price": "100",
        "origQty": "1",
        "executedQty": "0.5",
        "time": now - 120000,
        "clientOrderId": "test"
    }
    ex = DummyExchange([order], price=100, position_amt=0.5)
    bot = FuturesBot(ex, "TEST/USDT")
    bot_trading.PENDING_TTL_MIN = 1
    bot_trading.PENDING_USE_SR3 = False
    bot_trading.PENDING_CANCEL_CONFIRM_BARS = 1
    bot.revisar_ordenes_pendientes()
    assert 5 in ex.cancelled
    # TP y SL creados para la cantidad ejecutada
    assert len(ex.created) == 2
    qties = {o.get('quantity') for o in ex.created}
    assert qties == {0.5}
