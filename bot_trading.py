# -*- coding: utf-8 -*-
import os
import time
import ccxt
import json
import logging
import math
from dotenv import load_dotenv
from pattern_detection import detect_patterns


def _last_swing_high(ohlcv, window=3):
    highs = [c[2] for c in ohlcv]
    for i in range(len(highs) - window - 1, window, -1):
        local = highs[i - window : i + window + 1]
        if highs[i] == max(local):
            return highs[i]
    return None


def _last_swing_low(ohlcv, window=3):
    lows = [c[3] for c in ohlcv]
    for i in range(len(lows) - window - 1, window, -1):
        local = lows[i - window : i + window + 1]
        if lows[i] == min(local):
            return lows[i]
    return None


def detectar_breakout(exchange, symbol):
    """Busca rupturas de los últimos máximos o mínimos en 15m y 30m."""
    for tf in ["15m", "30m"]:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=50)
            if len(ohlcv) < 20:
                continue
            price = ohlcv[-1][4]
            last_high = _last_swing_high(ohlcv[:-1])
            last_low = _last_swing_low(ohlcv[:-1])
            if last_high and price > last_high:
                patterns = detect_patterns(ohlcv)
                price_range = (last_low if last_low else price * 0.99, last_high)
                return "buy", last_high, patterns, price_range
            if last_low and price < last_low:
                patterns = detect_patterns(ohlcv)
                price_range = (last_low, last_high if last_high else price * 1.01)
                return "sell", last_low, patterns, price_range
        except Exception:
            continue
    return None, None, [], (None, None)


load_dotenv()

# Configuración del logger para AWS Lambda
logger = logging.getLogger("bot")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def log(msg):
    logger.info(msg)



class FuturesBot:
    def __init__(self, exchange, symbol, leverage=5):
        self.exchange = exchange
        self.symbol = symbol
        self.leverage = leverage
        self.summary_file = "summary_futures.json"
        self._set_leverage()

    def _set_leverage(self):
        try:
            self.exchange.set_leverage(self.leverage, self.symbol)
            log(f"🚀 FUTUROS - APALANCAMIENTO ESTABLECIDO EN {self.leverage}X 🚀")
        except Exception as e:
            log(f"Futuros: Error al establecer apalancamiento: {e}")

    def obtener_posicion_abierta(self):
        """Devuelve la posición abierta actual o None si no hay."""
        try:
            market_id = self.exchange.market_id(self.symbol)
            info = self.exchange.fapiPrivate_get_positionrisk({"symbol": market_id})
            # El endpoint puede devolver lista o dict
            pos = info[0] if isinstance(info, list) else info
            amt = float(pos.get("positionAmt", 0))
            if amt != 0:
                return pos
        except Exception as e:
            log(f"Futuros: Error consultando posición: {e}")
        return None

    def tiene_posicion_abierta(self):
        return self.obtener_posicion_abierta() is not None

    def obtener_orden_abierta(self):
        """Devuelve la primera orden abierta o None si no hay."""
        try:
            orders = self.exchange.fetch_open_orders(self.symbol)
            if orders:
                return orders[0]
        except Exception as e:
            log(f"Futuros: Error consultando orden abierta: {e}")
        return None

    def tiene_orden_abierta(self):
        return bool(self.obtener_orden_abierta())

    def abrir_posicion(self, side, amount, price, price_range):
        """Abre una posición con una orden límite que permanece activa mientras el precio
        esté dentro del rango especificado."""
        try:
            log(f"Futuros: Orden límite {side} {amount} @ {price}")
            order = self.exchange.create_limit_order(self.symbol, side, amount, price)
            log("Futuros: Orden límite creada y permanece activa")

            rango_inf, rango_sup = price_range

            while True:
                time.sleep(5)
                info = self.exchange.fetch_order(order['id'], self.symbol)
                status = info.get('status')
                if status == 'closed':
                    entry_price = float(info.get('average') or info.get('price'))
                    open_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    stop_loss = entry_price * 0.98 if side == "buy" else entry_price * 1.02
                    log(
                        f"Futuros: Posición {side} abierta a {entry_price} a las {open_time}"
                    )
                    break
                if status == 'canceled':
                    log("Futuros: Orden cancelada externamente")
                    break

                ticker = self.exchange.fetch_ticker(self.symbol)
                price_now = ticker['last']

                if side == 'buy' and rango_inf is not None and price_now < rango_inf:
                    log(
                        f"Futuros: Precio {price_now} fuera del rango de análisis. Cancelando orden límite"
                    )
                    try:
                        self.exchange.cancel_order(order['id'], self.symbol)
                    finally:
                        break
                if side == 'sell' and rango_sup is not None and price_now > rango_sup:
                    log(
                        f"Futuros: Precio {price_now} fuera del rango de análisis. Cancelando orden límite"
                    )
                    try:
                        self.exchange.cancel_order(order['id'], self.symbol)
                    finally:
                        break

        except Exception as e:
            log(f"Futuros: Error al abrir posición: {e}")

    
    def cerrar_posicion(self):
        pos = self.obtener_posicion_abierta()
        if not pos:
            log("Futuros: No hay posición para cerrar")
            return

        try:
            amt = float(pos.get("positionAmt", 0))
            side = "buy" if amt > 0 else "sell"
            amount = abs(amt)
            entry_price = float(pos.get("entryPrice", 0))

            close_side = "sell" if side == "buy" else "buy"
            try:
                self.exchange.create_market_order(
                    self.symbol, close_side, amount, {"reduceOnly": True}
                )
            except Exception:
                self.exchange.create_market_order(self.symbol, close_side, amount)

            exit_price = self.exchange.fetch_ticker(self.symbol)["last"]

            profit = None
            if side and entry_price is not None:
                if side == "buy":
                    profit = (exit_price - entry_price) * amount
                else:
                    profit = (entry_price - exit_price) * amount

                resultado = "ganancia" if profit >= 0 else "pérdida"
                log(
                    f"Futuros: Posición {'LONG' if side == 'buy' else 'SHORT'} cerrada a {exit_price} con {resultado} {profit}"
                )

            if hasattr(self, "_actualizar_summary"):
                try:
                    data = {
                        "entry_price": entry_price,
                        "amount": amount,
                        "side": side,
                    }
                    self._actualizar_summary(data, exit_price)
                except Exception as e:
                    log(f"Futuros: Error actualizando summary: {e}")
        except Exception as e:
            log(f"Futuros: Error al cerrar posición: {e}")

    def evaluar_posicion(self):
        pos = self.obtener_posicion_abierta()
        if not pos:
            log("Futuros: No hay posición abierta para evaluar")
            return

        try:
            try:
                market_id = self.exchange.market_id(self.symbol)
                info = self.exchange.fapiPublic_get_premiumIndex({"symbol": market_id})
                price = float(info["markPrice"])
            except Exception:
                ticker = self.exchange.fetch_ticker(self.symbol)
                price = ticker["last"]

            entry = float(pos.get("entryPrice", 0))
            amt = float(pos.get("positionAmt", 0))
            side = "buy" if amt > 0 else "sell"
            amount = abs(amt)
            tp = entry * 1.01 if side == "buy" else entry * 0.99
            sl = entry * 0.99 if side == "buy" else entry * 1.01

            if side == "buy":
                if price >= tp:
                    pnl = (price - entry) * amount
                    log(f"TP alcanzado. Cerrando posición en {price} con PNL {pnl}")
                    self.cerrar_posicion()
                elif price <= sl:
                    pnl = (price - entry) * amount
                    log(f"Stop Loss alcanzado. Cerrando posición en {price} con PNL {pnl}")
                    self.cerrar_posicion()
                else:
                    log(f"Futuros: Precio actual: {price} — TP: {tp}, SL: {sl} -->")
            else:
                if price <= tp:
                    pnl = (entry - price) * amount
                    log(f"TP alcanzado. Cerrando posición en {price} con PNL {pnl}")
                    self.cerrar_posicion()
                elif price >= sl:
                    pnl = (entry - price) * amount
                    log(f"Stop Loss alcanzado. Cerrando posición en {price} con PNL {pnl}")
                    self.cerrar_posicion()
                else:
                    log(f"Futuros: Precio actual: {price} — TP: {tp}, SL: {sl} -->")

        except Exception as e:
            log(f"Futuros: Error al evaluar posición: {e}")

    def _actualizar_summary(self, pos, exit_price):
        summary = {"ganancia_total": 0.0}

        if os.path.exists(self.summary_file):
            with open(self.summary_file, "r") as f:
                summary = json.load(f)

        try:
            if not all(k in pos for k in ["entry_price", "amount", "side"]):
                log(f"Futuros: No se puede actualizar summary: claves faltantes en {pos}")
                return

            entry = pos["entry_price"]
            amount = pos["amount"]
            side = pos["side"]

            if side == "buy":
                profit = (exit_price - entry) * amount
            else:  # sell/short
                profit = (entry - exit_price) * amount

            summary["ganancia_total"] += profit

            with open(self.summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            log(f"Futuros: Ganancia acumulada: {summary['ganancia_total']}")

        except Exception as e:
            log(f"Futuros: Error actualizando summary: {e}")


def _run_iteration(exchange, bot, testnet, symbol, leverage):
    ticker = exchange.fetch_ticker(symbol)
    markets = exchange.load_markets()

    price = ticker["last"]
    precision = markets[symbol]["precision"]["amount"]
    decimals = abs(int(round(math.log10(precision))))
    amount = (110 * leverage) / price
    amount = round(amount, decimals)

    print(f"Precio actual de {symbol}: {price}")

    if bot.tiene_posicion_abierta():
        bot.evaluar_posicion()
    elif bot.tiene_orden_abierta():
        log("Orden pendiente detectada, esperando ejecución o cancelación.")
    else:
        side, level, patterns, rango = detectar_breakout(exchange, symbol)
        if side:
            order_price = level * 0.999 if side == "buy" else level * 1.001
            if patterns:
                print(f"Patrones detectados: {', '.join(patterns)}")
            bot.abrir_posicion(side, amount, order_price, rango)
        else:
            if testnet:
                print(f"TESTNET activo - Sin breakout - Apalancamiento: {leverage}x")
            else:
                print("Sin breakout identificado")

    return price


def handler(event, context):
    """AWS Lambda handler que ejecuta una iteración de trading."""
    load_dotenv()

    key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_API_SECRET")
    testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

    symbol = "BTC/USDT"
    leverage = 5

    exchange = ccxt.binance(
        {
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
    )

    if testnet:
        exchange.set_sandbox_mode(True)
        print("Modo TESTNET activado")

    bot = FuturesBot(exchange, symbol, leverage=leverage)

    price = _run_iteration(exchange, bot, testnet, symbol, leverage)

    return {
        "statusCode": 200,
        "body": json.dumps({"price": price}),
    }


