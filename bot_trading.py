# -*- coding: utf-8 -*-
import os
import time
from binance.client import Client
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
            klines = exchange.futures_klines(
                symbol=symbol.replace("/", ""), interval=tf, limit=50
            )
            ohlcv = [
                [k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
                for k in klines
            ]
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
            symbol = self.symbol.replace("/", "")
            self.exchange.futures_change_leverage(symbol=symbol, leverage=self.leverage)
            log(f"🚀 FUTUROS - APALANCAMIENTO ESTABLECIDO EN {self.leverage}X 🚀")
        except Exception as e:
            log(f"Futuros: Error al establecer apalancamiento: {e}")

    def obtener_posicion_abierta(self):
        """Devuelve la posición abierta actual o None si no hay."""
        try:
            symbol = self.symbol.replace("/", "")
            info = self.exchange.futures_position_information(symbol=symbol)
            pos = info[0] if isinstance(info, list) else info

            if pos is None:
                return None
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
            orders = self.exchange.futures_get_open_orders(symbol=self.symbol.replace("/", ""))
            if orders:
                return orders[0]
        except Exception as e:
            log(f"Futuros: Error consultando orden abierta: {e}")
        return None

    def tiene_orden_abierta(self):
        return bool(self.obtener_orden_abierta())

    def verificar_y_configurar_tp_sl(self, tp_pct=0.02, sl_pct=0.01):
        """Verifica y coloca las órdenes de TP y SL si no existen."""
        pos = self.obtener_posicion_abierta()
        if not pos:
            return

        try:
            amt = float(pos.get("positionAmt", 0))
            entry_price = float(pos.get("entryPrice", 0))
        except Exception:
            log("Futuros: Datos de posición no disponibles para configurar TP/SL")
            return

        side = "buy" if amt > 0 else "sell"
        amount = abs(amt)
        close_side = "sell" if side == "buy" else "buy"

        tp_price = entry_price * (1 + tp_pct) if side == "buy" else entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 - sl_pct) if side == "buy" else entry_price * (1 + sl_pct)

        try:
            symbol = self.symbol.replace("/", "")
            open_orders = self.exchange.futures_get_open_orders(symbol=symbol)
        except Exception as e:
            log(f"Futuros: Error obteniendo órdenes abiertas: {e}")
            open_orders = []

        has_tp = False
        has_sl = False
        for o in open_orders:
            o_type = (o.get("type") or o.get("info", {}).get("type", "")).upper()
            o_side = o.get("side") or o.get("info", {}).get("side")
            if o_side and o_side.lower() == close_side.lower():
                if o_type == "LIMIT":
                    has_tp = True
                if "STOP" in o_type:
                    has_sl = True

        if not has_tp:
            try:
                self.exchange.futures_create_order(
                    symbol=self.symbol.replace("/", ""),
                    side=close_side.upper(),
                    type="LIMIT",
                    quantity=amount,
                    price=tp_price,
                    timeInForce="GTC",
                    reduceOnly="true",
                )
                log(f"Futuros: Take Profit colocado en {tp_price}")
            except Exception as e:
                log(f"Futuros: Error al colocar TP: {e}")

        if not has_sl:
            try:
                self.exchange.futures_create_order(
                    symbol=self.symbol.replace("/", ""),
                    side=close_side.upper(),
                    type="STOP_MARKET",
                    quantity=amount,
                    stopPrice=sl_price,
                    reduceOnly="true",
                )
                log(f"Futuros: Stop Loss colocado en {sl_price}")
            except Exception as e:
                log(f"Futuros: Error al colocar SL: {e}")

    def abrir_posicion(self, side, amount, price, price_range):
        """Abre una posición con una orden límite que permanece activa mientras el precio
        esté dentro del rango especificado."""
        try:
            log(f"Futuros: Orden límite {side} {amount} @ {price}")
            order = self.exchange.futures_create_order(
                symbol=self.symbol.replace("/", ""),
                side=side.upper(),
                type="LIMIT",
                quantity=amount,
                price=price,
                timeInForce="GTC",
            )
            log("Futuros: Orden límite creada y permanece activa")

            rango_inf, rango_sup = price_range

            while True:
                time.sleep(5)
                info = self.exchange.futures_get_order(
                    symbol=self.symbol.replace("/", ""), orderId=order["orderId"]
                )
                status = info.get("status", "").lower()
                if status == 'closed':
                    entry_price = float(info.get('avgPrice') or info.get('price', 0))
                    open_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    stop_loss = entry_price * 0.98 if side == "buy" else entry_price * 1.02
                    log(
                        f"Futuros: Posición {side} abierta a {entry_price} a las {open_time}"
                    )
                    break
                if status == 'canceled':
                    log("Futuros: Orden cancelada externamente")
                    break

                ticker = self.exchange.futures_ticker_price(symbol=self.symbol.replace("/", ""))
                price_now = float(ticker['price'])

                if side == 'buy' and rango_inf is not None and price_now < rango_inf:
                    log(
                        f"Futuros: Precio {price_now} fuera del rango de análisis. Cancelando orden límite"
                    )
                    try:
                        self.exchange.futures_cancel_order(
                            symbol=self.symbol.replace("/", ""), orderId=order["orderId"]
                        )
                    finally:
                        break
                if side == 'sell' and rango_sup is not None and price_now > rango_sup:
                    log(
                        f"Futuros: Precio {price_now} fuera del rango de análisis. Cancelando orden límite"
                    )
                    try:
                        self.exchange.futures_cancel_order(
                            symbol=self.symbol.replace("/", ""), orderId=order["orderId"]
                        )
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
                self.exchange.futures_create_order(
                    symbol=self.symbol.replace("/", ""),
                    side=close_side.upper(),
                    type="MARKET",
                    quantity=amount,
                    reduceOnly="true",
                )
            except Exception:
                self.exchange.futures_create_order(
                    symbol=self.symbol.replace("/", ""),
                    side=close_side.upper(),
                    type="MARKET",
                    quantity=amount,
                )

            exit_price = float(
                self.exchange.futures_ticker_price(symbol=self.symbol.replace("/", ""))["price"]
            )

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
                info = self.exchange.futures_premium_index(symbol=self.symbol.replace("/", ""))
                price = float(info["markPrice"])
            except Exception:
                ticker = self.exchange.futures_ticker_price(symbol=self.symbol.replace("/", ""))
                price = float(ticker["price"])

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
    ticker = exchange.futures_ticker_price(symbol=symbol.replace("/", ""))
    info = exchange.futures_exchange_info()
    symbol_info = next(
        (s for s in info["symbols"] if s["symbol"] == symbol.replace("/", "")),
        {},
    )

    price = float(ticker["price"])
    decimals = symbol_info.get("quantityPrecision", 3)
    amount = (110 * leverage) / price
    amount = round(amount, decimals)

    print(f"Precio actual de {symbol}: {price}")

    if bot.tiene_posicion_abierta():
        bot.verificar_y_configurar_tp_sl()
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

    exchange = Client(key, secret, testnet=testnet)
    if testnet:
        print("Modo TESTNET activado")

    bot = FuturesBot(exchange, symbol, leverage=leverage)

    price = _run_iteration(exchange, bot, testnet, symbol, leverage)

    return {
        "statusCode": 200,
        "body": json.dumps({"price": price}),
    }


