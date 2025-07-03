# -*- coding: utf-8 -*-
import os
import time
import ccxt
import json
import logging
import math
from dotenv import load_dotenv
from patterns import detect_patterns


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
                return "buy", last_high, patterns
            if last_low and price < last_low:
                patterns = detect_patterns(ohlcv)
                return "sell", last_low, patterns
        except Exception:
            continue
    return None, None, []


load_dotenv()

# Configuración del logger
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def log(msg):
    print(msg)
    logging.info(msg)

def cargar_posicion(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def guardar_posicion(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def eliminar_posicion(path):
    if os.path.exists(path):
        os.remove(path)

class FuturesBot:
    def __init__(self, exchange, symbol, leverage=5):
        self.exchange = exchange
        self.symbol = symbol
        self.leverage = leverage
        self.pos_file = "open_positions_futures.json"
        self.summary_file = "summary_futures.json"
        self._set_leverage()

    def _set_leverage(self):
        try:
            self.exchange.set_leverage(self.leverage, self.symbol)
            log(f"Futuros: Apalancamiento establecido a {self.leverage}x")
        except Exception as e:
            log(f"Futuros: Error al establecer apalancamiento: {e}")

    def abrir_posicion(self, side, amount, price):
        """Abre una posición con una orden límite y define stop loss."""
        try:
            log(f"Futuros: Orden límite {side} {amount} @ {price}")
            order = self.exchange.create_limit_order(self.symbol, side, amount, price)
            time.sleep(5)
            info = self.exchange.fetch_order(order['id'], self.symbol)
            if info.get('status') != 'closed':
                log("Futuros: Orden límite no ejecutada")
                try:
                    self.exchange.cancel_order(order['id'], self.symbol)
                except Exception:
                    pass
                return

            entry_price = float(info.get('average') or info.get('price'))
            open_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            stop_loss = entry_price * 0.98 if side == "buy" else entry_price * 1.02
            log(
                f"Futuros: Posición {side} abierta a {entry_price} a las {open_time}"
            )
            guardar_posicion(
                self.pos_file,
                {
                    "side": side,
                    "amount": amount,
                    "entry_price": entry_price,
                    "open_time": open_time,
                    "stop_loss": stop_loss,
                },
            )
        except Exception as e:
            log(f"Futuros: Error al abrir posición: {e}")

    
    def cerrar_posicion(self):
        pos = cargar_posicion(self.pos_file)
        if not pos:
            log("Futuros: No hay posición para cerrar")
            eliminar_posicion(self.pos_file)
            return

        try:
            side = pos.get("side")
            amount = pos.get("amount", 0)
            entry_price = pos.get("entry_price")

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
                    self._actualizar_summary(pos, exit_price)
                except Exception as e:
                    log(f"Futuros: Error actualizando summary: {e}")
        except Exception as e:
            log(f"Futuros: Error al cerrar posición: {e}")
        finally:
            eliminar_posicion(self.pos_file)

    def evaluar_posicion(self):
        pos = cargar_posicion(self.pos_file)
        if not pos:
            log(f"Futuros: Error al evaluar posición: no se pudo cargar la posicion")
            return

        try:
            try:
                market_id = self.exchange.market_id(self.symbol)
                info = self.exchange.fapiPublic_get_premiumIndex({"symbol": market_id})
                price = float(info["markPrice"])
            except Exception:
                ticker = self.exchange.fetch_ticker(self.symbol)
                price = ticker["last"]

            # Validación de claves esperadas
            if not all(k in pos for k in ["entry_price", "side"]):
                log(f"Futuros: Error al evaluar posición: claves faltantes en {pos}")
                return

            entry = pos["entry_price"]
            side = pos["side"]
            amount = pos.get("amount", 0)
            tp = entry * 1.01 if side == "buy" else entry * 0.99
            sl = pos.get("stop_loss", entry * 0.99 if side == "buy" else entry * 1.01)

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


def main():
    key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_API_SECRET")
    testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    symbol = "BTC/USDT"
    leverage=5

    exchange = ccxt.binance({
        'apiKey': key,
        'secret': secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future'
        }
    })

    if testnet:
        exchange.set_sandbox_mode(True)
        log("Modo TESTNET activado")

    bot = FuturesBot(exchange, symbol, leverage=leverage)

    while True:
        ticker = exchange.fetch_ticker('BTC/USDT')
        markets = exchange.load_markets()
        hedgeMode = exchange.fapiPrivateGetPositionSideDual()


        price = ticker['last']
        precision = markets[symbol]["precision"]["amount"]
        decimales = abs(int(round(math.log10(precision))))  # → 5
        amount = (110*leverage) / price
        amount = round(amount, decimales)

        pos = cargar_posicion(bot.pos_file)
        if pos:
            bot.evaluar_posicion()
        else:
            side, level, patterns = detectar_breakout(exchange, symbol)
            if side:
                order_price = level * 0.999 if side == "buy" else level * 1.001
                if patterns:
                    log(f"Patrones detectados: {', '.join(patterns)}")
                bot.abrir_posicion(side, amount, order_price)
            else:
                if testnet:
                    log(f"TESTNET activo - Sin breakout - Apalancamiento: {leverage}x")
                else:
                    log("Sin breakout identificado")
        time.sleep(60)

if __name__ == "__main__":
    main()
