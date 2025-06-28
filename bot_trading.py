# -*- coding: utf-8 -*-
import os
import time
import ccxt
import json
import logging
import math
from dotenv import load_dotenv


def detectar_punto_resistencia(ohlcv, nivel_actual, margen=0.002):
    altos = [c[2] for c in ohlcv]
    for alto in altos[-20:]:
        if abs(alto - nivel_actual) / nivel_actual < margen:
            return True
    return False

def detectar_patron_vela(ohlcv):
    open_, high, low, close = ohlcv[-1][1:5]
    cuerpo = abs(close - open_)
    mecha_superior = high - max(open_, close)
    mecha_inferior = min(open_, close) - low

    if cuerpo < mecha_superior and cuerpo < mecha_inferior:
        return "doji"
    elif mecha_inferior > cuerpo * 2:
        return "martillo"
    elif mecha_superior > cuerpo * 2:
        return "estrella_fugaz"
    else:
        return None

def validar_entrada(exchange, symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=30)
        price = ohlcv[-1][4]

        resistencia = detectar_punto_resistencia(ohlcv, price)
        patron = detectar_patron_vela(ohlcv)

        if resistencia and patron == "estrella_fugaz":
            return "sell"
        elif not resistencia and patron == "martillo":
            return "buy"
        else:
            return None
    except Exception as e:
        print(f"Error en validar_entrada: {e}")
        return None


def detectar_movimiento_importante(exchange, symbol, umbral=0.01):
    """Devuelve la dirección de entrada cuando la última vela de distintos
    timeframes supera el umbral indicado."""
    for tf in ["1m", "5m", "15m", "30m"]:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=2)
            if len(ohlcv) < 2:
                continue
            open_, close = ohlcv[-1][1], ohlcv[-1][4]
            cambio = (close - open_) / open_
            if abs(cambio) >= umbral:
                return "buy" if cambio < 0 else "sell"
        except Exception:
            continue
    return None


def calcular_punto_rechazo(exchange, symbol, side):
    """Obtiene un punto de soporte o resistencia simple según la dirección."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="5m", limit=20)
        if side == "buy":
            lows = [c[3] for c in ohlcv]
            return min(lows)
        else:
            highs = [c[2] for c in ohlcv]
            return max(highs)
    except Exception:
        return None


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
        """Abre una posición con una orden límite en el punto de rechazo"""
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
            log(f"Futuros: Posición {side} abierta a {entry_price} a las {open_time}")
            guardar_posicion(self.pos_file, {
                "side": side,
                "amount": amount,
                "entry_price": entry_price,
                "open_time": open_time
            })
        except Exception as e:
            log(f"Futuros: Error al abrir posición: {e}")

    
    def cerrar_posicion(self):
        pos = cargar_posicion(self.pos_file)
        if not pos:
            log("Futuros: No hay posición para cerrar")
            eliminar_posicion(self.pos_file)
            return

        try:
            exit_price = self.exchange.fetch_ticker(self.symbol)["last"]
            side = pos.get("side")
            entry_price = pos.get("entry_price")
            amount = pos.get("amount", 0)

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
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker["last"]

            # Validación de claves esperadas
            if not all(k in pos for k in ["entry_price", "side"]):
                log(f"Futuros: Error al evaluar posición: claves faltantes en {pos}")
                return

            entry = pos["entry_price"]
            tp = entry * 1.01
            sl = entry * 0.99

            if pos["side"] == "buy" and (price >= tp or price <= sl):
                log(f"Futuros: Precio actual: {price} — TP: {tp}, SL: {sl}")
                self.cerrar_posicion()

            elif pos["side"] == "sell" and (price <= tp or price >= sl):
                log(f"Futuros: Precio actual: {price} — TP: {tp}, SL: {sl}")
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
        log(f"Price {price} - Precision {precision} - Amount {amount} - Decimales {decimales} - hedgeMode {hedgeMode}")

        pos = cargar_posicion(bot.pos_file)
        log("Modo TESTNET activado")
        if pos:
            bot.evaluar_posicion()
        else:
            direccion = detectar_movimiento_importante(exchange, symbol)
            if direccion:
                punto = calcular_punto_rechazo(exchange, symbol, direccion)
                if punto:
                    bot.abrir_posicion(direccion, amount, punto)
                else:
                    log("No se pudo calcular punto de rechazo")
            else:
                log("Sin movimiento importante, no se abre posición")
        time.sleep(60)

if __name__ == "__main__":
    main()
