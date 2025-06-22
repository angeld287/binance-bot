import os
import time
import ccxt
import json
import logging
import math
from dotenv import load_dotenv

import pandas as pd

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


load_dotenv()

# Configuración del logger
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

    def abrir_posicion(self, side, amount):
        try:
            position_side = "LONG" if side == "buy" else "SHORT"

            price = self.exchange.fetch_ticker(self.symbol)["last"]
            notional = amount * price
            precision = self.exchange.markets[self.symbol]["precision"]["amount"]
            if notional < 100:
                amount = 100 / price
                step = 10 ** -precision
                amount = math.ceil(amount / step) * step
                log(
                    f"Futuros: Ajustando cantidad a {amount} para cumplir el notional mínimo"
                )

            amount = float(self.exchange.amount_to_precision(self.symbol, amount))
            log(f"Futuros: Cantidad redondeada a {amount}")

            order = self.exchange.create_market_order(self.symbol, side, amount)

            entry_price = float(order['info'].get('avgFillPrice') or order['price'])
            log(f"Futuros: Posición {side} abierta a {entry_price}")
            guardar_posicion(self.pos_file, {
                "side": side,
                "amount": amount,
                "entry_price": entry_price
            })
        except Exception as e:
            log(f"Futuros: Error al abrir posición: {e}")

    
    def cerrar_posicion(self):
        try:
            pos = cargar_posicion(self.pos_file)
            if pos and hasattr(self, "_actualizar_summary"):
                try:
                    self._actualizar_summary(pos)
                except Exception as e:
                    log(f"Futuros: Error actualizando summary: {e}")
            eliminar_posicion(self.pos_file)
        except Exception as e:
            log(f"Futuros: Error al cerrar posición: {e}")

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

    def _actualizar_summary(self, pos):
        summary = {"ganancia_total": 0.0}

        if os.path.exists(self.summary_file):
            with open(self.summary_file, "r") as f:
                summary = json.load(f)

        try:
            if not all(k in pos for k in ["entry_price", "amount"]):
                log(f"Futuros: No se puede actualizar summary: claves faltantes en {pos}")
                return

            # Simula 1% de ganancia/pérdida
            profit = pos["entry_price"] * pos["amount"] * 0.01
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
    amount = 0.001

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

    bot = FuturesBot(exchange, symbol, leverage=5)

    while True:
        pos = cargar_posicion(bot.pos_file)
        log("Modo TESTNET activado")
        if pos:
            bot.evaluar_posicion()
        else:
            nueva_direccion = "buy" if int(time.time()) % 2 == 0 else "sell"
            bot.abrir_posicion(nueva_direccion, amount)
        time.sleep(60)

if __name__ == "__main__":
    main()
