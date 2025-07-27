# -*- coding: utf-8 -*-
import os
import time
from binance.client import Client
import json
import logging
import math
from dotenv import load_dotenv
from pattern_detection import detect_patterns


def get_proxies():
    """Devuelve diccionario de proxies o None si no se usa proxy."""
    testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    proxy = os.getenv("PROXY_URL")
    if not testnet and proxy:
        return {"http": proxy, "https": proxy}
    return None


def ajustar_precio(precio, tick_size):
    """Redondea el precio hacia abajo respetando el tick size."""
    try:
        decimales = 0
        if isinstance(tick_size, float) or isinstance(tick_size, int):
            tick_str = f"{tick_size}"
        else:
            tick_str = str(tick_size)
        if "." in tick_str:
            decimales = len(tick_str.rstrip("0").split(".")[1])
        base = math.floor(float(precio) / float(tick_size)) * float(tick_size)
        return float(f"{base:.{decimales}f}")
    except Exception:
        return precio


class LoggingClient:
    """Envuelve un Client para registrar cada request."""

    def __init__(self, client, testnet):
        self._client = client
        self.testnet = testnet

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                proxies = get_proxies()
                env = "testnet" if self.testnet else "producción"
                proxy_msg = "sí" if proxies else "no"
                log(f"Llamada {name} | entorno: {env} | usando proxy: {proxy_msg}")
                return attr(*args, **kwargs)

            return wrapper
        return attr


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


# Configuraciones personalizadas por par de trading
config_por_moneda = {
    "BTC/USDT": {
        "apalancamiento": 5,
        "atr_factor": 1.2,
    },
    "DOGE/USDT": {
        "apalancamiento": 2,
        "atr_factor": 2.0,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.05,
    },
}



class FuturesBot:
    def __init__(self, exchange, symbol, leverage=5):
        self.exchange = exchange
        self.symbol = symbol
        self.leverage = leverage
        self.summary_file = "summary_futures.json"
        self.sl_order_id = None
        self.tp_order_id = None
        self.tick_size = None
        self._set_leverage()
        self._init_precisions()

    def _symbol_config(self):
        cfg = config_por_moneda.get(self.symbol, {})
        return {
            "apalancamiento": int(cfg.get("apalancamiento", 3)),
            "atr_factor": float(cfg.get("atr_factor", 1.5)),
            "stop_loss_pct": float(cfg.get("stop_loss_pct", 0.015)),
            "take_profit_pct": float(cfg.get("take_profit_pct", 0.02)),
        }

    def _set_leverage(self):
        try:
            cfg = self._symbol_config()
            self.leverage = cfg["apalancamiento"]
            symbol = self.symbol.replace("/", "")
            self.exchange.futures_change_leverage(symbol=symbol, leverage=self.leverage)
            log(f"🚀 FUTUROS - APALANCAMIENTO ESTABLECIDO EN {self.leverage}X 🚀")
        except Exception as e:
            log(f"Futuros: Error al establecer apalancamiento: {e}")

    def _init_precisions(self):
        """Obtiene la precisión de cantidad y precio para el par"""
        try:
            info = self.exchange.futures_exchange_info()
            sym = self.symbol.replace("/", "")
            s_info = next(
                (s for s in info.get("symbols", []) if s.get("symbol") == sym),
                {},
            )
            self.quantity_precision = s_info.get("quantityPrecision", 3)
            self.price_precision = s_info.get("pricePrecision", 2)

            filters = s_info.get("filters", [])
            for f in filters:
                if f.get("filterType") == "PRICE_FILTER":
                    try:
                        self.tick_size = float(f.get("tickSize"))
                    except Exception:
                        self.tick_size = None
                    break

            expected_tick = 10 ** (-self.price_precision)
            log(f"Expected tick: {expected_tick}")
            log(f"Actual tick_size: {self.tick_size}")
            log(f"tick_size is None: {self.tick_size is None}")
            log(f"tick_size <= 0: {self.tick_size <= 0}")
            log(f"tick_size mismatch: {not math.isclose(self.tick_size, expected_tick, rel_tol=0.001)}")

            if not self.tick_size or self.tick_size <= 0 or not math.isclose(
                self.tick_size, expected_tick, rel_tol=0.001
            ):
                log(
                    f"Advertencia: tick_size {self.tick_size} inválido para {self.symbol}. "
                    f"Se ajusta a {expected_tick}"
                )
        except Exception as e:
            log(f"Futuros: Error obteniendo precision: {e}")
            self.quantity_precision = 3
            self.price_precision = 2
            self.tick_size = None

    def _fmt_qty(self, qty):
        return float(f"{qty:.{self.quantity_precision}f}")

    def _fmt_price(self, price):
        try:
            if price is None or float(price) <= 0:
                return price
            if self.tick_size:
                if float(price) < self.tick_size:
                    log(
                        f"Advertencia: precio {price} menor que tick_size {self.tick_size}"
                    )
                    log(
                        f"Diagnóstico: tick_size {self.tick_size} es mayor que el precio {price} antes de ajustar"
                    )
                    return None
                price = ajustar_precio(price, self.tick_size)
            return float(f"{float(price):.{self.price_precision}f}")
        except Exception:
            return price

    def obtener_posicion_abierta(self):
        """Devuelve la posición abierta actual o None si no hay."""
        try:
            symbol = self.symbol.replace("/", "")
            info = self.exchange.futures_position_information(symbol=symbol)
            if isinstance(info, list):
                pos = info[0] if len(info) > 0 else None
            else:
                pos = info

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

    def verificar_y_configurar_tp_sl(self, tp_pct=None, sl_pct=None):
        """Verifica y coloca las órdenes de TP y SL si no existen."""
        cfg = self._symbol_config()
        atr_factor = cfg["atr_factor"]
        if tp_pct is None:
            tp_pct = cfg["take_profit_pct"]
        if sl_pct is None:
            sl_pct = cfg["stop_loss_pct"]
        tp_pct *= atr_factor
        sl_pct *= atr_factor
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
        amount_f = self._fmt_qty(amount)
        tp_price_f = self._fmt_price(tp_price)
        sl_price_f = self._fmt_price(sl_price)

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
                    self.tp_order_id = o.get("orderId")
                if "STOP" in o_type:
                    has_sl = True
                    self.sl_order_id = o.get("orderId")

        if not has_tp:
            if tp_price_f is None or tp_price_f <= 0:
                log(f"Futuros: Precio de TP inválido {tp_price_f}. Orden no enviada")
            else:
                try:
                    order = self.exchange.futures_create_order(
                        symbol=self.symbol.replace("/", ""),
                        side=close_side.upper(),
                        type="LIMIT",
                        quantity=amount_f,
                        price=tp_price_f,
                        timeInForce="GTC",
                        reduceOnly="true",
                    )
                    self.tp_order_id = order.get("orderId")
                    log(f"Futuros: Take Profit colocado en {tp_price_f}")
                except Exception as e:
                    log(f"Futuros: Error al colocar TP: {e}")

        if not has_sl:
            if sl_price_f is None or sl_price_f <= 0:
                log(f"Futuros: Precio de SL inválido {sl_price_f}. Orden no enviada")
            else:
                try:
                    order = self.exchange.futures_create_order(
                        symbol=self.symbol.replace("/", ""),
                        side=close_side.upper(),
                        type="STOP_MARKET",
                        quantity=amount_f,
                        stopPrice=sl_price_f,
                        reduceOnly="true",
                    )
                    self.sl_order_id = order.get("orderId")
                    log(f"Futuros: Stop Loss colocado en {sl_price_f}")
                except Exception as e:
                    log(f"Futuros: Error al colocar SL: {e}")

    def abrir_posicion(self, side, amount, price, price_range):
        """Abre una posición con una orden límite que permanece activa mientras el precio
        esté dentro del rango especificado."""
        try:
            if price is None or price <= 0:
                log(f"Futuros: Precio inválido {price}. Orden no enviada")
                return
            qty = self._fmt_qty(amount)
            price_f = self._fmt_price(price)
            if price_f is None or price_f <= 0:
                log(f"Futuros: Precio inválido {price_f}. Orden no enviada")
                return

            monto = None
            try:
                if qty and price_f and qty > 0 and price_f > 0:
                    monto = float(f"{qty * price_f:.4f}")
            except Exception:
                monto = None

            mensaje = f"Futuros: Orden límite {side} {qty} @ {price_f}"
            if monto is not None:
                mensaje += f" | Monto: {monto} USDT"
            log(mensaje)
            order = self.exchange.futures_create_order(
                symbol=self.symbol.replace("/", ""),
                side=side.upper(),
                type="LIMIT",
                quantity=qty,
                price=price_f,
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

                ticker = self.exchange.futures_symbol_ticker(symbol=self.symbol.replace("/", ""))
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

    def cancelar_ordenes_tp_sl(self):
        """Cancela las órdenes de TP y SL registradas y cualquier orden reduceOnly restante."""
        symbol = self.symbol.replace("/", "")

        for attr in ["tp_order_id", "sl_order_id"]:
            oid = getattr(self, attr)
            if oid:
                try:
                    self.exchange.futures_cancel_order(symbol=symbol, orderId=oid)
                    log(f"Futuros: Orden {oid} cancelada")
                except Exception as e:
                    log(f"Futuros: Error cancelando orden {oid}: {e}")
                finally:
                    setattr(self, attr, None)

        try:
            open_orders = self.exchange.futures_get_open_orders(symbol=symbol)
            for o in open_orders:
                if str(o.get("reduceOnly", "")).lower() == "true":
                    try:
                        self.exchange.futures_cancel_order(symbol=symbol, orderId=o["orderId"])
                        log(f"Futuros: Orden pendiente {o['orderId']} cancelada")
                    except Exception:
                        pass
        except Exception as e:
            log(f"Futuros: Error cancelando órdenes pendientes: {e}")

    
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
            qty = self._fmt_qty(amount)
            try:
                self.exchange.futures_create_order(
                    symbol=self.symbol.replace("/", ""),
                    side=close_side.upper(),
                    type="MARKET",
                    quantity=qty,
                    reduceOnly="true",
                )
            except Exception:
                self.exchange.futures_create_order(
                    symbol=self.symbol.replace("/", ""),
                    side=close_side.upper(),
                    type="MARKET",
                    quantity=qty,
                )

            exit_price = float(
                self.exchange.futures_symbol_ticker(symbol=self.symbol.replace("/", ""))["price"]
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

            self.cancelar_ordenes_tp_sl()
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
                ticker = self.exchange.futures_symbol_ticker(symbol=self.symbol.replace("/", ""))
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


def _run_iteration(exchange, bot, testnet, symbol, leverage=None):
    ticker = exchange.futures_symbol_ticker(symbol=symbol.replace("/", ""))
    info = exchange.futures_exchange_info()
    symbol_info = next(
        (s for s in info["symbols"] if s["symbol"] == symbol.replace("/", "")),
        {},
    )

    for f in symbol_info.get("filters", []):
        if f.get("filterType") == "PRICE_FILTER":
            try:
                bot.tick_size = float(f.get("tickSize"))
            except Exception:
                bot.tick_size = None
            break

    price = float(ticker["price"])
    decimals = symbol_info.get("quantityPrecision", bot.quantity_precision)
    bot.quantity_precision = decimals
    bot.price_precision = symbol_info.get("pricePrecision", bot.price_precision)
    lev = leverage if leverage is not None else bot.leverage

    symbol_key = symbol.replace("/", "")
    if symbol_key == "BTCUSDT":
        base_amount = 110
    elif symbol_key == "DOGEUSDT":
        base_amount = 10
    else:
        base_amount = 110

    amount_raw = (base_amount * lev) / price
    amount = bot._fmt_qty(amount_raw)

    env_name = os.getenv("ENVIRONMENT") or ("TESTNET" if testnet else "PROD")
    log(f"Entorno: {env_name} | Par: {symbol} | Precio actual: {price}")

    if bot.tiene_posicion_abierta():
        bot.verificar_y_configurar_tp_sl()
        bot.evaluar_posicion()
    elif bot.tiene_orden_abierta():
        log("Orden pendiente detectada, esperando ejecución o cancelación.")
    else:
        side, level, patterns, rango = detectar_breakout(exchange, symbol)
        if side:
            order_price = level * 0.999 if side == "buy" else level * 1.001
            if order_price is None or order_price <= 0:
                log(f"Futuros: Precio calculado inválido {order_price}. Orden no enviada")
            else:
                order_price = bot._fmt_price(order_price)
                if patterns:
                    print(f"Patrones detectados: {', '.join(patterns)}")
                bot.abrir_posicion(side, amount, order_price, rango)
        else:
            if testnet:
                print(f"TESTNET activo - Sin breakout - Apalancamiento: {lev}x")
            else:
                print("Sin breakout identificado")

    return price


def handler(event, context):
    """AWS Lambda handler que ejecuta una iteración de trading."""
    load_dotenv()

    key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_API_SECRET")
    testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

    symbol = os.getenv("SYMBOL", "BTC/USDT")
    leverage = 5

    proxies = get_proxies()
    req_params = {"proxies": proxies} if proxies else None
    client = Client(key, secret, testnet=testnet, requests_params=req_params)
    exchange = LoggingClient(client, testnet)
    if testnet:
        print("Modo TESTNET activado")

    bot = FuturesBot(exchange, symbol, leverage=leverage)

    price = _run_iteration(exchange, bot, testnet, symbol)

    return {
        "statusCode": 200,
        "body": json.dumps({"price": price}),
    }


