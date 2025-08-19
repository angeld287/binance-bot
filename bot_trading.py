# -*- coding: utf-8 -*-
import os
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


def ajustar_precio(precio, tick_size, price_precision=6, direction="floor"):
    """Redondea el precio respetando el tick size."""
    try:
        if not price_precision or price_precision < 1:
            price_precision = 6

        precio = float(precio)
        tick_size = float(tick_size)
        pasos = precio / tick_size
        if direction == "ceil":
            pasos = math.ceil(pasos)
        else:
            pasos = math.floor(pasos)
        return round(pasos * tick_size, price_precision)
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
                #log(f"Llamada {name} | entorno: {env} | usando proxy: {proxy_msg}")
                return attr(*args, **kwargs)

            return wrapper
        return attr


ANALYSIS_WINDOW = 12


def _last_swing_high(ohlcv, window=ANALYSIS_WINDOW):
    highs = [c[2] for c in ohlcv]
    for i in range(len(highs) - window - 1, window, -1):
        local = highs[i - window : i + window + 1]
        if highs[i] == max(local):
            return highs[i]
    return None


def _last_swing_low(ohlcv, window=ANALYSIS_WINDOW):
    lows = [c[3] for c in ohlcv]
    for i in range(len(lows) - window - 1, window, -1):
        local = lows[i - window : i + window + 1]
        if lows[i] == min(local):
            return lows[i]
    return None


def detectar_breakout(exchange, symbol, window=ANALYSIS_WINDOW):
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
            if len(ohlcv) < window * 2 + 2:
                continue
            price = ohlcv[-1][4]
            last_high = _last_swing_high(ohlcv[:-1], window=window)
            last_low = _last_swing_low(ohlcv[:-1], window=window)
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

formatter = logging.Formatter("%(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def log(msg):
    logger.info(msg)


# Utilidades para leer porcentajes desde el entorno
def _get_pct_env(var, alt_var, default_decimal):
    """Devuelve el porcentaje en formato decimal (0.01=1%)."""
    val = os.getenv(var)
    if val is None and alt_var:
        val = os.getenv(alt_var)
    if val is None:
        return default_decimal
    try:
        pct = float(val)
    except ValueError:
        raise ValueError(f"Valor inválido para {var}: {val}")
    if pct <= 0:
        raise ValueError(f"{var} debe ser mayor que 0")
    return pct / 100.0

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
    def __init__(self, exchange, symbol, leverage=5, use_breakout_dynamic_stops=False):
        self.exchange = exchange
        self.symbol = symbol
        self.leverage = leverage
        self.use_breakout_dynamic_stops = use_breakout_dynamic_stops
        self.summary_file = "summary_futures.json"
        self.sl_order_id = None
        self.tp_order_id = None
        self.tick_size = None
        self.min_profit_sl_moved = False
        self.limit_order_info = None
        self._set_leverage()
        self._init_precisions()

    def _symbol_config(self):
        cfg = config_por_moneda.get(self.symbol, {})
        tp_default = float(cfg.get("take_profit_pct", 0.02))
        sl_default = float(cfg.get("stop_loss_pct", 0.01))
        tp_pct = _get_pct_env("TAKE_PROFIT_PCT", "TAKE_PROFIT_PERCENT", tp_default)
        sl_pct = _get_pct_env("STOP_LOSS_PCT", "STOP_LOSS_PERCENT", sl_default)
        return {
            "apalancamiento": int(cfg.get("apalancamiento", 3)),
            "atr_factor": float(cfg.get("atr_factor", 1.5)),
            "stop_loss_pct": sl_pct,
            "take_profit_pct": tp_pct,
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
            self.price_precision = int(s_info.get("pricePrecision", 6))
            if self.price_precision < 1:
                log(
                    f"[DEBUG] pricePrecision inválido {self.price_precision}. Se ajusta a 6"
                )
                self.price_precision = 6

            filters = s_info.get("filters", [])
            price_filter = next(
                (f for f in filters if f.get("filterType") == "PRICE_FILTER"),
                {},
            )
            symbol = sym
            if symbol.upper() == "DOGEUSDT":
                tick_size = 1e-06
            else:
                tick_size = float(price_filter.get("tickSize", 0))
            expected_tick = 1 / (10 ** s_info.get("pricePrecision", 6))

            # Validación
            if not tick_size or tick_size <= 0 or not math.isclose(
                tick_size, expected_tick, rel_tol=0.001
            ):
                log(
                    f"Advertencia: tick_size {tick_size} inválido para {self.symbol}. Se ajusta a {expected_tick}"
                )
                tick_size = expected_tick

            self.tick_size = tick_size
        except Exception as e:
            log(f"Futuros: Error obteniendo precision: {e}")
            self.quantity_precision = 3
            self.price_precision = 6
            self.tick_size = None

    def _fmt_qty(self, qty):
        return float(f"{qty:.{self.quantity_precision}f}")

    def _fmt_price(self, price, rounding="floor"):
        try:
            if price is None or float(price) <= 0:
                return price
            price_precision = (
                self.price_precision if self.price_precision and self.price_precision > 0 else 6
            )
            if self.tick_size:
                if float(price) < self.tick_size:
                    log(
                        f"Advertencia: precio {price} menor que tick_size {self.tick_size}"
                    )
                    log(
                        f"Diagnóstico: tick_size {self.tick_size} es mayor que el precio {price} antes de ajustar"
                    )
                    return None
                return ajustar_precio(price, self.tick_size, price_precision, rounding)
            log(f"[DEBUG] Precisión final a usar: {price_precision}")
            return round(float(price), price_precision)
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

    def verificar_orden_limit(self):
        """Valida la orden límite activa y la cancela si sale del rango."""
        info = self.limit_order_info
        if not info:
            try:
                orders = self.exchange.futures_get_open_orders(symbol=self.symbol.replace("/", ""))
                if not orders:
                    return
                order = orders[0]
                order_id = order.get("orderId")
                side = order.get("side", "").lower()
                rango_inf = None
                rango_sup = None
                # Guardar información básica para siguientes iteraciones
                self.limit_order_info = {
                    "orderId": order_id,
                    "side": side,
                    "rango_inf": rango_inf,
                    "rango_sup": rango_sup,
                }
            except Exception:
                return
        else:
            order_id = info.get("orderId")
            side = info.get("side")
            rango_inf = info.get("rango_inf")
            rango_sup = info.get("rango_sup")

        try:
            order = self.exchange.futures_get_order(
                symbol=self.symbol.replace("/", ""), orderId=order_id
            )
            status = (order.get("status") or "").lower()

            if status in ("filled", "closed"):
                self.limit_order_info = None
                self.min_profit_sl_moved = False
                return

            if status == "canceled":
                log("Futuros: Orden cancelada externamente")
                self.limit_order_info = None
                return

            ticker = self.exchange.futures_symbol_ticker(
                symbol=self.symbol.replace("/", "")
            )
            price_now = float(ticker["price"])

            if side == "buy" and rango_inf is not None and price_now < rango_inf:
                log(
                    f"Futuros: Precio {price_now} fuera del rango de análisis. Cancelando orden límite"
                )
                self.exchange.futures_cancel_order(
                    symbol=self.symbol.replace("/", ""), orderId=order_id
                )
                self.limit_order_info = None

            if side == "sell" and rango_sup is not None and price_now > rango_sup:
                log(
                    f"Futuros: Precio {price_now} fuera del rango de análisis. Cancelando orden límite"
                )
                self.exchange.futures_cancel_order(
                    symbol=self.symbol.replace("/", ""), orderId=order_id
                )
                self.limit_order_info = None

        except Exception as e:
            log(f"Futuros: Error validando orden límite: {e}")

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

        notional = amount_f * entry_price
        if notional < 5:
            log(f"Futuros: Notional {notional} menor al mínimo requerido")
            return

        tp_price_f = self._fmt_price(tp_price, rounding="ceil")
        sl_price_f = self._fmt_price(sl_price, rounding="floor")

        log(
            "Futuros: TP%={:.3f}, SL%={:.3f} -> TP={}, SL={}, side={}, reduceOnly=true".format(
                tp_pct * 100, sl_pct * 100, tp_price_f, sl_price_f, side
            )
        )

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

            # Guardar información de la orden para validaciones posteriores
            rango_inf, rango_sup = price_range
            self.limit_order_info = {
                "orderId": order.get("orderId"),
                "side": side,
                "rango_inf": rango_inf,
                "rango_sup": rango_sup,
            }

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

    def cancelar_stop_loss(self):
        """Cancela únicamente la orden de Stop Loss activa, si existe."""
        symbol = self.symbol.replace("/", "")

        oid = getattr(self, "sl_order_id", None)
        if oid:
            try:
                self.exchange.futures_cancel_order(symbol=symbol, orderId=oid)
                log(f"Futuros: Stop Loss {oid} cancelado")
            except Exception as e:
                log(f"Futuros: Error cancelando Stop Loss {oid}: {e}")
            finally:
                self.sl_order_id = None

        try:
            open_orders = self.exchange.futures_get_open_orders(symbol=symbol)
            for o in open_orders:
                o_type = (o.get("type") or o.get("info", {}).get("type", "")).upper()
                if "STOP" in o_type:
                    try:
                        self.exchange.futures_cancel_order(symbol=symbol, orderId=o["orderId"])
                        log(f"Futuros: Stop Loss pendiente {o['orderId']} cancelado")
                    except Exception:
                        pass
        except Exception as e:
            log(f"Futuros: Error cancelando SL pendientes: {e}")

    
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
            self.min_profit_sl_moved = False
        except Exception as e:
            log(f"Futuros: Error al cerrar posición: {e}")

    def evaluar_posicion(self):
        if not self.use_breakout_dynamic_stops:
            return

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
                    if not self.min_profit_sl_moved:
                        log("Futuros: Objetivo mínimo alcanzado. Moviendo Stop Loss")
                        self.cancelar_stop_loss()
                        new_stop = entry * 1.01 * 0.998
                        new_stop_f = self._fmt_price(new_stop)
                        qty = self._fmt_qty(amount)
                        try:
                            order = self.exchange.futures_create_order(
                                symbol=self.symbol.replace("/", ""),
                                side="SELL",
                                type="STOP_MARKET",
                                quantity=qty,
                                stopPrice=new_stop_f,
                                reduceOnly="true",
                            )
                            self.sl_order_id = order.get("orderId")
                            self.min_profit_sl_moved = True
                            log(f"Futuros: Stop Loss movido a {new_stop_f}")
                        except Exception as e:
                            log(f"Futuros: Error ajustando SL: {e}")
                elif price <= sl and not self.min_profit_sl_moved:
                    pnl = (price - entry) * amount
                    log(f"Stop Loss alcanzado. Cerrando posición en {price} con PNL {pnl}")
                    self.cerrar_posicion()
                else:
                    log(f"Futuros: Precio actual: {price} — TP: {tp}, SL: {sl} -->")
            else:
                if price <= tp:
                    if not self.min_profit_sl_moved:
                        log("Futuros: Objetivo mínimo alcanzado. Moviendo Stop Loss")
                        self.cancelar_stop_loss()
                        new_stop = entry * 0.99 * 1.002
                        new_stop_f = self._fmt_price(new_stop)
                        qty = self._fmt_qty(amount)
                        try:
                            order = self.exchange.futures_create_order(
                                symbol=self.symbol.replace("/", ""),
                                side="BUY",
                                type="STOP_MARKET",
                                quantity=qty,
                                stopPrice=new_stop_f,
                                reduceOnly="true",
                            )
                            self.sl_order_id = order.get("orderId")
                            self.min_profit_sl_moved = True
                            log(f"Futuros: Stop Loss movido a {new_stop_f}")
                        except Exception as e:
                            log(f"Futuros: Error ajustando SL: {e}")
                elif price >= sl and not self.min_profit_sl_moved:
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
                if symbol.upper() == "DOGEUSDT":
                    bot.tick_size = 1e-06
                else:
                    bot.tick_size = float(f.get("tickSize"))
            except Exception:
                bot.tick_size = None
            break

    price = float(ticker["price"])
    decimals = symbol_info.get("quantityPrecision", bot.quantity_precision)
    bot.quantity_precision = decimals
    bot.price_precision = symbol_info.get("pricePrecision", bot.price_precision)
    lev = leverage if leverage is not None else bot.leverage

    env_name = os.getenv("ENVIRONMENT") or ("TESTNET" if testnet else "PROD")
    log(f"Par: {symbol} | Precio actual: {price}")

    if bot.tiene_posicion_abierta():
        bot.verificar_y_configurar_tp_sl()
        bot.evaluar_posicion()
    elif bot.tiene_orden_abierta():
        bot.verificar_orden_limit()
        log("Orden pendiente detectada, esperando ejecución o cancelación.")
    else:
        side, level, patterns, rango = detectar_breakout(exchange, symbol)
        if side:
            order_price = level * 0.999 if side == "buy" else level * 1.001
            if order_price is None or order_price <= 0:
                log(f"Futuros: Precio calculado inválido {order_price}. Orden no enviada")
            else:
                order_price = bot._fmt_price(order_price)
                cfg = bot._symbol_config()
                atr_factor = cfg["atr_factor"]
                sl_pct = cfg["stop_loss_pct"] * atr_factor
                sl_price = (
                    order_price * (1 - sl_pct)
                    if side == "buy"
                    else order_price * (1 + sl_pct)
                )
                symbol_key = symbol.replace("/", "")
                if symbol_key == "BTCUSDT":
                    base_amount = 110
                elif symbol_key == "DOGEUSDT":
                    base_amount = 6
                else:
                    base_amount = 110

                amount_raw = (base_amount * lev) / price
                qty = bot._fmt_qty(amount_raw)
                if patterns:
                    print(f"Patrones detectados: {', '.join(patterns)}")
                bot.abrir_posicion(side, qty, order_price, rango)
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
    use_breakout_dynamic_stops = (
        os.getenv("USE_BREAKOUT_DYNAMIC_STOPS", "false").lower() == "true"
    )

    proxies = get_proxies()
    req_params = {"proxies": proxies} if proxies else None
    client = Client(key, secret, testnet=testnet, requests_params=req_params)
    exchange = LoggingClient(client, testnet)
    if testnet:
        print("Modo TESTNET activado")

    bot = FuturesBot(
        exchange,
        symbol,
        leverage=leverage,
        use_breakout_dynamic_stops=use_breakout_dynamic_stops,
    )

    price = _run_iteration(exchange, bot, testnet, symbol)

    return {
        "statusCode": 200,
        "body": json.dumps({"price": price}),
    }


