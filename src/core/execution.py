# -*- coding: utf-8 -*-
import os
import json
import math
import time
import hashlib
import re

from binance.client import Client
from binance.exceptions import BinanceAPIException
from analysis.resistance_levels import next_resistances
from analysis.support_levels import next_supports
from analysis.sr_levels import get_sr_levels
from strategies import generate_signal

from .logging_utils import logger, log, debug_log
from .positions import get_current_position_info, has_active_position



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







CID_ALLOWED_RE = re.compile(r"^[A-Za-z0-9-_]+$")
ORDER_META_BY_CID = {}
ORDER_META_BY_OID = {}
IDEMPOTENCY_REGISTRY: dict[str, float] = {}

def _clean_idempotency():
    now = time.time()
    expired = [k for k, v in IDEMPOTENCY_REGISTRY.items() if v < now]
    for k in expired:
        del IDEMPOTENCY_REGISTRY[k]

def register_idempotency(key: str, ttl_min: int):
    _clean_idempotency()
    IDEMPOTENCY_REGISTRY[key] = time.time() + ttl_min * 60

def idempotency_hit(key: str) -> bool:
    _clean_idempotency()
    return key in IDEMPOTENCY_REGISTRY


def generate_client_order_id(params: dict, epoch_ms: int | None = None) -> str:
    """Genera un clientOrderId compacto y determinístico."""
    base = json.dumps(params, sort_keys=True, separators=(",", ":"))
    hfull = hashlib.sha256(base.encode()).hexdigest()
    if epoch_ms is None:
        epoch_ms = int(hfull[:13], 16)
    h = hfull[:8]
    return f"bot-{epoch_ms}-{h}"


def _log_cid(cid: str):
    charset_ok = bool(CID_ALLOWED_RE.fullmatch(cid))
    logger.info(
        "clientOrderId=%s | len=%d | charset_ok=%s",
        cid,
        len(cid),
        charset_ok,
    )


def store_order_metadata(cid: str, order: dict | None, meta: dict):
    if cid:
        ORDER_META_BY_CID[cid] = meta
    if order and order.get("orderId") is not None:
        ORDER_META_BY_OID[str(order["orderId"])] = meta
    logger.info("metadata mapped | cid=%s | oid=%s | data=%s", cid, order.get("orderId") if order else None, meta)


def get_order_metadata(order: dict) -> dict:
    cid = order.get("clientOrderId") or ""
    oid = str(order.get("orderId", ""))
    meta = ORDER_META_BY_CID.get(cid)
    if meta is None:
        meta = ORDER_META_BY_OID.get(oid)
    if meta is None:
        meta = _parse_client_id(cid)
    return meta


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "y")


# Parámetros de monitoreo de órdenes pendientes
PENDING_TTL_MIN = int(os.getenv("PENDING_TTL_MIN", "10"))
PENDING_MAX_GAP_BPS = int(os.getenv("PENDING_MAX_GAP_BPS", "80"))
PENDING_GAP_ATR_MULT = float(os.getenv("PENDING_GAP_ATR_MULT", "0.0"))
PENDING_USE_SR3 = _env_bool("PENDING_USE_SR3", True)
PENDING_SR_BUFFER_BPS = int(os.getenv("PENDING_SR_BUFFER_BPS", "15"))
PENDING_CANCEL_CONFIRM_BARS = int(os.getenv("PENDING_CANCEL_CONFIRM_BARS", "2"))
SR_TIMEFRAME = os.getenv("SR_TIMEFRAME", "15m")
DRY_RUN = _env_bool("DRY_RUN", False)



def _parse_client_id(cid: str):
    data = {}
    if not cid:
        return data
    for part in cid.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            data[k] = v
    return data


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
        raise ValueError(f"❌❌❌❌❌ Valor inválido para {var}: {val}")
    if pct <= 0:
        raise ValueError(f"❌❌❌❌❌ {var} debe ser mayor que 0")
    return pct / 100.0


def compute_trailing_sl_candidate(symbol, position, price, sup_levels, res_levels,
                                  tick_size, price_precision, min_move_ticks, buffer_ticks):
    """
    position: objeto con positionAmt y entryPrice
    Retorna dict con información del SL candidato.
    """
    try:
        amt = float(position.get("positionAmt", 0)) if position else 0.0
    except Exception:
        amt = 0.0
    try:
        entry = float(position.get("entryPrice", 0)) if position else 0.0
    except Exception:
        entry = 0.0
    if amt > 0:
        side = "LONG"
    elif amt < 0:
        side = "SHORT"
    else:
        return {
            "ok": False,
            "side": "LONG",
            "entry": entry,
            "sl_act": None,
            "anchor": entry,
            "sl_cand": entry,
            "reason": "no_position",
        }
    sl_act = None
    try:
        sl_act = float(position.get("stopLossPrice"))
    except Exception:
        sl_act = None
    if sl_act is None:
        try:
            side_close = "SELL" if amt > 0 else "BUY"
            orders = Client().futures_get_open_orders(symbol=symbol.replace("/", ""))
            o = next(
                (
                    o
                    for o in orders
                    if (o.get("reduceOnly") or o.get("closePosition"))
                    and (o.get("type") or "").upper() in ("STOP", "STOP_MARKET")
                    and (o.get("side") or "").upper() == side_close
                ),
                None,
            )
            if o:
                sl_act = float(o.get("stopPrice") or o.get("price"))
        except Exception:
            sl_act = None

    def round_to_tick(p):
        if p is None or tick_size is None:
            return p
        return ajustar_precio(p, tick_size, price_precision)

    if side == "LONG":
        anchor = next((float(s.get("level")) for s in (sup_levels or []) if s.get("level") and float(s.get("level")) > entry), None)
        if anchor is None:
            return {
                "ok": False,
                "side": side,
                "entry": entry,
                "sl_act": sl_act,
                "anchor": entry,
                "sl_cand": entry,
                "reason": "no_valid_level",
            }
        sl_cand = round_to_tick(anchor - buffer_ticks * tick_size)
    else:
        anchor = next((float(r.get("level")) for r in (res_levels or []) if r.get("level") and float(r.get("level")) < entry), None)
        if anchor is None:
            return {
                "ok": False,
                "side": side,
                "entry": entry,
                "sl_act": sl_act,
                "anchor": entry,
                "sl_cand": entry,
                "reason": "no_valid_level",
            }
        sl_cand = round_to_tick(anchor + buffer_ticks * tick_size)

    if sl_act is not None:
        monotonic = sl_cand > sl_act if side == "LONG" else sl_cand < sl_act
        if not monotonic:
            return {
                "ok": False,
                "side": side,
                "entry": entry,
                "sl_act": sl_act,
                "anchor": anchor,
                "sl_cand": sl_cand,
                "reason": "non_monotonic",
            }
        if abs(sl_cand - sl_act) < min_move_ticks * tick_size:
            return {
                "ok": False,
                "side": side,
                "entry": entry,
                "sl_act": sl_act,
                "anchor": anchor,
                "sl_cand": sl_cand,
                "reason": "delta_below_min",
            }

    return {
        "ok": True,
        "side": side,
        "entry": entry,
        "sl_act": sl_act,
        "anchor": anchor,
        "sl_cand": sl_cand,
        "reason": "",
    }


def apply_trailing_sl(exchange, symbol, side, qty, sl_cand, tick_size, price_precision, qty_precision):
    """Coloca/actualiza STOP_MARKET reduceOnly al precio sl_cand."""
    side_close = "SELL" if side == "LONG" else "BUY"
    sym = symbol.replace("/", "")
    try:
        orders = exchange.futures_get_open_orders(symbol=sym)
    except Exception:
        orders = []

    sl_order = next(
        (
            o
            for o in orders
            if (o.get("reduceOnly") or o.get("closePosition"))
            and (o.get("type") or "").upper() in ("STOP", "STOP_MARKET")
            and (o.get("side") or "").upper() == side_close
        ),
        None,
    )

    price_rounded = ajustar_precio(sl_cand, tick_size, price_precision)

    if sl_order:
        try:
            stop_price = float(sl_order.get("stopPrice") or sl_order.get("price") or 0.0)
        except Exception:
            stop_price = 0.0
        if abs(stop_price - price_rounded) < 0.5 * tick_size:
            log("⏭️ SL sin cambios (dif < 0.5t)")
            return
        try:
            exchange.futures_cancel_order(symbol=sym, orderId=sl_order.get("orderId"))
        except Exception as e:
            log(f"⚠️ Error cancelando SL: {e}")

    qty_fmt = float(f"{qty:.{qty_precision}f}")
    try:
        exchange.futures_create_order(
            symbol=sym,
            side=side_close,
            type="STOP_MARKET",
            stopPrice=price_rounded,
            quantity=qty_fmt,
            reduceOnly=True,
        )
        log(f"🔒 SL actualizado → {price_rounded:.6f} ({side_close} STOP_MARKET reduceOnly)")
    except Exception as e:
        log(f"⚠️ Error actualizando SL: {e}")

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
        except Exception as e:
            log(f"❌❌❌❌❌❌❌ Error al establecer apalancamiento: {e}")

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
                debug_log(
                    f"pricePrecision inválido {self.price_precision}. Se ajusta a 6"
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
            log(f"❌❌❌❌❌ Error obteniendo precision: {e}")
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
            debug_log(f"Precisión final a usar: {price_precision}")
            return round(float(price), price_precision)
        except Exception:
            return price

    def obtener_posicion_abierta(self):
        """Devuelve la posición abierta actual o None si no hay."""
        return get_current_position_info(self.exchange, self.symbol)

    def tiene_posicion_abierta(self):
        return has_active_position(self.exchange, self.symbol)

    def obtener_orden_abierta(self):
        """Devuelve la primera orden abierta o None si no hay."""
        try:
            orders = self.exchange.futures_get_open_orders(symbol=self.symbol.replace("/", ""))
            if orders:
                return orders[0]
        except Exception as e:
            log(f"❌❌❌❌❌ Error consultando orden abierta: {e}")
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
            log(f"❌❌❌❌❌ Error validando orden límite: {e}")

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
            log("❌❌❌❌❌ Datos de posición no disponibles para configurar TP/SL")
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

        reduce_only = True

        try:
            current_price = float(
                self.exchange.futures_symbol_ticker(
                    symbol=self.symbol.replace("/", "")
                )["price"]
            )
        except Exception:
            current_price = None

        if entry_price > 0 and current_price is not None:
            pnl_pct = (
                (current_price - entry_price) / entry_price * 100
                if side == "buy"
                else (entry_price - current_price) / entry_price * 100
            )
            log(
                f"Futuros: TP%={tp_pct * 100:.3f}, SL%={sl_pct * 100:.3f} -> TP={tp_price_f}, SL={sl_price_f}, "
                f"side={side}, reduceOnly={reduce_only} | PnL%={pnl_pct:+.2f}% @ precio={current_price:.6f} (entry={entry_price:.6f})"
            )
        else:
            log(
                "Futuros: TP%={:.3f}, SL%={:.3f} -> TP={}, SL={}, side={}, reduceOnly=true".format(
                    tp_pct * 100, sl_pct * 100, tp_price_f, sl_price_f, side
                )
            )

        try:
            symbol = self.symbol.replace("/", "")
            open_orders = self.exchange.futures_get_open_orders(symbol=symbol)
        except Exception as e:
            log(f"❌❌❌❌❌ Error obteniendo órdenes abiertas: {e}")
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
                    log(f"❌❌❌❌❌ Error al colocar TP: {e}")

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
                    log(f"❌❌❌❌❌ Error al colocar SL: {e}")

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

            symbol = self.symbol.replace("/", "")
            try:
                open_orders = self.exchange.futures_get_open_orders(symbol=symbol)
            except Exception as e:
                log(f"❌❌❌❌❌ Error consultando órdenes previas: {e}")
                open_orders = []
            matches = [
                o for o in open_orders
                if (o.get("side", "").upper() == side.upper()
                    and (o.get("type") or o.get("info", {}).get("type", "")).upper() == "LIMIT"
                    and float(o.get("price", 0)) == price_f
                    and float(o.get("origQty") or o.get("quantity") or 0) == qty)
            ]
            log(f"Futuros: pre-check openOrders coincidencias={len(matches)}")
            if matches:
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

            # Snapshot de SR3 para rastrear la orden
            sr = get_sr_levels(self.symbol, SR_TIMEFRAME)
            s3 = sr.get("S", [None, None, None])[2] if sr.get("S") else None
            r3 = sr.get("R", [None, None, None])[2] if sr.get("R") else None
            s3_i = int(s3) if s3 else 0
            r3_i = int(r3) if r3 else 0
            signal_ts = int(time.time() * 1000)
            params = {
                "symbol": symbol,
                "side": side.upper(),
                "type": "LIMIT",
                "price": price_f,
                "quantity": qty,
                "timeInForce": "GTC",
            }
            client_id = generate_client_order_id(params)
            _log_cid(client_id)
            hit = idempotency_hit(client_id)
            log(f"idempotency_key={client_id} | hit={hit}")
            if hit:
                return
            metadata = {
                "sr3S": s3_i,
                "sr3R": r3_i,
                "srasof": sr.get("asof"),
                "ttl": PENDING_TTL_MIN,
                "signal_ts": signal_ts,
                "cfm": 0,
                "base_id": client_id,
            }

            try:
                order = self.exchange.futures_create_order(
                    symbol=symbol,
                    side=side.upper(),
                    type="LIMIT",
                    quantity=qty,
                    price=price_f,
                    timeInForce="GTC",
                    newClientOrderId=client_id,
                )
            except BinanceAPIException as e:
                if e.code == -1022:
                    log("⚠️ Binance -1022, reintentando sin clientOrderId")
                    order = self.exchange.futures_create_order(
                        symbol=symbol,
                        side=side.upper(),
                        type="LIMIT",
                        quantity=qty,
                        price=price_f,
                        timeInForce="GTC",
                    )
                    client_id = ""
                else:
                    raise

            _log_cid(client_id or "")
            store_order_metadata(client_id, order, metadata)
            register_idempotency(client_id, PENDING_TTL_MIN)
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
            log(f"❌❌❌❌❌ Error al abrir posición: {e}")

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
                    log(f"❌❌❌❌❌ Error cancelando orden {oid}: {e}")
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
            log(f"❌❌❌❌❌ Error cancelando órdenes pendientes: {e}")

    def cancelar_stop_loss(self):
        """Cancela únicamente la orden de Stop Loss activa, si existe."""
        symbol = self.symbol.replace("/", "")

        oid = getattr(self, "sl_order_id", None)
        if oid:
            try:
                self.exchange.futures_cancel_order(symbol=symbol, orderId=oid)
                log(f"Futuros: Stop Loss {oid} cancelado")
            except Exception as e:
                log(f"❌❌❌❌❌ Error cancelando Stop Loss {oid}: {e}")
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
            log(f"❌❌❌❌❌ Error cancelando SL pendientes: {e}")


    def revisar_ordenes_pendientes(self):
        """Monitorea órdenes límite abiertas y las cancela si pierden validez."""
        symbol = self.symbol.replace("/", "")
        try:
            orders = self.exchange.futures_get_open_orders(symbol=symbol)
        except Exception as e:
            log(f"❌❌❌❌❌ Error obteniendo órdenes pendientes: {e}")
            return

        now = int(time.time() * 1000)
        for o in orders:
            status = (o.get("status") or "").upper()
            o_type = (o.get("type") or "").upper()
            reduce_only = str(o.get("reduceOnly", "")).lower() == "true"
            if status != "NEW" or o_type != "LIMIT" or reduce_only:
                continue

            side = (o.get("side") or "").upper()
            try:
                entry = float(o.get("price", 0))
                executed = float(o.get("executedQty", 0))
                orig = float(o.get("origQty", 0))
                create_t = int(o.get("time"))
            except Exception:
                continue

            age_min = (now - create_t) / 60000.0
            try:
                ticker = self.exchange.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker.get("price"))
            except Exception:
                current_price = entry
            dist_bps = abs(current_price - entry) / entry * 10000 if entry else 0

            gap_atr_ok = False
            if PENDING_GAP_ATR_MULT > 0:
                try:
                    kl = self.exchange.futures_klines(
                        symbol=symbol, interval=SR_TIMEFRAME, limit=15
                    )
                    trs = []
                    for i in range(1, len(kl)):
                        high = float(kl[i][2])
                        low = float(kl[i][3])
                        prev_close = float(kl[i - 1][4])
                        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                        trs.append(tr)
                    atr = sum(trs[-14:]) / min(14, len(trs)) if trs else None
                    if atr is not None:
                        gap_atr_ok = abs(current_price - entry) >= atr * PENDING_GAP_ATR_MULT
                except Exception:
                    gap_atr_ok = False

            cid_data = get_order_metadata(o)
            s3 = cid_data.get("sr3S")
            r3 = cid_data.get("sr3R")
            if (s3 is None or r3 is None) and PENDING_USE_SR3:
                sr = get_sr_levels(self.symbol, SR_TIMEFRAME)
                if s3 is None:
                    s_vals = sr.get("S", [])
                    s3 = s_vals[2] if len(s_vals) >= 3 else None
                if r3 is None:
                    r_vals = sr.get("R", [])
                    r3 = r_vals[2] if len(r_vals) >= 3 else None
            try:
                s3 = float(s3) if s3 is not None else None
            except Exception:
                s3 = None
            try:
                r3 = float(r3) if r3 is not None else None
            except Exception:
                r3 = None

            too_old = age_min >= PENDING_TTL_MIN
            too_far = dist_bps >= PENDING_MAX_GAP_BPS or gap_atr_ok
            sr_invalid = False
            if PENDING_USE_SR3 and (s3 or r3):
                if side == "SELL" and s3:
                    sr_invalid = current_price <= s3 * (1 - PENDING_SR_BUFFER_BPS / 10000.0)
                if side == "BUY" and r3:
                    sr_invalid = current_price >= r3 * (1 + PENDING_SR_BUFFER_BPS / 10000.0)

            reasons = []
            if too_old:
                reasons.append("ttl")
            if too_far:
                reasons.append("gap")
            if sr_invalid:
                reasons.append("sr3")
            if not reasons:
                continue

            cfm = int(cid_data.get("cfm", 0))
            base_id = cid_data.get("base_id", o.get("clientOrderId") or f"oid{o.get('orderId')}")
            if cfm + 1 < PENDING_CANCEL_CONFIRM_BARS:
                new_id = f"{base_id}-c{cfm + 1}"
                new_meta = dict(cid_data)
                new_meta["cfm"] = cfm + 1
                new_meta["base_id"] = base_id
                if not DRY_RUN:
                    try:
                        if hasattr(self.exchange, "futures_cancel_replace_order"):
                            kwargs = {
                                "symbol": symbol,
                                "side": side,
                                "type": "LIMIT",
                                "timeInForce": "GTC",
                                "price": entry,
                                "quantity": orig,
                                "cancelReplaceMode": "MODIFY",
                                "newClientOrderId": new_id,
                            }
                            if o.get("clientOrderId"):
                                kwargs["cancelOrigClientOrderId"] = o.get("clientOrderId")
                            else:
                                kwargs["cancelOrigOrderId"] = o.get("orderId")
                            self.exchange.futures_cancel_replace_order(**kwargs)
                            store_order_metadata(new_id, None, new_meta)
                    except Exception:
                        pass
                continue

            motivo = ",".join(reasons)
            if DRY_RUN:
                log(
                    f"🚫 ORDEN PENDIENTE CANCELADA | {self.symbol} | side={side} | age={age_min:.1f}m | dist={dist_bps:.1f}bps | R3={r3} | S3={s3} | motivo={motivo}"
                )
                continue

            try:
                if o.get("clientOrderId"):
                    self.exchange.futures_cancel_order(
                        symbol=symbol, origClientOrderId=o.get("clientOrderId")
                    )
                else:
                    self.exchange.futures_cancel_order(
                        symbol=symbol, orderId=o.get("orderId")
                    )
            except Exception as e:
                log(
                    f"❌ Error cancelando orden {o.get('orderId') or o.get('clientOrderId')}: {e}"
                )
            else:
                log(
                    f"🚫 ORDEN PENDIENTE CANCELADA | {self.symbol} | side={side} | age={age_min:.1f}m | dist={dist_bps:.1f}bps | R3={r3} | S3={s3} | motivo={motivo}"
                )
                if executed > 0:
                    try:
                        self.verificar_y_configurar_tp_sl()
                    except Exception as e:
                        log(f"❌ Error ajustando TP/SL: {e}")

    
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
                    log(f"❌❌❌❌❌ Error actualizando summary: {e}")

            self.cancelar_ordenes_tp_sl()
            self.min_profit_sl_moved = False
        except Exception as e:
            log(f"❌❌❌❌❌ Error al cerrar posición: {e}")

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
                            log(f"❌❌❌❌❌ Error ajustando SL: {e}")
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
                            log(f"❌❌❌❌❌ Error ajustando SL: {e}")
                elif price >= sl and not self.min_profit_sl_moved:
                    pnl = (entry - price) * amount
                    log(f"Stop Loss alcanzado. Cerrando posición en {price} con PNL {pnl}")
                    self.cerrar_posicion()
                else:
                    log(f"Futuros: Precio actual: {price} — TP: {tp}, SL: {sl} -->")

        except Exception as e:
            log(f"❌❌❌❌❌ Error al evaluar posición: {e}")

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
            log(f"❌❌❌❌❌ Error actualizando summary: {e}")


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

    sup_levels = None
    levels = None

    try:
        symbol_ref = (
            self.symbol
            if "self" in locals() and hasattr(self, "symbol")
            else (symbol if "symbol" in locals() else os.getenv("SYMBOL", "DOGE/USDT"))
        )
        interval = os.getenv("SUP_INTERVAL", "5m")

        sup_levels = next_supports(
            symbol_ref, interval=interval, limit=500, log_fn=log, exchange=exchange
        )
        if sup_levels:
            top_sup = sup_levels[0]
            razones = ", ".join(top_sup.get("reasons", [])[:3])
            log(
                f"🛡️ Próximo soporte: {top_sup['level']:.6f} "
                f"(score {top_sup['score']:.2f}, dist≈{top_sup['distance_pct']:.2f}%)"
                + (f" | razones: {razones}" if razones else "")
            )
        else:
            log("🛡️ Próximo soporte: no encontrado (datos insuficientes)")
    except Exception as e:
        log(f"⚠️ Error calculando soportes: {e}")

    try:
        symbol_ref = (
            self.symbol
            if "self" in locals() and hasattr(self, "symbol")
            else (symbol if "symbol" in locals() else os.getenv("SYMBOL", "DOGE/USDT"))
        )
        interval = os.getenv("RES_INTERVAL", "5m")
        levels = next_resistances(
            symbol_ref, interval=interval, limit=500, exchange=exchange
        )

        if levels:
            top = levels[0]
            razones = ", ".join(top.get("reasons", [])[:3])
            log(
                f"🧱 Próxima resistencia: {top['level']:.6f} "
                f"(score {top['score']:.2f}, dist≈{top['distance_pct']:.2f}%)"
                f"{f' | razones: {razones}' if razones else ''}"
            )
            resumen_top3 = ", ".join(
                [
                    f"{x['level']:.6f} (s{x['score']:.2f}, d{x['distance_pct']:.2f}%)"
                    for x in levels[:3]
                ]
            )
            log(f"🧱 Top3 resistencias: {resumen_top3}")
        else:
            log("🧱 Próxima resistencia: no encontrada (datos insuficientes)")
    except Exception as e:
        log(f"❌❌❌❌❌ Error calculando resistencias: {e}")
    
    position_info = bot.obtener_posicion_abierta()
    position_for_sl = position_info or {"positionAmt": 0, "entryPrice": 0}

    BE_BUFFER_TICKS = int(os.getenv("BE_BUFFER_TICKS", "3"))
    MIN_MOVE_TICKS = int(os.getenv("MIN_MOVE_TICKS", "2"))
    mode = os.getenv("TRAILING_SL_MODE", "observe")

    result = compute_trailing_sl_candidate(
        symbol_ref,
        position_for_sl,
        price,
        sup_levels,
        levels,
        bot.tick_size,
        bot.price_precision,
        MIN_MOVE_TICKS,
        BE_BUFFER_TICKS,
    )

    log(
        f"🔎 OBS: {('LONG' if float(position_for_sl.get('positionAmt', 0))>0 else 'SHORT')} entry={result['entry']:.6f} px={price:.6f} "
        f"SL_act={result['sl_act'] if result['sl_act'] is not None else '—'} "
        f"| {'S*' if result['side']=='LONG' else 'R*'}={result['anchor']:.6f} buf={BE_BUFFER_TICKS}t → SL_cand={result['sl_cand']:.6f} "
        f"| {'ok' if result['ok'] else 'skip:'+result['reason']}"
    )

    if mode == "execute" and result["ok"]:
        qty_abs = abs(float(position_info["positionAmt"]))
        apply_trailing_sl(
            exchange,
            symbol_ref,
            result["side"],
            qty_abs,
            result["sl_cand"],
            bot.tick_size,
            bot.price_precision,
            bot.quantity_precision,
        )

    if position_info:
        bot.verificar_y_configurar_tp_sl()
        bot.evaluar_posicion()
    elif bot.tiene_orden_abierta():
        bot.verificar_orden_limit()
        order = bot.obtener_orden_abierta()
        if order:
            side = order.get("side")
            o_type = order.get("type")
            try:
                orig_qty = float(order.get("origQty") or 0)
            except Exception:
                orig_qty = 0.0
            price_order = (
                order.get("price")
                or order.get("stopPrice")
                or price
            )
            try:
                price_order = float(price_order)
            except Exception:
                price_order = price

            precio_apertura_val = bot._fmt_price(price_order)
            if precio_apertura_val is None:
                precio_apertura_val = price_order
            precio_apertura = f"{precio_apertura_val:.{bot.price_precision if bot.price_precision else 6}f}"

            qty_val = bot._fmt_qty(orig_qty)
            cantidad = f"{qty_val:.{bot.quantity_precision if bot.quantity_precision else 4}f}"

            monto_usdt = f"{precio_apertura_val * qty_val:.2f}"
            order_id = order.get("orderId")
            log(
                f"📌 Orden pendiente: {side} {cantidad} {symbol} @ {precio_apertura} | Monto nominal ≈ {monto_usdt} USDT | tipo={o_type}{f' | id={order_id}' if order_id else ''}. Esperando ejecución o cancelación."
            )
        else:
            log("Orden pendiente detectada, esperando ejecución o cancelación.")
    else:
        side, level, patterns, rango = generate_signal(exchange, symbol)
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
                print(f"TESTNET activo - Sin señal de estrategia - Apalancamiento: {lev}x")
            else:
                print("Sin señal de estrategia")

    return price


def run_iteration(exchange, cfg):
    symbol = cfg.get("symbol", "BTC/USDT")
    leverage = cfg.get("leverage")
    use_breakout_dynamic_stops = cfg.get("use_breakout_dynamic_stops", False)
    testnet = cfg.get("testnet", False)
    bot = FuturesBot(
        exchange,
        symbol,
        leverage=leverage,
        use_breakout_dynamic_stops=use_breakout_dynamic_stops,
    )
    return _run_iteration(exchange, bot, testnet, symbol, leverage)


