from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import math
import os
from typing import Any, Iterable, Sequence

from common.utils import sanitize_client_order_id
from config.utils import parse_bool
from core.ports.broker import BrokerPort
from core.ports.market_data import MarketDataPort
from core.ports.settings import SettingsProvider, get_symbol
from utils.tp_store_s3 import load_tp_value, persist_tp_value

logger = logging.getLogger("bot.strategy.wedge")

STRATEGY_NAME = "WedgeFormation"


@dataclass(slots=True)
class TrendLine:
    slope: float
    intercept: float
    touches: int

    def value_at(self, index: float) -> float:
        return self.slope * index + self.intercept


@dataclass(slots=True)
class WedgePattern:
    upper: TrendLine
    lower: TrendLine
    kind: str
    convergence_ratio: float
    bars: int


def _timeframe_to_minutes(value: str) -> int:
    value = value.strip()
    if not value:
        return 1
    unit = value[-1].lower()
    qty = int(value[:-1]) if value[:-1] else 1
    factor = {"m": 1, "h": 60, "d": 1440}.get(unit, 1)
    return max(1, qty * factor)


def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


class WedgeFormationStrategy:
    """Detect wedge formations and place entry/TP orders accordingly."""

    def __init__(
        self,
        market_data: MarketDataPort,
        broker: BrokerPort,
        settings: SettingsProvider,
    ) -> None:
        self._market_data = market_data
        self._broker = broker
        self._settings = settings

    # ------------------------------------------------------------------
    def run(
        self,
        exchange: BrokerPort | None = None,
        market_data: MarketDataPort | None = None,
        settings: SettingsProvider | None = None,
        now_utc: datetime | None = None,
        event: Any | None = None,
    ) -> dict[str, Any]:
        exch = exchange or self._broker
        md = market_data or self._market_data
        settings = settings or self._settings
        now = now_utc or datetime.utcnow()

        symbol = get_symbol(settings)
        timeframe = os.getenv("WEDGE_TIMEFRAME", "15m")
        log_prefix = f"WEDGE{symbol}{timeframe}"
        cid_prefix = f"WEDGE_{symbol}_{timeframe}"

        filters_enabled = parse_bool(os.getenv("WEDGE_FILTERS_ENABLED"), default=False)
        min_touches = _safe_int(os.getenv("WEDGE_MIN_TOUCHES_PER_SIDE"), 2)
        tol_atr_mult = _safe_float(os.getenv("WEDGE_TOUCH_TOL_ATR"), 0.25)
        min_convergence = _safe_float(os.getenv("WEDGE_MIN_CONVERGENCE"), 0.2)
        min_bars = _safe_int(os.getenv("WEDGE_MIN_BARS"), 20)
        max_bars = _safe_int(os.getenv("WEDGE_MAX_BARS"), 120)
        rr_min = _safe_float(os.getenv("RR_MIN"), 1.0)
        order_ttl_bars = _safe_int(os.getenv("ORDER_TTL_BARS"), 5)
        buffer_atr_mult = _safe_float(os.getenv("WEDGE_BUFFER_ATR"), 0.15)

        logger.info(
            "%s settings %s",
            log_prefix,
            {
                "filters_enabled": filters_enabled,
                "min_touches": min_touches,
                "tol_atr_mult": tol_atr_mult,
                "min_convergence": min_convergence,
                "min_bars": min_bars,
                "max_bars": max_bars,
                "order_ttl_bars": order_ttl_bars,
            },
        )

        # ------------------------------------------------------------------
        # Guard: existing position
        position = None
        try:
            position = exch.get_position(symbol)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("%s position_check.error %s", log_prefix, exc)
        position_amt = 0.0
        if position is not None:
            try:
                position_amt = float(position.get("positionAmt", 0.0))
            except (TypeError, ValueError):
                position_amt = 0.0
        if not math.isclose(position_amt, 0.0, abs_tol=1e-12):
            logger.info("%s guard position_open amount=%s", log_prefix, position_amt)
            self._ensure_tp_for_position(
                exch=exch,
                symbol=symbol,
                cid_prefix=cid_prefix,
                now=now,
                qty=abs(position_amt),
                is_long=position_amt > 0,
                log_prefix=log_prefix,
            )
            return {
                "status": "skipped_existing_position",
                "symbol": symbol,
                "positionAmt": position_amt,
            }

        # ------------------------------------------------------------------
        # Guard: existing pending order
        open_orders: list[Any] = []
        try:
            open_orders = exch.open_orders(symbol)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("%s open_orders.error %s", log_prefix, exc)
        wedge_orders = [
            o
            for o in open_orders
            if str(o.get("clientOrderId", "")).startswith(cid_prefix)
        ]
        if wedge_orders:
            logger.info("%s guard pending_orders=%d", log_prefix, len(wedge_orders))
            cancelled_any = False
            timeframe_min = _timeframe_to_minutes(timeframe)
            ttl_ms = timeframe_min * 60_000 * order_ttl_bars
            now_ms = int(now.timestamp() * 1000)
            for order in wedge_orders:
                order_time = int(order.get("time") or order.get("updateTime") or 0)
                if ttl_ms and order_time and now_ms - order_time >= ttl_ms:
                    order_id = order.get("orderId")
                    client_id = order.get("clientOrderId")
                    logger.info(
                        "%s ttl_cancel orderId=%s clientId=%s age_ms=%s ttl_ms=%s",
                        log_prefix,
                        order_id,
                        client_id,
                        now_ms - order_time,
                        ttl_ms,
                    )
                    try:
                        exch.cancel_order(symbol, orderId=order_id, clientOrderId=client_id)
                        cancelled_any = True
                    except Exception as exc:  # pragma: no cover
                        logger.warning("%s cancel.error %s", log_prefix, exc)
            status = "cancelled_ttl" if cancelled_any else "pending_order_exists"
            return {"status": status, "symbol": symbol, "open_orders": len(wedge_orders)}

        # ------------------------------------------------------------------
        candles = md.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=max_bars + 5)
        if len(candles) < min_bars:
            logger.info("%s skip not_enough_candles=%d", log_prefix, len(candles))
            return {"status": "not_enough_candles", "symbol": symbol, "candles": len(candles)}

        atr = self._compute_atr(candles)
        pivots_high, pivots_low = self._extract_pivots(candles)
        logger.info(
            "%s pivots high=%d low=%d atr=%.6f",
            log_prefix,
            len(pivots_high),
            len(pivots_low),
            atr,
        )

        if len(pivots_high) < min_touches or len(pivots_low) < min_touches:
            logger.info("%s skip not_enough_touches", log_prefix)
            return {"status": "not_enough_touches", "symbol": symbol}

        pattern = self._detect_wedge(
            pivots_high=pivots_high,
            pivots_low=pivots_low,
            min_bars=min_bars,
            max_bars=max_bars,
            min_convergence=min_convergence,
        )
        if pattern is None:
            logger.info("%s skip no_wedge_detected", log_prefix)
            return {"status": "no_wedge", "symbol": symbol}

        logger.info(
            "%s wedge kind=%s conv=%.3f bars=%d slopes=(%.6f, %.6f)",
            log_prefix,
            pattern.kind,
            pattern.convergence_ratio,
            pattern.bars,
            pattern.upper.slope,
            pattern.lower.slope,
        )

        if filters_enabled:
            tol_value = tol_atr_mult * atr
            if not self._validate_touch_tolerance(pivots_high, pattern.upper, tol_value):
                logger.info("%s filter touch_upper_failed tol=%.6f", log_prefix, tol_value)
                return {"status": "filter_touch_upper", "symbol": symbol}
            if not self._validate_touch_tolerance(pivots_low, pattern.lower, tol_value):
                logger.info("%s filter touch_lower_failed tol=%.6f", log_prefix, tol_value)
                return {"status": "filter_touch_lower", "symbol": symbol}
            if pattern.convergence_ratio < min_convergence:
                logger.info(
                    "%s filter convergence_failed ratio=%.6f min=%.6f",
                    log_prefix,
                    pattern.convergence_ratio,
                    min_convergence,
                )
                return {"status": "filter_convergence", "symbol": symbol}
            if pattern.bars < min_bars or pattern.bars > max_bars:
                logger.info(
                    "%s filter bars_failed bars=%d range=[%d,%d]",
                    log_prefix,
                    pattern.bars,
                    min_bars,
                    max_bars,
                )
                return {"status": "filter_bars", "symbol": symbol, "bars": pattern.bars}

        current_index = len(candles) - 1
        buffer = buffer_atr_mult * atr
        upper_now = pattern.upper.value_at(current_index)
        lower_now = pattern.lower.value_at(current_index)

        if pattern.kind == "rising":
            side = "SELL"
            entry_line = upper_now
            tp_line = lower_now
            entry_price = entry_line + buffer
            tp_price = tp_line
            sl_theoretical = entry_price + buffer if buffer > 0 else entry_price + atr * 0.5
        else:
            side = "BUY"
            entry_line = lower_now
            tp_line = upper_now
            entry_price = entry_line - buffer
            tp_price = tp_line
            sl_theoretical = entry_price - buffer if buffer > 0 else entry_price - atr * 0.5

        if side == "SELL" and tp_price >= entry_price:
            logger.info(
                "%s geometry skip invalid_prices entry=%.6f tp=%.6f side=%s",
                log_prefix,
                entry_price,
                tp_price,
                side,
            )
            return {"status": "invalid_prices", "symbol": symbol}
        if side == "BUY" and tp_price <= entry_price:
            logger.info(
                "%s geometry skip invalid_prices entry=%.6f tp=%.6f side=%s",
                log_prefix,
                entry_price,
                tp_price,
                side,
            )
            return {"status": "invalid_prices", "symbol": symbol}

        rr = self._estimate_rr(entry_price, tp_price, sl_theoretical)
        if filters_enabled and rr < rr_min:
            logger.info("%s filter rr_failed rr=%.3f min=%.3f", log_prefix, rr, rr_min)
            return {"status": "filter_rr", "symbol": symbol, "rr": rr}

        # ------------------------------------------------------------------
        # Re-check guards before order placement
        try:
            post_position = exch.get_position(symbol)
        except Exception:
            post_position = None
        if post_position is not None:
            try:
                post_amt = float(post_position.get("positionAmt", 0.0))
            except (TypeError, ValueError):
                post_amt = 0.0
            if not math.isclose(post_amt, 0.0, abs_tol=1e-12):
                logger.info("%s guard.post position_detected amt=%s", log_prefix, post_amt)
                return {"status": "race_position", "symbol": symbol}
        try:
            open_orders = exch.open_orders(symbol)
        except Exception:
            open_orders = []
        for order in open_orders:
            if str(order.get("clientOrderId", "")).startswith(cid_prefix):
                logger.info("%s guard.post order_exists", log_prefix)
                return {"status": "race_order", "symbol": symbol}

        entry_price_norm = exch.round_price_to_tick(symbol, entry_price)
        tp_price_norm = exch.round_price_to_tick(symbol, tp_price)

        filters = exch.get_symbol_filters(symbol)
        lot = filters.get("LOT_SIZE", {}) if isinstance(filters, dict) else {}
        step_size = float(lot.get("stepSize", 0.0) or 0.0)
        min_qty = float(lot.get("minQty", 0.0) or 0.0)
        price_filter = filters.get("PRICE_FILTER", {}) if isinstance(filters, dict) else {}
        tick_size = float(price_filter.get("tickSize", 0.0) or 0.0)
        min_notional = float(
            (filters.get("MIN_NOTIONAL", {}) or {}).get("notional")
            or (filters.get("MIN_NOTIONAL", {}) or {}).get("minNotional", 0.0)
        )

        risk_notional = float(getattr(settings, "RISK_NOTIONAL_USDT", 0.0) or 0.0)
        qty_target_src = "NONE"
        if risk_notional > 0 and entry_price_norm > 0:
            qty_target = risk_notional / entry_price_norm
            qty_target_src = "NOTIONAL"
        else:
            balance = 0.0
            try:
                balance = float(exch.get_available_balance_usdt())
            except Exception:
                balance = 0.0
            risk_pct = float(getattr(settings, "RISK_PCT", 0.0) or 0.0)
            if risk_pct > 0 and balance > 0 and entry_price_norm > 0:
                qty_target = (balance * risk_pct) / entry_price_norm
                qty_target_src = "PCT"
            else:
                qty_target = 0.0

        def _ceil_to_step(x: float, step: float) -> float:
            return math.ceil(x / step) * step if step else x

        qty_min_by_notional = _ceil_to_step(
            min_notional / entry_price_norm if entry_price_norm else 0.0,
            step_size,
        )
        qty = max(qty_target, min_qty, qty_min_by_notional)
        qty_norm = exch.round_qty_to_step(symbol, qty)

        if entry_price_norm * qty_norm < min_notional and entry_price_norm > 0:
            qty_norm = exch.round_qty_to_step(
                symbol,
                _ceil_to_step(min_notional / entry_price_norm, step_size),
            )

        logger.info(
            "%s sizing entry_price=%.6f qty_target=%.6f qty_norm=%.6f src=%s",
            log_prefix,
            entry_price_norm,
            qty_target,
            qty_norm,
            qty_target_src,
        )

        if qty_norm <= 0:
            logger.info("%s skip qty_not_positive", log_prefix)
            return {"status": "qty_invalid", "symbol": symbol}

        now_minute = int(now.timestamp() // 60)
        entry_client_id = sanitize_client_order_id(
            f"{cid_prefix}_{now_minute}_ENTRY"
        )
        tp_client_id = sanitize_client_order_id(f"{cid_prefix}_{now_minute}_TP")

        try:
            order = exch.place_entry_limit(
                symbol,
                side,
                entry_price_norm,
                qty_norm,
                entry_client_id,
                timeInForce="GTC",
            )
        except TypeError:  # pragma: no cover - legacy interface fallback
            order = exch.place_entry_limit(
                symbol,
                side,
                entry_price_norm,
                qty_norm,
                entry_client_id,
            )

        persisted = persist_tp_value(symbol, tp_price_norm, now.timestamp())
        # TODO: Cuando implementemos SL: aplicar SL únicamente al cierre de vela fuera del patrón
        # en el timeframe WEDGE_TIMEFRAME; hasta entonces, sin SL automático.
        logger.info(
            "%s order entry placed side=%s price=%.6f qty=%.6f clientId=%s tp=%.6f persisted=%s",
            log_prefix,
            side,
            entry_price_norm,
            qty_norm,
            entry_client_id,
            tp_price_norm,
            persisted,
        )

        return {
            "status": "entry_order_placed",
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price_norm,
            "qty": qty_norm,
            "clientOrderId": entry_client_id,
            "tp_price": tp_price_norm,
            "order": order,
        }

    # ------------------------------------------------------------------
    def _ensure_tp_for_position(
        self,
        exch: BrokerPort,
        symbol: str,
        cid_prefix: str,
        now: datetime,
        qty: float,
        is_long: bool,
        log_prefix: str,
    ) -> None:
        if qty <= 0:
            return
        try:
            open_orders = exch.open_orders(symbol)
        except Exception:
            open_orders = []
        for order in open_orders:
            client_id = str(order.get("clientOrderId", ""))
            if client_id.startswith(cid_prefix) and "TP" in client_id:
                logger.info("%s tp_exists clientId=%s", log_prefix, client_id)
                return

        tp_value = load_tp_value(symbol)
        if tp_value is None:
            logger.info("%s tp_missing_storage", log_prefix)
            return

        tp_price = exch.round_price_to_tick(symbol, tp_value)
        qty_norm = exch.round_qty_to_step(symbol, qty)
        if qty_norm <= 0:
            logger.info("%s tp_skip qty_norm<=0", log_prefix)
            return

        side = "SELL" if is_long else "BUY"
        client_id = sanitize_client_order_id(
            f"{cid_prefix}_{int(now.timestamp() // 60)}_TP"
        )
        try:
            exch.place_tp_reduce_only(symbol, side, tp_price, qty_norm, client_id)
            logger.info(
                "%s tp_order side=%s price=%.6f qty=%.6f clientId=%s",
                log_prefix,
                side,
                tp_price,
                qty_norm,
                client_id,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("%s tp_place.error %s", log_prefix, exc)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_atr(candles: Sequence[Sequence[float]], period: int = 14) -> float:
        if len(candles) < 2:
            return 0.0
        trs: list[float] = []
        prev_close = float(candles[0][4])
        for candle in candles[1:]:
            high = float(candle[2])
            low = float(candle[3])
            close = float(candle[4])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
            prev_close = close
        if not trs:
            return 0.0
        window = trs[-period:] if len(trs) >= period else trs
        return sum(window) / len(window)

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_pivots(
        candles: Sequence[Sequence[float]],
        left: int = 2,
        right: int = 2,
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        pivots_high: list[tuple[int, float]] = []
        pivots_low: list[tuple[int, float]] = []
        for idx in range(left, len(candles) - right):
            high = highs[idx]
            low = lows[idx]
            if high >= max(highs[idx - left : idx]) and high >= max(highs[idx + 1 : idx + 1 + right]):
                pivots_high.append((idx, high))
            if low <= min(lows[idx - left : idx]) and low <= min(lows[idx + 1 : idx + 1 + right]):
                pivots_low.append((idx, low))
        return pivots_high, pivots_low

    # ------------------------------------------------------------------
    def _detect_wedge(
        self,
        pivots_high: list[tuple[int, float]],
        pivots_low: list[tuple[int, float]],
        min_bars: int,
        max_bars: int,
        min_convergence: float,
    ) -> WedgePattern | None:
        upper = self._fit_line(pivots_high)
        lower = self._fit_line(pivots_low)
        if upper is None or lower is None:
            return None

        first_idx = min(pivots_high[0][0], pivots_low[0][0])
        last_idx = max(pivots_high[-1][0], pivots_low[-1][0])
        bars = last_idx - first_idx + 1
        if bars < min_bars or bars > max_bars:
            logger.debug("wedge bars_out_of_range=%s", bars)
        width_start = abs(upper.value_at(first_idx) - lower.value_at(first_idx))
        width_end = abs(upper.value_at(last_idx) - lower.value_at(last_idx))
        convergence_ratio = (width_start - width_end) / width_start if width_start else 0.0

        kind = None
        if upper.slope > 0 and lower.slope > 0 and lower.slope > upper.slope:
            kind = "rising"
        elif upper.slope < 0 and lower.slope < 0 and upper.slope < lower.slope:
            kind = "falling"

        if kind is None:
            return None
        if width_end >= width_start:
            return None
        if convergence_ratio < min_convergence:
            logger.debug(
                "wedge convergence_low ratio=%.6f min=%.6f", convergence_ratio, min_convergence
            )

        return WedgePattern(
            upper=upper,
            lower=lower,
            kind=kind,
            convergence_ratio=convergence_ratio,
            bars=bars,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _fit_line(points: list[tuple[int, float]]) -> TrendLine | None:
        if len(points) < 2:
            return None
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        n = float(len(points))
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xx = sum(x * x for x in xs)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sum_xx - sum_x * sum_x
        if math.isclose(denom, 0.0, abs_tol=1e-9):
            return None
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        return TrendLine(slope=slope, intercept=intercept, touches=len(points))

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_touch_tolerance(
        pivots: Iterable[tuple[int, float]],
        line: TrendLine,
        tolerance: float,
    ) -> bool:
        if tolerance <= 0:
            return True
        for idx, price in pivots:
            projected = line.value_at(idx)
            if abs(price - projected) > tolerance:
                return False
        return True

    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_rr(entry: float, tp: float, sl: float) -> float:
        numer = abs(entry - tp)
        denom = abs(entry - sl)
        if denom <= 0:
            return float("inf")
        return numer / denom


__all__ = ["WedgeFormationStrategy", "STRATEGY_NAME"]
