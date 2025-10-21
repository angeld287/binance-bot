from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import math
import os
from decimal import Decimal, DivisionByZero, InvalidOperation, ROUND_DOWN, ROUND_UP
from typing import Any, Iterable, Sequence

from common.rounding_diag import emit_rounding_diag, format_rounding_diag_number
from common.utils import sanitize_client_order_id
from config.utils import parse_bool
from core.ports.broker import BrokerPort
from core.ports.market_data import MarketDataPort
from core.ports.settings import SettingsProvider, get_symbol
from common.precision import format_decimal, round_to_step, to_decimal
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


@dataclass(slots=True)
class SymbolFilters:
    tick_size: Decimal
    step_size: Decimal
    min_notional: Decimal
    min_qty: Decimal


@dataclass(slots=True)
class QtyGuardResult:
    success: bool
    qty: Decimal | None
    adjusted: bool
    reason: str | None
    notional: Decimal


@dataclass(slots=True)
class PrecisionComputation:
    price_requested: Decimal | None
    price_adjusted: Decimal | None
    qty_requested: Decimal | None
    qty_adjusted: Decimal | None
    stop_requested: Decimal | None = None
    stop_adjusted: Decimal | None = None


class OrderPrecisionError(ValueError):
    """Raised when order precision validation fails before dispatch."""

    def __init__(self, tag: str, reason: str) -> None:
        super().__init__(reason)
        self.tag = tag
        self.reason = reason


def to_api_str(value: Decimal | Any) -> str:
    dec = value if isinstance(value, Decimal) else to_decimal(value)
    normalized = dec.normalize()
    if normalized == normalized.to_integral():
        normalized = normalized.quantize(Decimal("1"))
    else:
        exponent = normalized.as_tuple().exponent
        if exponent < 0:
            quantum = Decimal(1).scaleb(exponent)
            normalized = normalized.quantize(quantum)
    return format(normalized, "f")


def to_decimal_or_none(value: Any) -> Decimal | None:
    try:
        if value is None:
            return None
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def is_multiple(value_dec: Decimal | None, step_dec: Decimal | None) -> bool | None:
    try:
        if value_dec is None or step_dec is None or step_dec == 0:
            return None
        return (value_dec % step_dec) == 0
    except Exception:
        return None


def assert_is_multiple(dec_value: Decimal | Any, tick_or_step: Decimal | Any) -> bool:
    step = tick_or_step if isinstance(tick_or_step, Decimal) else to_decimal(tick_or_step)
    if step <= 0:
        return True
    value = dec_value if isinstance(dec_value, Decimal) else to_decimal(dec_value)
    if value == 0:
        return True
    try:
        ratio = (value / step).normalize()
    except (InvalidOperation, DivisionByZero):
        return False
    return ratio == ratio.to_integral_value()


def quantize_price_to_tick(
    price: Decimal | Any, tick_size: Decimal | Any, *, side: str | None
) -> Decimal:
    price_dec = price if isinstance(price, Decimal) else to_decimal(price)
    try:
        tick_quant = Decimal(str(tick_size))
    except (InvalidOperation, ValueError, TypeError):
        return price_dec
    if tick_quant <= 0:
        return price_dec
    if price_dec == 0:
        return Decimal("0")
    side_norm = (side or "").upper()
    rounding = ROUND_UP if side_norm == "SELL" else ROUND_DOWN
    price_for_quant = Decimal(str(price_dec))
    return price_for_quant.quantize(tick_quant, rounding=rounding)


def ceil_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    if value <= 0:
        return Decimal("0")
    quotient = (value / step).to_integral_value(rounding=ROUND_UP)
    return quotient * step


def log_precision_summary(
    *,
    log_prefix: str,
    symbol: str,
    market: str,
    timeframe: str,
    side: str,
    order_type: str,
    tick_size: Decimal,
    step_size: Decimal,
    min_notional: Decimal,
    requested_price: Decimal | None,
    adjusted_price: Decimal | None,
    requested_qty: Decimal | None,
    adjusted_qty: Decimal | None,
    requested_stop: Decimal | None = None,
    adjusted_stop: Decimal | None = None,
) -> None:
    price_multiple = (
        assert_is_multiple(adjusted_price, tick_size)
        if adjusted_price is not None
        else True
    )
    qty_multiple = (
        assert_is_multiple(adjusted_qty, step_size)
        if adjusted_qty is not None
        else True
    )
    stop_multiple = (
        assert_is_multiple(adjusted_stop, tick_size)
        if adjusted_stop is not None
        else True
    )

    logger.info(
        "%s precision_summary %s",
        log_prefix,
        {
            "symbol": symbol,
            "market": market,
            "timeframe": timeframe,
            "side": side,
            "order_type": order_type,
            "tickSize": repr(tick_size),
            "stepSize": repr(step_size),
            "minNotional": repr(min_notional),
            "requested_price": repr(requested_price),
            "adjusted_price": repr(adjusted_price),
            "requested_qty": repr(requested_qty),
            "adjusted_qty": repr(adjusted_qty),
            "requested_stopPrice": repr(requested_stop),
            "adjusted_stopPrice": repr(adjusted_stop),
            "is_multiple_price": price_multiple,
            "is_multiple_qty": qty_multiple,
            "is_multiple_stop": stop_multiple,
        },
    )


def log_order_not_sent(
    *,
    log_prefix: str,
    symbol: str,
    market: str,
    timeframe: str,
    side: str,
    order_type: str,
    tag: str,
    reason: str,
    filters: SymbolFilters | None = None,
) -> None:
    payload: dict[str, Any] = {
        "tag": "ORDER_NOT_SENT",
        "sent": False,
        "symbol": symbol,
        "market": market,
        "timeframe": timeframe,
        "side": side,
        "orderType": order_type,
        "reason": reason,
        "detail_tag": tag,
    }
    if filters is not None:
        payload["tickSize"] = format_decimal(filters.tick_size)
        payload["stepSize"] = format_decimal(filters.step_size)
    logger.warning("%s %s", log_prefix, payload)


def compute_order_precision(
    *,
    price_requested: Decimal | None,
    qty_requested: Decimal | None,
    stop_requested: Decimal | None,
    side: str,
    order_type: str,
    filters: SymbolFilters,
    exchange: BrokerPort,
    symbol: str,
) -> PrecisionComputation:
    tick_size = filters.tick_size
    step_size = filters.step_size

    if tick_size is None or tick_size <= 0 or step_size is None or step_size <= 0:
        raise OrderPrecisionError("ORDER_REJECT_TICK_INVALID", "invalid_tick_or_step")

    price_adjusted: Decimal | None = None
    if price_requested is not None:
        price_adjusted = to_decimal(
            exchange.round_price_to_tick(symbol, price_requested)
        )
        if not assert_is_multiple(price_adjusted, tick_size):
            raise OrderPrecisionError("ORDER_REJECT_TICK_INVALID", "price_not_multiple")

    stop_adjusted: Decimal | None = None
    if stop_requested is not None:
        stop_adjusted = to_decimal(
            exchange.round_price_to_tick(symbol, stop_requested)
        )
        if not assert_is_multiple(stop_adjusted, tick_size):
            raise OrderPrecisionError("ORDER_REJECT_TICK_INVALID", "stop_not_multiple")

    qty_adjusted: Decimal | None = None
    if qty_requested is not None:
        qty_adjusted = qty_requested
        if qty_adjusted > 0 and not assert_is_multiple(qty_adjusted, step_size):
            qty_adjusted = round_to_step(qty_adjusted, step_size, rounding=ROUND_DOWN)
        if qty_adjusted <= 0:
            raise OrderPrecisionError("ORDER_REJECT_TICK_INVALID", "qty_non_positive")
        if not assert_is_multiple(qty_adjusted, step_size):
            raise OrderPrecisionError("ORDER_REJECT_TICK_INVALID", "qty_not_multiple")

    return PrecisionComputation(
        price_requested=price_requested,
        price_adjusted=price_adjusted,
        qty_requested=qty_requested,
        qty_adjusted=qty_adjusted,
        stop_requested=stop_requested,
        stop_adjusted=stop_adjusted,
    )


def get_symbol_filters(
    exchange: BrokerPort,
    symbol: str,
    market: str = "futures",
) -> SymbolFilters:
    """Return precision filters for ``symbol`` using the broker cache.

    Parameters
    ----------
    exchange:
        Broker adapter exposing ``get_symbol_filters`` backed by
        :class:`common.precision.FiltersCache`.
    symbol:
        Symbol to resolve (e.g. ``"ADAUSDT"``).
    market:
        Market namespace (``"futures"`` or ``"spot"``). The strategy currently
        supports futures only but the argument is kept to avoid diverging from
        the rest of the codebase conventions and for clearer logging.
    """

    market_norm = (market or "futures").strip().lower() or "futures"
    if market_norm != "futures":
        logger.warning(
            "precision.filters_unsupported_market symbol=%s market=%s", symbol, market
        )

    filters_raw: dict[str, Any] | None = None
    if hasattr(exchange, "get_symbol_filters"):
        try:
            filters_raw = exchange.get_symbol_filters(symbol)
        except Exception as exc:  # pragma: no cover - surface upstream error
            logger.warning(
                "precision.filters_fetch_failed symbol=%s market=%s err=%s",
                symbol,
                market_norm,
                exc,
            )
            filters_raw = None

    if not isinstance(filters_raw, dict) or not filters_raw:
        raise ValueError(f"Filters for {symbol} not available (market={market_norm})")

    price_filter = filters_raw.get("PRICE_FILTER", {})
    lot_filter = filters_raw.get("LOT_SIZE", {})
    min_notional_filter = filters_raw.get("MIN_NOTIONAL", {})

    tick_size_raw = (
        price_filter.get("tickSize")
        or price_filter.get("tick_size")
        or price_filter.get("minPrice")
        or "0"
    )
    step_size_raw = (
        lot_filter.get("stepSize")
        or lot_filter.get("step_size")
        or lot_filter.get("minQty")
        or "0"
    )
    min_notional_raw = (
        min_notional_filter.get("notional")
        or min_notional_filter.get("minNotional")
        or min_notional_filter.get("min_notional")
        or "0"
    )
    min_qty_raw = lot_filter.get("minQty") or lot_filter.get("min_qty") or "0"

    if market_norm == "futures" and os.getenv("ROUNDING_DIAG") == "1":
        diag_payload: dict[str, Any] = {
            "tag": "A_filters",
            "symbol": symbol,
            "market": "futures",
            "tickSize_raw": None,
            "stepSize_raw": None,
            "minNotional_raw": None,
            "source": "exchangeInfoFutures",
            "fetched_at": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        }
        try:
            tick_size_dec = to_decimal_or_none(tick_size_raw)
            step_size_dec = to_decimal_or_none(step_size_raw)
            min_notional_dec = to_decimal_or_none(min_notional_raw)
            diag_payload["tickSize_raw"] = (
                format_rounding_diag_number(tick_size_dec)
                if tick_size_dec is not None
                else None
            )
            diag_payload["stepSize_raw"] = (
                format_rounding_diag_number(step_size_dec)
                if step_size_dec is not None
                else None
            )
            diag_payload["minNotional_raw"] = (
                format_rounding_diag_number(min_notional_dec)
                if min_notional_dec is not None
                else None
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            diag_payload["warn"] = str(exc)
        emit_rounding_diag(diag_payload, logger=logger)

    tick_size = to_decimal(tick_size_raw)
    step_size = to_decimal(step_size_raw)
    min_notional = to_decimal(min_notional_raw)
    min_qty = to_decimal(min_qty_raw)

    if tick_size <= 0 or step_size <= 0:
        raise ValueError(
            f"Invalid precision filters for {symbol} (market={market_norm}): "
            f"tickSize={tick_size} stepSize={step_size}"
        )

    return SymbolFilters(
        tick_size=tick_size,
        step_size=step_size,
        min_notional=min_notional,
        min_qty=min_qty,
    )


def apply_qty_guards(
    symbol: str,
    side: str,
    order_type: str,
    price_dec: Decimal,
    qty_dec: Decimal,
    filters: SymbolFilters,
    *,
    allow_increase: bool = True,
    max_iterations: int = 100,
) -> QtyGuardResult:
    """Validate quantity constraints returning adjustments or failure."""

    qty = qty_dec
    adjusted = False
    step = filters.step_size
    min_qty = filters.min_qty
    notional = price_dec * qty if price_dec > 0 else Decimal("0")

    if min_qty > 0 and qty < min_qty:
        if allow_increase:
            qty = round_to_step(min_qty, step, rounding=ROUND_DOWN)
            adjusted = True
        else:
            return QtyGuardResult(
                success=False,
                qty=None,
                adjusted=False,
                reason="min_qty",
                notional=notional,
            )

    notional = price_dec * qty if price_dec > 0 else Decimal("0")

    if price_dec <= 0:
        return QtyGuardResult(
            success=False,
            qty=None,
            adjusted=adjusted,
            reason="price_non_positive",
            notional=notional,
        )

    if filters.min_notional > 0 and notional < filters.min_notional:
        if allow_increase and step > 0:
            iterations = 0
            qty_candidate = qty
            notional_candidate = notional
            while (
                notional_candidate < filters.min_notional
                and iterations < max_iterations
            ):
                qty_candidate = qty_candidate + step
                qty_candidate = round_to_step(
                    qty_candidate, step, rounding=ROUND_DOWN
                )
                notional_candidate = price_dec * qty_candidate
                iterations += 1
            if notional_candidate >= filters.min_notional and qty_candidate > 0:
                qty = qty_candidate
                notional = notional_candidate
                adjusted = True
            else:
                return QtyGuardResult(
                    success=False,
                    qty=None,
                    adjusted=adjusted,
                    reason="min_notional",
                    notional=notional_candidate,
                )
        else:
            return QtyGuardResult(
                success=False,
                qty=None,
                adjusted=adjusted,
                reason="min_notional",
                notional=notional,
            )

    return QtyGuardResult(
        success=True,
        qty=qty,
        adjusted=adjusted,
        reason=None,
        notional=price_dec * qty,
    )


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
        configured_timeframe = str(settings.get("INTERVAL", "") or "").strip()
        env_timeframe = str(os.getenv("WEDGE_TIMEFRAME", "")).strip()
        timeframe = env_timeframe or configured_timeframe or "15m"
        market_env = str(os.getenv("WEDGE_MARKET", "")).strip()
        market_cfg = str(settings.get("MARKET", "") or settings.get("MARKET_TYPE", "") or "").strip()
        market = (market_env or market_cfg or "futures").lower()
        log_prefix = f"WEDGE{symbol}{timeframe}"
        cid_prefix = f"WEDGE_{symbol}_{timeframe}"
        rounding_diag_enabled = os.getenv("ROUNDING_DIAG") == "1"

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
                market=market,
                timeframe=timeframe,
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

        entry_price = float(exch.round_price_to_tick(symbol, entry_price))
        tp_price = float(exch.round_price_to_tick(symbol, tp_price))

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

        try:
            filters = get_symbol_filters(exch, symbol, market)
        except ValueError as exc:
            logger.warning(
                "%s %s",
                log_prefix,
                {
                    "tag": "ORDER_REJECT_TICK_INVALID",
                    "symbol": symbol,
                    "market": market,
                    "timeframe": timeframe,
                    "side": side,
                    "orderType": "LIMIT",
                    "reason": str(exc),
                },
            )
            log_order_not_sent(
                log_prefix=log_prefix,
                symbol=symbol,
                market=market,
                timeframe=timeframe,
                side=side,
                order_type="LIMIT",
                tag="ORDER_REJECT_TICK_INVALID",
                reason=str(exc),
            )
            return {"status": "precision_filters_unavailable", "symbol": symbol}

        entry_price_raw_dec = to_decimal(entry_price)
        tp_price_raw_dec = to_decimal(tp_price)
        stop_price_raw_dec = None
        try:
            stop_price_raw_dec = to_decimal(sl_theoretical)
        except (InvalidOperation, DivisionByZero, ValueError, TypeError):
            stop_price_raw_dec = None

        entry_price_dec = to_decimal(
            exch.round_price_to_tick(symbol, entry_price_raw_dec)
        )
        exit_side = "SELL" if side == "BUY" else "BUY"
        tp_price_dec = to_decimal(
            exch.round_price_to_tick(symbol, tp_price_raw_dec)
        )

        entry_price_norm_dec = entry_price_dec
        tp_price_norm_dec = tp_price_dec

        step_size_dec = filters.step_size
        min_qty_dec = filters.min_qty
        min_notional_dec = filters.min_notional

        risk_notional_dec = to_decimal(
            getattr(settings, "RISK_NOTIONAL_USDT", Decimal("0")) or Decimal("0")
        )
        qty_target_src = "NONE"
        qty_target_dec = Decimal("0")
        if risk_notional_dec > 0 and entry_price_norm_dec > 0:
            qty_target_dec = risk_notional_dec / entry_price_norm_dec
            qty_target_src = "NOTIONAL"
        else:
            balance_dec = Decimal("0")
            try:
                balance_dec = to_decimal(exch.get_available_balance_usdt())
            except Exception:
                balance_dec = Decimal("0")
            risk_pct_dec = to_decimal(
                getattr(settings, "RISK_PCT", Decimal("0")) or Decimal("0")
            )
            if (
                risk_pct_dec > 0
                and balance_dec > 0
                and entry_price_norm_dec > 0
            ):
                qty_target_dec = (balance_dec * risk_pct_dec) / entry_price_norm_dec
                qty_target_src = "PCT"

        qty_min_by_notional_dec = Decimal("0")
        if entry_price_norm_dec > 0 and min_notional_dec > 0:
            qty_min_by_notional_dec = ceil_to_step(
                min_notional_dec / entry_price_norm_dec, step_size_dec
            )

        qty_candidate_dec = max(qty_target_dec, min_qty_dec, qty_min_by_notional_dec)

        qty_raw_dec = qty_candidate_dec

        if rounding_diag_enabled:
            diag_payload: dict[str, Any] = {
                "tag": "B_pre_adjust",
                "symbol": symbol,
                "side": side,
                "orderType": "LIMIT",
                "timeInForce": "GTC",
                "price_raw": None,
                "qty_raw": None,
                "stop_raw": None,
                "tickSize": None,
                "stepSize": None,
                "price_div_tick": None,
                "is_multiple_price": None,
                "is_multiple_qty": None,
                "is_multiple_stop": None,
            }
            try:
                price_raw_dec_diag = to_decimal_or_none(entry_price_raw_dec)
                qty_raw_dec_diag = to_decimal_or_none(qty_raw_dec)
                stop_raw_dec_diag = to_decimal_or_none(stop_price_raw_dec)
                tick_dec = to_decimal_or_none(filters.tick_size)
                step_dec = to_decimal_or_none(filters.step_size)

                if price_raw_dec_diag is not None:
                    diag_payload["price_raw"] = format_rounding_diag_number(
                        price_raw_dec_diag
                    )
                if qty_raw_dec_diag is not None:
                    diag_payload["qty_raw"] = format_rounding_diag_number(
                        qty_raw_dec_diag
                    )
                if stop_raw_dec_diag is not None:
                    diag_payload["stop_raw"] = format_rounding_diag_number(
                        stop_raw_dec_diag
                    )
                if tick_dec is not None:
                    diag_payload["tickSize"] = format_rounding_diag_number(tick_dec)
                if step_dec is not None:
                    diag_payload["stepSize"] = format_rounding_diag_number(step_dec)

                if (
                    price_raw_dec_diag is not None
                    and tick_dec not in (None, Decimal("0"))
                ):
                    try:
                        diag_payload["price_div_tick"] = str(
                            price_raw_dec_diag / tick_dec
                        )
                    except Exception:
                        diag_payload["price_div_tick"] = None

                price_multiple_raw = is_multiple(price_raw_dec_diag, tick_dec)
                qty_multiple_raw = is_multiple(qty_raw_dec_diag, step_dec)
                stop_multiple_raw = (
                    True
                    if stop_raw_dec_diag is None
                    else is_multiple(stop_raw_dec_diag, tick_dec)
                )

                diag_payload["is_multiple_price"] = (
                    bool(price_multiple_raw)
                    if price_multiple_raw is not None
                    else None
                )
                diag_payload["is_multiple_qty"] = (
                    bool(qty_multiple_raw)
                    if qty_multiple_raw is not None
                    else None
                )
                diag_payload["is_multiple_stop"] = (
                    True
                    if stop_multiple_raw is True
                    else (
                        bool(stop_multiple_raw)
                        if stop_multiple_raw is not None
                        else None
                    )
                )
            except Exception as exc:  # pragma: no cover - diagnostics only
                diag_payload["warn"] = str(exc)
            emit_rounding_diag(diag_payload, logger=logger)

        qty_dec = round_to_step(qty_raw_dec, step_size_dec, rounding=ROUND_DOWN)
        guard = apply_qty_guards(
            symbol,
            side,
            "LIMIT",
            entry_price_dec,
            qty_dec,
            filters,
            allow_increase=True,
        )
        if not guard.success or guard.qty is None:
            logger.warning(
                "%s precision_abort_entry symbol=%s market=%s timeframe=%s side=%s "
                "requested_price=%s adjusted_price=%s requested_qty=%s tickSize=%s "
                "stepSize=%s minNotional=%s minQty=%s reason=%s",
                log_prefix,
                symbol,
                market,
                timeframe,
                side,
                format_decimal(entry_price_raw_dec),
                format_decimal(entry_price_dec),
                format_decimal(qty_raw_dec),
                format_decimal(filters.tick_size),
                format_decimal(filters.step_size),
                format_decimal(filters.min_notional),
                format_decimal(filters.min_qty),
                guard.reason or "unknown",
            )
            return {"status": "precision_abort_entry", "symbol": symbol, "reason": guard.reason}

        qty_dec = guard.qty

        logger.info(
            "%s precision_entry symbol=%s market=%s timeframe=%s side=%s "
            "requested_price=%s adjusted_price=%s requested_qty=%s adjusted_qty=%s "
            "tickSize=%s stepSize=%s minNotional=%s minQty=%s action=%s notional=%s",
            log_prefix,
            symbol,
            market,
            timeframe,
            side,
            format_decimal(entry_price_raw_dec),
            format_decimal(entry_price_dec),
            format_decimal(qty_raw_dec),
            format_decimal(qty_dec),
            format_decimal(filters.tick_size),
            format_decimal(filters.step_size),
            format_decimal(filters.min_notional),
            format_decimal(filters.min_qty),
            "rounded" if guard.adjusted else "unchanged",
            format_decimal(guard.notional),
        )

        logger.info(
            "%s sizing entry_price=%s qty_target=%s qty_final=%s src=%s",
            log_prefix,
            to_api_str(entry_price_norm_dec),
            to_api_str(qty_target_dec),
            to_api_str(qty_dec),
            qty_target_src,
        )

        if qty_dec <= 0:
            logger.info("%s skip qty_not_positive", log_prefix)
            return {"status": "qty_invalid", "symbol": symbol}

        now_minute = int(now.timestamp() // 60)
        entry_client_id = sanitize_client_order_id(
            f"{cid_prefix}_{now_minute}_ENTRY"
        )
        tp_client_id = sanitize_client_order_id(f"{cid_prefix}_{now_minute}_TP")

        entry_price_requested = entry_price_norm_dec
        qty_requested = qty_dec

        try:
            entry_precision = compute_order_precision(
                price_requested=entry_price_requested,
                qty_requested=qty_requested,
                stop_requested=None,
                side=side,
                order_type="LIMIT",
                filters=filters,
                exchange=exch,
                symbol=symbol,
            )
        except OrderPrecisionError as exc:
            logger.warning(
                "%s %s",
                log_prefix,
                {
                    "tag": exc.tag,
                    "symbol": symbol,
                    "market": market,
                    "timeframe": timeframe,
                    "side": side,
                    "orderType": "LIMIT",
                    "reason": exc.reason,
                    "tickSize": format_decimal(filters.tick_size),
                    "stepSize": format_decimal(filters.step_size),
                },
            )
            log_order_not_sent(
                log_prefix=log_prefix,
                symbol=symbol,
                market=market,
                timeframe=timeframe,
                side=side,
                order_type="LIMIT",
                tag=exc.tag,
                reason=exc.reason,
                filters=filters,
            )
            return {
                "status": "precision_reject_entry",
                "symbol": symbol,
                "reason": exc.reason,
            }

        entry_price_adjusted = entry_precision.price_adjusted
        qty_adjusted = entry_precision.qty_adjusted

        if rounding_diag_enabled:
            diag_payload_post: dict[str, Any] = {
                "tag": "C_post_adjust",
                "symbol": symbol,
                "price_adj": None,
                "qty_adj": None,
                "stop_adj": None,
                "rule_price": None,
                "rule_qty": None,
                "is_multiple_price_adj": None,
                "is_multiple_qty_adj": None,
                "is_multiple_stop_adj": None,
            }
            try:
                price_adj_dec = to_decimal_or_none(entry_price_adjusted)
                qty_adj_dec = to_decimal_or_none(qty_adjusted)
                stop_adj_dec = to_decimal_or_none(None)
                tick_dec = to_decimal_or_none(filters.tick_size)
                step_dec = to_decimal_or_none(filters.step_size)

                if price_adj_dec is not None:
                    diag_payload_post["price_adj"] = format_rounding_diag_number(
                        price_adj_dec
                    )
                if qty_adj_dec is not None:
                    diag_payload_post["qty_adj"] = format_rounding_diag_number(
                        qty_adj_dec
                    )

                is_multiple_price_adj = is_multiple(price_adj_dec, tick_dec)
                is_multiple_qty_adj = is_multiple(qty_adj_dec, step_dec)
                is_multiple_stop_adj = True if stop_adj_dec is None else is_multiple(
                    stop_adj_dec, tick_dec
                )

                diag_payload_post["is_multiple_price_adj"] = (
                    bool(is_multiple_price_adj)
                    if is_multiple_price_adj is not None
                    else None
                )
                diag_payload_post["is_multiple_qty_adj"] = (
                    bool(is_multiple_qty_adj)
                    if is_multiple_qty_adj is not None
                    else None
                )
                diag_payload_post["is_multiple_stop_adj"] = (
                    True
                    if is_multiple_stop_adj is True
                    else (
                        bool(is_multiple_stop_adj)
                        if is_multiple_stop_adj is not None
                        else None
                    )
                )

                rule_price = "UP" if (side or "").upper() == "SELL" else "DOWN"
                rule_qty = "DOWN"
                diag_payload_post["rule_price"] = rule_price
                diag_payload_post["rule_qty"] = rule_qty
            except Exception as exc:  # pragma: no cover - diagnostics only
                diag_payload_post["warn"] = str(exc)
            emit_rounding_diag(diag_payload_post, logger=logger)

        log_precision_summary(
            log_prefix=log_prefix,
            symbol=symbol,
            market=market,
            timeframe=timeframe,
            side=side,
            order_type="LIMIT",
            tick_size=filters.tick_size,
            step_size=filters.step_size,
            min_notional=filters.min_notional,
            requested_price=entry_precision.price_requested,
            adjusted_price=entry_precision.price_adjusted,
            requested_qty=entry_precision.qty_requested,
            adjusted_qty=entry_precision.qty_adjusted,
        )

        entry_price_payload = to_api_str(entry_price_adjusted)
        qty_payload = to_api_str(qty_adjusted)

        try:
            order = exch.place_entry_limit(
                symbol,
                side,
                entry_price_payload,
                qty_payload,
                entry_client_id,
                timeInForce="GTC",
            )
        except TypeError:  # pragma: no cover - legacy interface fallback
            order = exch.place_entry_limit(
                symbol,
                side,
                entry_price_payload,
                qty_payload,
                entry_client_id,
            )

        tp_qty_guard = apply_qty_guards(
            symbol,
            exit_side,
            "LIMIT",
            tp_price_dec,
            qty_dec,
            filters,
            allow_increase=False,
        )
        if not tp_qty_guard.success or tp_qty_guard.qty is None:
            logger.warning(
                "%s precision_abort_tp symbol=%s market=%s timeframe=%s side=%s "
                "requested_price=%s adjusted_price=%s requested_qty=%s tickSize=%s "
                "stepSize=%s minNotional=%s minQty=%s reason=%s",
                log_prefix,
                symbol,
                market,
                timeframe,
                exit_side,
                format_decimal(tp_price_raw_dec),
                format_decimal(tp_price_dec),
                format_decimal(qty_dec),
                format_decimal(filters.tick_size),
                format_decimal(filters.step_size),
                format_decimal(filters.min_notional),
                format_decimal(filters.min_qty),
                tp_qty_guard.reason or "unknown",
            )
            return {"status": "precision_abort_tp", "symbol": symbol, "reason": tp_qty_guard.reason}

        logger.info(
            "%s precision_tp symbol=%s market=%s timeframe=%s side=%s "
            "requested_price=%s adjusted_price=%s qty=%s tickSize=%s stepSize=%s "
            "minNotional=%s minQty=%s action=%s notional=%s",
            log_prefix,
            symbol,
            market,
            timeframe,
            exit_side,
            format_decimal(tp_price_raw_dec),
            format_decimal(tp_price_dec),
            format_decimal(tp_qty_guard.qty),
            format_decimal(filters.tick_size),
            format_decimal(filters.step_size),
            format_decimal(filters.min_notional),
            format_decimal(filters.min_qty),
            "rounded" if tp_qty_guard.adjusted else "unchanged",
            format_decimal(tp_qty_guard.notional),
        )

        tp_price_requested = tp_price_norm_dec
        tp_stop_requested = tp_price_requested
        tp_qty_requested = tp_qty_guard.qty

        try:
            tp_precision = compute_order_precision(
                price_requested=tp_price_requested,
                qty_requested=tp_qty_requested,
                stop_requested=tp_stop_requested,
                side=exit_side,
                order_type="TP_LIMIT",
                filters=filters,
                exchange=exch,
                symbol=symbol,
            )
        except OrderPrecisionError as exc:
            logger.warning(
                "%s %s",
                log_prefix,
                {
                    "tag": exc.tag,
                    "symbol": symbol,
                    "market": market,
                    "timeframe": timeframe,
                    "side": exit_side,
                    "orderType": "TP_LIMIT",
                    "reason": exc.reason,
                    "tickSize": format_decimal(filters.tick_size),
                    "stepSize": format_decimal(filters.step_size),
                },
            )
            log_order_not_sent(
                log_prefix=log_prefix,
                symbol=symbol,
                market=market,
                timeframe=timeframe,
                side=exit_side,
                order_type="TP_LIMIT",
                tag=exc.tag,
                reason=exc.reason,
                filters=filters,
            )
            return {
                "status": "precision_reject_tp",
                "symbol": symbol,
                "reason": exc.reason,
            }

        log_precision_summary(
            log_prefix=log_prefix,
            symbol=symbol,
            market=market,
            timeframe=timeframe,
            side=exit_side,
            order_type="TP_LIMIT",
            tick_size=filters.tick_size,
            step_size=filters.step_size,
            min_notional=filters.min_notional,
            requested_price=tp_precision.price_requested,
            adjusted_price=tp_precision.price_adjusted,
            requested_qty=tp_precision.qty_requested,
            adjusted_qty=tp_precision.qty_adjusted,
            requested_stop=tp_precision.stop_requested,
            adjusted_stop=tp_precision.stop_adjusted,
        )

        tp_price_payload = to_api_str(tp_precision.price_adjusted)
        tp_qty_payload = to_api_str(tp_precision.qty_adjusted)

        persisted = persist_tp_value(
            symbol, tp_precision.price_adjusted, now.timestamp()
        )
        # TODO: Cuando implementemos SL: aplicar SL únicamente al cierre de vela fuera del patrón
        # en el timeframe WEDGE_TIMEFRAME; hasta entonces, sin SL automático.
        logger.info(
            "%s order entry placed side=%s price=%s qty=%s clientId=%s tp=%s persisted=%s",
            log_prefix,
            side,
            entry_price_payload,
            qty_payload,
            entry_client_id,
            tp_price_payload,
            persisted,
        )

        return {
            "status": "entry_order_placed",
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price_payload,
            "qty": qty_payload,
            "clientOrderId": entry_client_id,
            "tp_price": tp_price_payload,
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
        *,
        market: str,
        timeframe: str,
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

        side = "SELL" if is_long else "BUY"

        try:
            filters = get_symbol_filters(exch, symbol, market)
        except ValueError as exc:
            logger.warning(
                "%s %s",
                log_prefix,
                {
                    "tag": "ORDER_REJECT_TICK_INVALID",
                    "symbol": symbol,
                    "market": market,
                    "timeframe": timeframe,
                    "side": side,
                    "orderType": "TP_LIMIT",
                    "reason": str(exc),
                },
            )
            log_order_not_sent(
                log_prefix=log_prefix,
                symbol=symbol,
                market=market,
                timeframe=timeframe,
                side=side,
                order_type="TP_LIMIT",
                tag="ORDER_REJECT_TICK_INVALID",
                reason=str(exc),
            )
            return

        tp_price_raw_dec = Decimal(str(tp_value))
        tp_price_dec = to_decimal(
            exch.round_price_to_tick(symbol, tp_price_raw_dec)
        )
        qty_raw_dec = to_decimal(qty)
        qty_dec = round_to_step(qty_raw_dec, filters.step_size, rounding=ROUND_DOWN)

        guard = apply_qty_guards(
            symbol,
            side,
            "LIMIT",
            tp_price_dec,
            qty_dec,
            filters,
            allow_increase=False,
        )

        if not guard.success or guard.qty is None or guard.qty <= 0:
            logger.warning(
                "%s tp_precision_abort symbol=%s market=%s timeframe=%s side=%s "
                "requested_price=%s adjusted_price=%s requested_qty=%s tickSize=%s "
                "stepSize=%s minNotional=%s minQty=%s reason=%s",
                log_prefix,
                symbol,
                market,
                timeframe,
                side,
                format_decimal(tp_price_raw_dec),
                format_decimal(tp_price_dec),
                format_decimal(qty_raw_dec),
                format_decimal(filters.tick_size),
                format_decimal(filters.step_size),
                format_decimal(filters.min_notional),
                format_decimal(filters.min_qty),
                guard.reason or "unknown",
            )
            return

        logger.info(
            "%s tp_precision symbol=%s market=%s timeframe=%s side=%s "
            "requested_price=%s adjusted_price=%s requested_qty=%s adjusted_qty=%s "
            "tickSize=%s stepSize=%s minNotional=%s minQty=%s action=%s notional=%s",
            log_prefix,
            symbol,
            market,
            timeframe,
            side,
            format_decimal(tp_price_raw_dec),
            format_decimal(tp_price_dec),
            format_decimal(qty_raw_dec),
            format_decimal(guard.qty),
            format_decimal(filters.tick_size),
            format_decimal(filters.step_size),
            format_decimal(filters.min_notional),
            format_decimal(filters.min_qty),
            "rounded" if guard.adjusted else "unchanged",
            format_decimal(guard.notional),
        )

        tp_qty_dec = guard.qty

        if tp_qty_dec <= 0:
            logger.info("%s tp_skip qty<=0", log_prefix)
            return
        client_id = sanitize_client_order_id(
            f"{cid_prefix}_{int(now.timestamp() // 60)}_TP"
        )
        tp_price_requested = tp_price_dec
        tp_stop_requested = tp_price_requested
        tp_qty_requested = tp_qty_dec

        try:
            tp_precision = compute_order_precision(
                price_requested=tp_price_requested,
                qty_requested=tp_qty_requested,
                stop_requested=tp_stop_requested,
                side=side,
                order_type="TP_LIMIT",
                filters=filters,
                exchange=exch,
                symbol=symbol,
            )
        except OrderPrecisionError as exc:
            logger.warning(
                "%s %s",
                log_prefix,
                {
                    "tag": exc.tag,
                    "symbol": symbol,
                    "market": market,
                    "timeframe": timeframe,
                    "side": side,
                    "orderType": "TP_LIMIT",
                    "reason": exc.reason,
                    "tickSize": format_decimal(filters.tick_size),
                    "stepSize": format_decimal(filters.step_size),
                },
            )
            log_order_not_sent(
                log_prefix=log_prefix,
                symbol=symbol,
                market=market,
                timeframe=timeframe,
                side=side,
                order_type="TP_LIMIT",
                tag=exc.tag,
                reason=exc.reason,
                filters=filters,
            )
            return

        log_precision_summary(
            log_prefix=log_prefix,
            symbol=symbol,
            market=market,
            timeframe=timeframe,
            side=side,
            order_type="TP_LIMIT",
            tick_size=filters.tick_size,
            step_size=filters.step_size,
            min_notional=filters.min_notional,
            requested_price=tp_precision.price_requested,
            adjusted_price=tp_precision.price_adjusted,
            requested_qty=tp_precision.qty_requested,
            adjusted_qty=tp_precision.qty_adjusted,
            requested_stop=tp_precision.stop_requested,
            adjusted_stop=tp_precision.stop_adjusted,
        )

        tp_price_payload = to_api_str(tp_precision.price_adjusted)
        tp_qty_payload = to_api_str(tp_precision.qty_adjusted)

        try:
            exch.place_tp_reduce_only(
                symbol, side, tp_price_payload, tp_qty_payload, client_id
            )
            logger.info(
                "%s tp_order symbol=%s market=%s timeframe=%s side=%s price=%s qty=%s clientId=%s",
                log_prefix,
                symbol,
                market,
                timeframe,
                side,
                tp_price_payload,
                tp_qty_payload,
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
