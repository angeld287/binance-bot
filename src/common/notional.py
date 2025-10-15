"""Helpers to enforce min notional constraints with safety buffers."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_UP
from typing import List

from .precision import round_to_step, round_to_tick, to_decimal


@dataclass(slots=True)
class NotionalSizingResult:
    """Result payload returned by :func:`ensure_min_notional_with_buffers`."""

    qty_raw: Decimal
    qty_rounded: Decimal
    price_used: Decimal
    notional: Decimal
    min_notional: Decimal
    min_notional_target: Decimal
    step_size: Decimal
    tick_size: Decimal
    buffer_pct: Decimal
    buffer_usd: Decimal
    adjustments: List[str] = field(default_factory=list)


def _coerce_positive(value: Decimal) -> Decimal:
    if value <= 0:
        return Decimal("0")
    return value


def ensure_min_notional_with_buffers(
    *,
    qty: object,
    price: object,
    side: str,
    step_size: object,
    tick_size: object,
    min_notional: object,
    buffer_pct: object,
    buffer_usd: object,
    strict_rounding: bool = True,
) -> NotionalSizingResult:
    """Return a :class:`NotionalSizingResult` enforcing min notional buffers."""

    qty_dec = to_decimal(qty)
    step = to_decimal(step_size)
    tick = to_decimal(tick_size)
    min_notional_dec = _coerce_positive(to_decimal(min_notional))

    try:
        buffer_pct_dec = _coerce_positive(to_decimal(buffer_pct))
    except Exception:
        buffer_pct_dec = Decimal("0")
    try:
        buffer_usd_dec = _coerce_positive(to_decimal(buffer_usd))
    except Exception:
        buffer_usd_dec = Decimal("0")

    side_norm = (side or "").upper()

    price_dec = to_decimal(price)
    if strict_rounding:
        if tick > 0:
            price_dec = round_to_tick(price_dec, tick, side=side_norm)
    min_notional_target = min_notional_dec
    if min_notional_dec > 0:
        pct_target = min_notional_dec * (Decimal("1") + buffer_pct_dec)
        usd_target = min_notional_dec + buffer_usd_dec
        min_notional_target = max(min_notional_dec, pct_target, usd_target)

    qty_rounded = qty_dec
    if strict_rounding and step > 0:
        qty_rounded = round_to_step(qty_dec, step)

    adjustments: List[str] = []
    notional = price_dec * qty_rounded

    if min_notional_target > 0 and price_dec > 0:
        if notional < min_notional_target:
            required_qty = min_notional_target / price_dec
            if step > 0:
                if strict_rounding:
                    required_qty = round_to_step(required_qty, step, rounding=ROUND_UP)
                else:
                    steps = (required_qty / step).to_integral_value(rounding=ROUND_UP)
                    required_qty = steps * step
            if required_qty > qty_rounded:
                adjustments.append("qty_increased_for_min_notional")
                qty_rounded = required_qty
                notional = price_dec * qty_rounded
            if step > 0 and notional < min_notional_target:
                # Add one step at a time until the constraint is met.
                adjustments.append("qty_increment_loop")
                max_iterations = 10_000
                iterations = 0
                while notional < min_notional_target and iterations < max_iterations:
                    qty_rounded += step
                    qty_rounded = round_to_step(qty_rounded, step, rounding=ROUND_UP)
                    notional = price_dec * qty_rounded
                    iterations += 1
                if iterations >= max_iterations:
                    adjustments.append("qty_increment_failed")
    elif min_notional_target > 0 and price_dec <= 0:
        adjustments.append("price_non_positive")

    return NotionalSizingResult(
        qty_raw=qty_dec,
        qty_rounded=qty_rounded,
        price_used=price_dec,
        notional=notional,
        min_notional=min_notional_dec,
        min_notional_target=min_notional_target,
        step_size=step,
        tick_size=tick,
        buffer_pct=buffer_pct_dec,
        buffer_usd=buffer_usd_dec,
        adjustments=adjustments,
    )
