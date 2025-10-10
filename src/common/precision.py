"""Decimal-based helpers to enforce Binance Futures precision rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from threading import RLock
from typing import Any, Callable, Dict, Protocol

__all__ = [
    "Decimal",
    "ROUND_DOWN",
    "ROUND_UP",
    "format_decimal",
    "round_to_step",
    "round_to_tick",
    "to_decimal",
    "ExchangeFiltersProvider",
    "FiltersCache",
]


def to_decimal(value: Any) -> Decimal:
    """Return ``value`` converted to :class:`~decimal.Decimal`.

    ``float`` inputs are stringified first to avoid inheriting binary
    representation artefacts. ``None`` is treated as ``0``.
    """

    if isinstance(value, Decimal):
        return value
    if value is None:
        return Decimal("0")
    if isinstance(value, (int, str)):
        return Decimal(str(value))
    try:
        return Decimal(str(float(value)))
    except (InvalidOperation, TypeError, ValueError) as err:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from err


def round_to_tick(
    price: Any,
    tick_size: Any,
    *,
    side: str | None = None,
) -> Decimal:
    """Round ``price`` to the closest valid ``tick_size`` multiple.

    ``side`` determines the rounding direction according to Binance rules:
    BUY orders round down, SELL orders round up. If ``side`` is omitted the
    value is rounded down.
    """

    tick = to_decimal(tick_size)
    if tick <= 0:
        return to_decimal(price)
    px = to_decimal(price)
    side_norm = (side or "").upper()
    rounding = ROUND_DOWN if side_norm != "SELL" else ROUND_UP
    return px.quantize(tick, rounding=rounding)


def round_to_step(
    qty: Any,
    step_size: Any,
    *,
    rounding: str = ROUND_DOWN,
) -> Decimal:
    """Round ``qty`` to a valid ``step_size`` multiple using ``rounding``."""

    step = to_decimal(step_size)
    if step <= 0:
        return to_decimal(qty)
    quantity = to_decimal(qty)
    return quantity.quantize(step, rounding=rounding)


def format_decimal(value: Any, max_places: int | None = None) -> str:
    """Return a string serialisation of ``value`` without binary tails."""

    dec = to_decimal(value)
    if max_places is not None:
        quantum = Decimal(1).scaleb(-int(max_places))
        dec = dec.quantize(quantum, rounding=ROUND_DOWN)
    text = format(dec, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


# ---------------------------------------------------------------------------
# Exchange filters cache ----------------------------------------------------


class ExchangeFiltersProvider(Protocol):
    """Protocol exposing the subset of ``binance.Client`` used below."""

    def futures_exchange_info(self) -> dict[str, Any]:  # pragma: no cover - lib
        ...


@dataclass(slots=True)
class _CachedFilters:
    filters: Dict[str, Dict[str, Any]]
    expires_at: datetime


class FiltersCache:
    """Cache Binance ``exchangeInfo`` responses with TTL per symbol."""

    def __init__(self, *, ttl: timedelta = timedelta(minutes=5)) -> None:
        self._ttl = ttl
        self._cache: _CachedFilters | None = None
        self._lock = RLock()

    def get(
        self,
        client: ExchangeFiltersProvider,
        symbol: str,
        *,
        on_refresh_error: Callable[[Exception], None] | None = None,
    ) -> Dict[str, Any]:
        now = datetime.utcnow()
        with self._lock:
            if self._cache is None or self._cache.expires_at <= now:
                try:
                    data = client.futures_exchange_info()
                    filters = {
                        sym.get("symbol", ""): {
                            f["filterType"]: f for f in sym.get("filters", [])
                        }
                        for sym in data.get("symbols", [])
                    }
                    self._cache = _CachedFilters(
                        filters=filters, expires_at=now + self._ttl
                    )
                except Exception as err:  # pragma: no cover - network failure
                    if on_refresh_error is not None:
                        on_refresh_error(err)
                    if self._cache is None:
                        raise

            assert self._cache is not None  # for type-checkers
            try:
                return self._cache.filters[symbol]
            except KeyError as err:
                raise ValueError(f"Symbol {symbol} not found in exchangeInfo") from err
