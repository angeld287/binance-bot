"""Geometry helpers to detect parallel price channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Line:
    """Simple line representation using slope and intercept."""

    slope: float
    intercept: float

    def value_at(self, x: float) -> float:
        return self.slope * x + self.intercept


def find_pivots(
    candles: Sequence[Sequence[float]],
    *,
    left: int = 2,
    right: int = 2,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return pivot highs and lows detected on ``candles``."""

    if not candles:
        return [], []

    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    pivots_high: list[tuple[int, float]] = []
    pivots_low: list[tuple[int, float]] = []

    for idx in range(left, len(candles) - right):
        high = highs[idx]
        low = lows[idx]
        if high >= max(highs[idx - left : idx]) and high >= max(
            highs[idx + 1 : idx + 1 + right]
        ):
            pivots_high.append((idx, high))
        if low <= min(lows[idx - left : idx]) and low <= min(
            lows[idx + 1 : idx + 1 + right]
        ):
            pivots_low.append((idx, low))

    return pivots_high, pivots_low


def fit_line(pivots: Sequence[tuple[int, float]]) -> Line | None:
    """Return the least-squares line fitting ``pivots`` or ``None``."""

    if len(pivots) < 2:
        return None
    sum_x = sum(idx for idx, _ in pivots)
    sum_y = sum(price for _, price in pivots)
    sum_xx = sum(idx * idx for idx, _ in pivots)
    sum_xy = sum(idx * price for idx, price in pivots)
    n = float(len(pivots))
    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0:
        return None
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return Line(slope=slope, intercept=intercept)


def slope(line: Line | None) -> float:
    """Return the slope of ``line`` or 0 when ``None``."""

    return 0.0 if line is None else float(line.slope)


def are_parallel(m1: float, m2: float, tolerance: float) -> bool:
    """Return ``True`` if slopes ``m1`` and ``m2`` differ within ``tolerance``."""

    return abs(float(m1) - float(m2)) <= abs(float(tolerance))


def vertical_gap_pct(line_sup: Line, line_inf: Line, price_ref: float, index: float) -> float:
    """Return vertical distance between ``line_sup`` and ``line_inf`` relative to ``price_ref``."""

    sup_val = line_sup.value_at(index)
    inf_val = line_inf.value_at(index)
    if price_ref == 0:
        return 0.0
    return abs(sup_val - inf_val) / abs(price_ref)


def has_min_touches(
    line: Line,
    pivots: Iterable[tuple[int, float]],
    *,
    tolerance: float,
    min_touches: int,
) -> bool:
    """Return ``True`` if ``line`` is touched by at least ``min_touches`` pivots."""

    touches = 0
    tol = abs(float(tolerance))
    for idx, price in pivots:
        expected = line.value_at(idx)
        if abs(expected - price) <= tol:
            touches += 1
        if touches >= min_touches:
            return True
    return False


def has_min_duration(pivots: Sequence[tuple[int, float]], *, min_bars: int) -> bool:
    """Return ``True`` if the pivot span covers at least ``min_bars`` bars."""

    if len(pivots) < 2:
        return False
    start = pivots[0][0]
    end = pivots[-1][0]
    return (end - start) >= max(0, int(min_bars))


__all__ = [
    "Line",
    "find_pivots",
    "fit_line",
    "slope",
    "are_parallel",
    "vertical_gap_pct",
    "has_min_touches",
    "has_min_duration",
]
