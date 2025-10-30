"""Classify EMA directionality based on slopes and spread."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmaTrendThresholds:
    neutral_fast: float = 0.0005
    neutral_slow: float = 0.0002
    strong_fast: float = 0.0020
    strong_slow: float = 0.0010
    strong_spread: float = 0.0020


def classify_trend(
    slope_fast: float | None,
    slope_slow: float | None,
    spread: float | None,
    thresholds: EmaTrendThresholds,
) -> tuple[str, str | None]:
    """Return ``(classification, notes)`` for the EMA slopes."""

    if slope_fast is None or slope_slow is None or spread is None:
        return "NEUTRO", None

    sf = slope_fast
    ss = slope_slow
    sp = spread

    notes = f"fast={sf:.6f} slow={ss:.6f} spread={sp:.6f}"

    if abs(sf) < thresholds.neutral_fast and abs(ss) < thresholds.neutral_slow:
        return "NEUTRO", f"Pendientes planas ({notes})"

    strong_up = sf > thresholds.strong_fast and ss > thresholds.strong_slow and sp > thresholds.strong_spread
    strong_down = sf < -thresholds.strong_fast and ss < -thresholds.strong_slow and sp < -thresholds.strong_spread

    if strong_up:
        return "ASC_FUERTE", f"Ambas EMAs suben con fuerza ({notes})"
    if strong_down:
        return "DESC_FUERTE", f"Ambas EMAs bajan con fuerza ({notes})"

    if sf > 0 and ss > 0:
        return "ASC_SUAVE", f"EMAs ascendentes ({notes})"
    if sf < 0 and ss < 0:
        return "DESC_SUAVE", f"EMAs descendentes ({notes})"

    return "NEUTRO", f"Pendientes mixtas ({notes})"
