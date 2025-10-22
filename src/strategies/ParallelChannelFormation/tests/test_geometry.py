import math
from dataclasses import dataclass
from decimal import Decimal
import sys
import types
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[3]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

strategies_pkg = types.ModuleType("strategies")
strategies_pkg.__path__ = [str(PROJECT_SRC / "strategies")]
sys.modules.setdefault("strategies", strategies_pkg)

if "strategies.wedge_formation" not in sys.modules:
    wedge_pkg = types.ModuleType("strategies.wedge_formation")
    wedge_pkg.__path__ = []
    sys.modules["strategies.wedge_formation"] = wedge_pkg

if "strategies.wedge_formation.strategy" not in sys.modules:
    wedge_strategy = types.ModuleType("strategies.wedge_formation.strategy")

    @dataclass
    class SymbolFilters:
        tick_size: Decimal
        step_size: Decimal
        min_notional: Decimal
        min_qty: Decimal

    @dataclass
    class PrecisionResult:
        price_requested: Decimal | None
        qty_requested: Decimal | None
        stop_requested: Decimal | None
        side: str
        order_type: str
        filters: SymbolFilters
        exchange: object
        symbol: str
        price_adjusted: Decimal | None
        qty_adjusted: Decimal | None
        stop_adjusted: Decimal | None = None
        price_payload: str | None = None
        stop_payload: str | None = None
        price_decimals: int | None = None
        stop_decimals: int | None = None

    @dataclass
    class QtyGuardResult:
        success: bool
        qty: Decimal | None
        adjusted: bool
        reason: str | None

    class OrderPrecisionError(Exception):
        def __init__(self, tag: str, reason: str) -> None:
            super().__init__(reason)
            self.tag = tag
            self.reason = reason

    def compute_order_precision(**kwargs):
        price = kwargs.get("price_requested")
        qty = kwargs.get("qty_requested")
        return PrecisionResult(
            price_requested=price,
            qty_requested=qty,
            stop_requested=None,
            side=kwargs.get("side", ""),
            order_type=kwargs.get("order_type", ""),
            filters=kwargs.get("filters"),
            exchange=kwargs.get("exchange"),
            symbol=kwargs.get("symbol", ""),
            price_adjusted=price,
            qty_adjusted=qty,
        )

    def apply_qty_guards(**kwargs):
        return QtyGuardResult(True, kwargs.get("qty_dec"), False, None)

    def get_symbol_filters(exchange, symbol):
        raw = exchange.get_symbol_filters(symbol)
        price = Decimal(str(raw["PRICE_FILTER"]["tickSize"]))
        step = Decimal(str(raw["LOT_SIZE"]["stepSize"]))
        min_notional = Decimal(str(raw["MIN_NOTIONAL"]["minNotional"]))
        min_qty = Decimal(str(raw["LOT_SIZE"].get("minQty", "0")))
        return SymbolFilters(price, step, min_notional, min_qty)

    class WedgeFormationStrategy:
        @staticmethod
        def _compute_atr(candles, period: int = 14):
            if len(candles) < 2:
                return 0.0
            trs = []
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

    wedge_strategy.SymbolFilters = SymbolFilters
    wedge_strategy.apply_qty_guards = apply_qty_guards
    wedge_strategy.compute_order_precision = compute_order_precision
    wedge_strategy.get_symbol_filters = get_symbol_filters
    wedge_strategy.OrderPrecisionError = OrderPrecisionError
    wedge_strategy.WedgeFormationStrategy = WedgeFormationStrategy
    sys.modules["strategies.wedge_formation.strategy"] = wedge_strategy

if "strategies.breakout_dual_tf" not in sys.modules:
    dual_pkg = types.ModuleType("strategies.breakout_dual_tf")
    dual_pkg.__path__ = []
    sys.modules["strategies.breakout_dual_tf"] = dual_pkg

if "strategies.breakout_dual_tf.filters" not in sys.modules:
    filters_pkg = types.ModuleType("strategies.breakout_dual_tf.filters")
    filters_pkg.__path__ = []
    sys.modules["strategies.breakout_dual_tf.filters"] = filters_pkg

if "strategies.breakout_dual_tf.filters.ema_distance" not in sys.modules:
    ema_module = types.ModuleType("strategies.breakout_dual_tf.filters.ema_distance")

    class DummyResult:
        def __init__(self, ema_fast, ema_slow):
            self.ok = True
            self.reason = ""
            self.ema7 = float(ema_fast or 0.0)
            self.ema25 = float(ema_slow or 0.0)
            self.nearest_ema_label = "ema7"
            self.dist_to_ema7_pct = 0.0
            self.dist_to_ema25_pct = 0.0

    def compute_ema_distance(ohlc, ema_fast, ema_slow, side="", **_kwargs):
        return DummyResult(ema_fast, ema_slow)

    ema_module.compute_ema_distance = compute_ema_distance
    sys.modules["strategies.breakout_dual_tf.filters.ema_distance"] = ema_module

from strategies.ParallelChannelFormation.geometry_utils import (
    are_parallel,
    find_pivots,
    fit_line,
    has_min_duration,
    has_min_touches,
    vertical_gap_pct,
)


def _sample_candles():
    candles = []
    base = 100.0
    for idx in range(40):
        high = base + math.sin(idx / 3) * 2 + idx * 0.05
        low = base - math.sin(idx / 3) * 2 - idx * 0.05
        candles.append([idx, base, high, low, base + math.sin(idx), 1.0])
    return candles


def test_parallel_detection_with_tolerance():
    candles = _sample_candles()
    pivots_high, pivots_low = find_pivots(candles, left=2, right=2)
    upper = fit_line(pivots_high)
    lower = fit_line(pivots_low)
    assert upper is not None and lower is not None
    assert are_parallel(upper.slope, lower.slope, tolerance=0.2)


def test_vertical_gap_and_duration_constraints():
    candles = _sample_candles()
    pivots_high, pivots_low = find_pivots(candles, left=2, right=2)
    upper = fit_line(pivots_high)
    lower = fit_line(pivots_low)
    assert upper is not None and lower is not None
    gap_pct = vertical_gap_pct(upper, lower, price_ref=float(candles[-1][4]), index=len(candles) - 1)
    assert gap_pct > 0.01
    assert has_min_duration(pivots_high, min_bars=10)
    assert has_min_duration(pivots_low, min_bars=10)


def test_min_touches_respects_tolerance():
    candles = _sample_candles()
    pivots_high, pivots_low = find_pivots(candles, left=2, right=2)
    upper = fit_line(pivots_high)
    lower = fit_line(pivots_low)
    assert upper is not None and lower is not None
    assert has_min_touches(upper, pivots_high, tolerance=3.0, min_touches=2)
    assert has_min_touches(lower, pivots_low, tolerance=3.0, min_touches=2)
