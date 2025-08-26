"""Breakout trading strategy implementation."""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List

from analysis.pattern_detection import detect_patterns

from .base import Strategy

ANALYSIS_WINDOW = 12

class BreakoutStrategy(Strategy):
    """Simple breakout strategy.

    The entry logic is based on detecting swing highs/lows on recent
    candles.  Exit management is intentionally lightweight; in the real
    bot it would handle trailing stops, take profit and other
    house‑keeping tasks.  For the purposes of this kata the method merely
    logs its invocation so that the orchestrator can call it without
    side effects.
    """

    name = "breakout"

    # ------------------------------------------------------------------
    # Entry planning
    # ------------------------------------------------------------------
    def _last_swing_high(self, ohlcv: List[List[float]], window: int = ANALYSIS_WINDOW) -> Optional[float]:
        highs = [c[2] for c in ohlcv]
        for i in range(len(highs) - window - 1, window, -1):
            local = highs[i - window : i + window + 1]
            if highs[i] == max(local):
                return highs[i]
        return None

    def _last_swing_low(self, ohlcv: List[List[float]], window: int = ANALYSIS_WINDOW) -> Optional[float]:
        lows = [c[3] for c in ohlcv]
        for i in range(len(lows) - window - 1, window, -1):
            local = lows[i - window : i + window + 1]
            if lows[i] == min(local):
                return lows[i]
        return None

    def plan_entry(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        exchange = ctx["client"]
        symbol = ctx["symbol"]
        for tf in ["15m", "30m"]:
            try:
                klines = exchange.futures_klines(symbol=symbol.replace("/", ""), interval=tf, limit=50)
                ohlcv = [[k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in klines]
                if len(ohlcv) < ANALYSIS_WINDOW * 2 + 2:
                    continue
                price = ohlcv[-1][4]
                last_high = self._last_swing_high(ohlcv[:-1])
                last_low = self._last_swing_low(ohlcv[:-1])
                if last_high and price > last_high:
                    patterns = detect_patterns(ohlcv)
                    price_range = (last_low if last_low else price * 0.99, last_high)
                    return {
                        "side": "buy",
                        "trigger": last_high,
                        "patterns": patterns,
                        "price_range": price_range,
                    }
                if last_low and price < last_low:
                    patterns = detect_patterns(ohlcv)
                    price_range = (last_low, last_high if last_high else price * 1.01)
                    return {
                        "side": "sell",
                        "trigger": last_low,
                        "patterns": patterns,
                        "price_range": price_range,
                    }
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Exit management
    # ------------------------------------------------------------------
    def manage_exits(self, ctx: Dict[str, Any]) -> None:  # pragma: no cover - placeholder
        """Handle stop loss / take profit updates.

        The real project contains complex trailing logic.  Here we only
        emit a log message so tests can ensure the orchestrator delegates
        exit handling to the strategy.
        """
        logger = ctx.get("log")
        if callable(logger):
            logger("[STRAT:breakout] manage_exits called")

    def close_mode(self) -> str:
        return "default"
