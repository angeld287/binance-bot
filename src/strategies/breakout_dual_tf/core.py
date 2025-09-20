"""Dual timeframe breakout strategy implementation.

This module introduces a breakout strategy that analyses support and
resistance levels on a higher timeframe and executes trades on a lower
timeframe.  The implementation follows the requirements described in the
user story and keeps the logic self-contained so it can be easily reused by
other orchestration layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence
import json
import logging
import math

from config.settings import get_stop_loss_pct, get_take_profit_pct
from core.domain.models.Signal import Signal
from core.ports.broker import BrokerPort
from core.ports.market_data import MarketDataPort
from core.ports.settings import SettingsProvider, get_symbol
from core.ports.strategy import Strategy

logger = logging.getLogger("bot.strategy.breakout_dual_tf")


# ---------------------------------------------------------------------------
# Helper dataclasses


@dataclass(slots=True)
class Level:
    """Support/resistance level detected on the higher timeframe."""

    price: float
    level_type: str  # "S" or "R"
    timestamp: int
    score: float = 0.0

    def key(self) -> tuple[float, str]:
        return (round(self.price, 8), self.level_type)


@dataclass(slots=True)
class BreakoutSignalPayload:
    """Detailed breakout signal used for order generation."""

    symbol: str
    action: str
    direction: str
    level: Level
    entry_price: float
    sl: float
    tp1: float
    tp2: float
    rr: float
    atr: float
    volume_rel: float
    ema_fast: float
    ema_slow: float
    exec_tf: str
    candle: Sequence[float]
    swing: float


@dataclass(slots=True)
class PendingBreakout:
    """State machine for break-and-retest logic."""

    level: Level
    direction: str
    breakout_ts: int
    breakout_close: float
    atr: float
    exec_tf: str
    highest_high: float
    lowest_low: float
    retest_seen: bool = False
    retest_ts: int | None = None

    def is_expired(self, now_ts: int, bar_ms: int, timeout_bars: int) -> bool:
        return now_ts - self.breakout_ts > timeout_bars * bar_ms


@dataclass(slots=True)
class CooldownEntry:
    """Track cooldown windows after SL/BE exits."""

    level_price: float
    direction: str
    start_ts: int
    expires_ts: int
    reason: str = "sl"


# ---------------------------------------------------------------------------
# Utility helpers


def _interval_to_minutes(interval: str) -> int:
    units = {"m": 1, "h": 60, "d": 1440}
    try:
        return int(interval[:-1]) * units[interval[-1]]
    except (KeyError, ValueError):  # pragma: no cover - defensive guard
        return 60


def downscale_interval(interval: str) -> str:
    """Return execution timeframe derived from ``interval``."""

    mapping = {
        "4h": "30m",
        "2h": "15m",
        "1h": "15m",
        "30m": "5m",
        "15m": "5m",
    }
    return mapping.get(interval, interval)


def _compute_atr(candles: Sequence[Sequence[float]], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    closes = [float(c[4]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    trs: list[float] = []
    prev_close = closes[0]
    for h, l, c in zip(highs[1:], lows[1:], closes[1:]):
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    if not trs:
        return 0.0
    last = trs[-period:]
    return sum(last) / len(last)


def _compute_relative_volume(candles: Sequence[Sequence[float]], lookback: int = 20) -> float:
    vols = [float(c[5]) for c in candles if len(c) > 5]
    if len(vols) < 2:
        return 0.0
    last_volume = vols[-1]
    history = vols[-(lookback + 1) : -1]
    if not history:
        history = vols[:-1]
    if not history:
        return 0.0
    avg = sum(history) / len(history)
    if avg <= 0:
        return 0.0
    return last_volume / avg


def _ema(values: Sequence[float], period: int) -> float:
    if not values:
        return 0.0
    multiplier = 2 / (period + 1)
    ema_val = values[0]
    for value in values[1:]:
        ema_val = (value - ema_val) * multiplier + ema_val
    return ema_val


def _find_recent_swing(
    candles: Sequence[Sequence[float]],
    direction: str,
    left: int = 2,
    right: int = 2,
) -> float:
    if len(candles) < left + right + 1:
        return float(candles[-1][3 if direction == "LONG" else 2])
    start = len(candles) - right - 1
    for idx in range(start, left - 1, -1):
        lows = [float(c[3]) for c in candles]
        highs = [float(c[2]) for c in candles]
        if direction == "LONG":
            pivot_low = lows[idx]
            if pivot_low <= min(lows[idx - left : idx]) and pivot_low <= min(
                lows[idx + 1 : idx + 1 + right]
            ):
                return pivot_low
        else:
            pivot_high = highs[idx]
            if pivot_high >= max(highs[idx - left : idx]) and pivot_high >= max(
                highs[idx + 1 : idx + 1 + right]
            ):
                return pivot_high
    return float(candles[-1][3 if direction == "LONG" else 2])


def _snap_price(price: float, tick: float, *, side: str) -> float:
    if tick <= 0:
        return price
    if side == "BUY":
        return math.floor(price / tick) * tick
    return math.ceil(price / tick) * tick


# ---------------------------------------------------------------------------
# Strategy implementation


class BreakoutDualTFStrategy(Strategy):
    """Breakout strategy combining dual timeframe analysis."""

    DEFAULT_CONFIG: dict[str, Any] = {
        "LEVEL_WINDOW": 50,
        "PIVOT_LEFT": 2,
        "PIVOT_RIGHT": 2,
        "K_ATR": 0.3,
        "VOL_REL_MIN": 1.5,
        "RR_MIN": 1.8,
        "RETEST_TOL_ATR": 0.2,
        "RETEST_TIMEOUT": 6,
        "USE_RETEST": True,
        "COOLDOWN_BARS": 8,
        "MAX_RETRIES": 1,
        "COOLDOWN_BAND_ATR": 0.5,
    }

    def __init__(
        self,
        market_data: MarketDataPort | None,
        broker: BrokerPort | None,
        settings: SettingsProvider,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._market_data = market_data
        self._broker = broker
        self._settings = settings
        self._config = dict(self.DEFAULT_CONFIG)
        if config:
            self._config.update(config)
        max_retries_env = getattr(settings, "MAX_RETRIES", None)
        cooldown_env = getattr(settings, "COOLDOWN_BARS", None)
        if max_retries_env is not None:
            try:
                self._config["MAX_RETRIES"] = int(max_retries_env)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
        if cooldown_env is not None:
            try:
                self._config["COOLDOWN_BARS"] = int(cooldown_env)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass

        self._level_attempts: dict[tuple[float, str, str], int] = {}
        self._cooldowns: list[CooldownEntry] = []
        self._pending_breakouts: dict[tuple[float, str, str], PendingBreakout] = {}
        self._last_level_atr: float = 0.0
        self._last_payload: BreakoutSignalPayload | None = None
        self._exec_tf: str = downscale_interval(getattr(settings, "INTERVAL", "1h"))
        self._bar_ms: int = _interval_to_minutes(self._exec_tf) * 60_000

    # ------------------------------------------------------------------
    def _fetch_candles(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        provider = self._market_data
        if provider is None:
            return []
        if hasattr(provider, "fetch_ohlcv"):
            candles = provider.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        else:  # pragma: no cover - fallback for legacy providers
            lookback = _interval_to_minutes(timeframe) * limit
            candles = provider.get_klines(symbol=symbol, interval=timeframe, lookback_min=lookback)
        return [
            [
                float(c[0]),
                float(c[1]),
                float(c[2]),
                float(c[3]),
                float(c[4]),
                float(c[5]) if len(c) > 5 else 0.0,
            ]
            for c in candles
        ]

    # ------------------------------------------------------------------
    def get_levels(self, symbol: str, tf: str | None = None) -> list[Level]:
        timeframe = tf or getattr(self._settings, "INTERVAL", "1h")
        limit = max(int(self._config["LEVEL_WINDOW"]) * 3, 120)
        candles = self._fetch_candles(symbol, timeframe, limit)
        if len(candles) < 10:
            return []

        self._last_level_atr = _compute_atr(candles[-int(self._config["LEVEL_WINDOW"]) - 1 :])
        window = int(self._config["LEVEL_WINDOW"])
        left = int(self._config["PIVOT_LEFT"])
        right = int(self._config["PIVOT_RIGHT"])
        start_idx = max(left, len(candles) - window - right)
        end_idx = len(candles) - right
        current_price = float(candles[-1][4])
        tolerance = self._last_level_atr * 0.25 if self._last_level_atr else current_price * 0.001

        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]

        candidates: list[Level] = []
        for idx in range(start_idx, end_idx):
            ts = int(candles[idx][0])
            if idx - left < 0 or idx + right >= len(candles):
                continue
            high = highs[idx]
            low = lows[idx]
            if high >= max(highs[idx - left : idx]) and high >= max(highs[idx + 1 : idx + 1 + right]):
                if high > current_price * 0.98:
                    touches = sum(1 for h in highs[-window:] if abs(h - high) <= tolerance)
                    score = touches * 2 + (idx / len(candles))
                    candidates.append(Level(price=high, level_type="R", timestamp=ts, score=score))
            if low <= min(lows[idx - left : idx]) and low <= min(lows[idx + 1 : idx + 1 + right]):
                if low < current_price * 1.02:
                    touches = sum(1 for l in lows[-window:] if abs(l - low) <= tolerance)
                    score = touches * 2 + (idx / len(candles))
                    candidates.append(Level(price=low, level_type="S", timestamp=ts, score=score))

        dedup: dict[tuple[str, int], Level] = {}
        for lvl in candidates:
            key = (lvl.level_type, int(round(lvl.price / (tolerance or 1.0), 2)))
            if key not in dedup or lvl.score > dedup[key].score:
                dedup[key] = lvl

        levels = sorted(dedup.values(), key=lambda lv: (-lv.score, -lv.timestamp))
        active_keys = {
            (round(lvl.price, 8), lvl.level_type, "LONG" if lvl.level_type == "R" else "SHORT")
            for lvl in levels
        }
        to_remove = [key for key in self._level_attempts if key not in active_keys]
        for key in to_remove:
            self._level_attempts.pop(key, None)
        return levels

    # ------------------------------------------------------------------
    def _purge_cooldowns(self, now_ts: int) -> None:
        self._cooldowns = [cd for cd in self._cooldowns if cd.expires_ts > now_ts]

    # ------------------------------------------------------------------
    def register_trade_exit(
        self,
        *,
        level_price: float,
        direction: str,
        time_ms: int,
        reason: str = "sl",
    ) -> None:
        bar_ms = self._bar_ms or (_interval_to_minutes(self._exec_tf) * 60_000)
        cooldown_bars = int(self._config["COOLDOWN_BARS"])
        entry = CooldownEntry(
            level_price=level_price,
            direction=direction,
            start_ts=time_ms,
            expires_ts=time_ms + cooldown_bars * bar_ms,
            reason=reason,
        )
        self._cooldowns.append(entry)
        self._purge_cooldowns(time_ms)

    # ------------------------------------------------------------------
    def _is_in_cooldown(
        self,
        *,
        level: Level,
        direction: str,
        price: float,
        now_ts: int,
    ) -> bool:
        self._purge_cooldowns(now_ts)
        band_atr = self._config["COOLDOWN_BAND_ATR"] * (self._last_level_atr or 0.0)
        band_atr = band_atr or level.price * 0.001
        for cd in self._cooldowns:
            if cd.direction != direction:
                continue
            if abs(price - cd.level_price) <= band_atr and cd.expires_ts > now_ts:
                logger.info(
                    json.dumps(
                        {
                            "action": "reject",
                            "reason": "cooldown",
                            "level": level.price,
                            "direction": direction,
                            "band": band_atr,
                            "cooldown_expires": cd.expires_ts,
                        }
                    )
                )
                return True
        return False

    # ------------------------------------------------------------------
    def _register_attempt(self, level: Level, direction: str) -> None:
        key = (round(level.price, 8), level.level_type, direction)
        self._level_attempts[key] = self._level_attempts.get(key, 0) + 1

    # ------------------------------------------------------------------
    def _attempts(self, level: Level, direction: str) -> int:
        key = (round(level.price, 8), level.level_type, direction)
        return self._level_attempts.get(key, 0)

    # ------------------------------------------------------------------
    def _log_reject(self, reason: str, *, level: Level | None = None, data: dict[str, Any] | None = None) -> None:
        payload = {"action": "reject", "reason": reason}
        if level is not None:
            payload["level"] = {"price": level.price, "type": level.level_type}
        if data:
            payload.update(data)
        logger.info(json.dumps(payload))

    # ------------------------------------------------------------------
    def should_trigger_breakout(
        self,
        candle_exec: Sequence[float],
        levels: Sequence[Level],
        context: dict[str, Any],
    ) -> BreakoutSignalPayload | None:
        if not levels:
            return None

        exec_candles: Sequence[Sequence[float]] = context.get("exec_candles", [])
        if len(exec_candles) < 55:
            self._log_reject("not_enough_exec_candles", data={"len": len(exec_candles)})
            return None

        close = float(candle_exec[4])
        high = float(candle_exec[2])
        low = float(candle_exec[3])
        ts = int(candle_exec[0])
        atr_exec = _compute_atr(exec_candles)
        if atr_exec <= 0:
            self._log_reject("atr_unavailable")
            return None
        vol_rel = _compute_relative_volume(exec_candles)
        ema_fast = _ema([float(c[4]) for c in exec_candles], 15)
        ema_slow = _ema([float(c[4]) for c in exec_candles], 50)
        exec_tf = context.get("exec_tf", self._exec_tf)
        bar_ms = _interval_to_minutes(exec_tf) * 60_000
        self._bar_ms = bar_ms

        rr_min = float(self._config["RR_MIN"])
        vol_threshold = float(self._config["VOL_REL_MIN"])

        sorted_levels = sorted(levels, key=lambda lv: (-lv.score, abs(close - lv.price)))

        max_retries = int(self._config["MAX_RETRIES"])
        use_retest = bool(self._config.get("USE_RETEST", True))
        retest_tol = float(self._config["RETEST_TOL_ATR"])
        retest_timeout = int(self._config["RETEST_TIMEOUT"])

        for level in sorted_levels:
            direction = "LONG" if level.level_type == "R" else "SHORT"
            action = "BUY" if direction == "LONG" else "SELL"
            k_atr = float(self._config["K_ATR"])
            breakout_ok = close >= level.price + k_atr * atr_exec if direction == "LONG" else close <= level.price - k_atr * atr_exec
            ema_ok = ema_fast > ema_slow if direction == "LONG" else ema_fast < ema_slow

            if vol_rel < vol_threshold:
                self._log_reject(
                    "vol_rel",
                    level=level,
                    data={"vol_rel": vol_rel, "threshold": vol_threshold},
                )
                return None

            if not ema_ok:
                self._log_reject(
                    "ema_filter",
                    level=level,
                    data={"ema_fast": ema_fast, "ema_slow": ema_slow, "direction": direction},
                )
                return None

            if self._is_in_cooldown(level=level, direction=direction, price=close, now_ts=ts):
                return None

            attempts = self._attempts(level, direction)
            if attempts >= max_retries:
                self._log_reject(
                    "retry_limit",
                    level=level,
                    data={"attempts": attempts, "max": max_retries, "direction": direction},
                )
                return None

            key = (round(level.price, 8), level.level_type, direction)
            pending = self._pending_breakouts.get(key)

            if use_retest:
                if breakout_ok and pending is None:
                    pending = PendingBreakout(
                        level=level,
                        direction=direction,
                        breakout_ts=ts,
                        breakout_close=close,
                        atr=atr_exec,
                        exec_tf=exec_tf,
                        highest_high=high,
                        lowest_low=low,
                    )
                    self._pending_breakouts[key] = pending
                    logger.info(
                        json.dumps(
                            {
                                "action": "pending_retest",
                                "level": {"price": level.price, "type": level.level_type},
                                "direction": direction,
                                "k_atr": k_atr,
                                "atr": atr_exec,
                                "close": close,
                            }
                        )
                    )
                    continue

                if pending:
                    if pending.is_expired(ts, bar_ms, retest_timeout):
                        self._log_reject(
                            "retest_timeout",
                            level=level,
                            data={"timeout": retest_timeout, "exec_tf": exec_tf},
                        )
                        self._pending_breakouts.pop(key, None)
                        continue
                    if direction == "LONG":
                        pending.highest_high = max(pending.highest_high, high)
                        pending.lowest_low = min(pending.lowest_low, low)
                        if not pending.retest_seen and low <= level.price + retest_tol * atr_exec:
                            pending.retest_seen = True
                            pending.retest_ts = ts
                            logger.info(
                                json.dumps(
                                    {
                                        "action": "retest_detected",
                                        "level": {"price": level.price, "type": level.level_type},
                                        "direction": direction,
                                        "tolerance_atr": retest_tol,
                                    }
                                )
                            )
                        if pending.retest_seen and close >= max(level.price + k_atr * atr_exec, pending.breakout_close):
                            self._pending_breakouts.pop(key, None)
                        else:
                            continue
                    else:
                        pending.lowest_low = min(pending.lowest_low, low)
                        pending.highest_high = max(pending.highest_high, high)
                        if not pending.retest_seen and high >= level.price - retest_tol * atr_exec:
                            pending.retest_seen = True
                            pending.retest_ts = ts
                            logger.info(
                                json.dumps(
                                    {
                                        "action": "retest_detected",
                                        "level": {"price": level.price, "type": level.level_type},
                                        "direction": direction,
                                        "tolerance_atr": retest_tol,
                                    }
                                )
                            )
                        if pending.retest_seen and close <= min(level.price - k_atr * atr_exec, pending.breakout_close):
                            self._pending_breakouts.pop(key, None)
                        else:
                            continue
            else:
                if not breakout_ok:
                    self._log_reject(
                        "breakout_threshold",
                        level=level,
                        data={"close": close, "required": level.price + k_atr * atr_exec, "direction": direction},
                    )
                    continue

            swing = _find_recent_swing(exec_candles, direction)
            sl, tp1, tp2 = self._compute_sl_tp(direction, close, level, atr_exec, swing)
            if sl <= 0:
                self._log_reject("invalid_sl", level=level, data={"sl": sl})
                continue
            risk = abs(close - sl)
            reward = abs(tp2 - close)
            if risk <= 0:
                self._log_reject("risk_zero", level=level)
                continue
            rr = reward / risk
            if rr < rr_min:
                self._log_reject("rr_filter", level=level, data={"rr": rr, "min": rr_min})
                continue

            self._register_attempt(level, direction)
            payload = BreakoutSignalPayload(
                symbol=context.get("symbol", get_symbol(self._settings)),
                action=action,
                direction=direction,
                level=level,
                entry_price=close,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                rr=rr,
                atr=atr_exec,
                volume_rel=vol_rel,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                exec_tf=exec_tf,
                candle=candle_exec,
                swing=swing,
            )
            return payload

        return None

    # ------------------------------------------------------------------
    def _compute_sl_tp(
        self,
        direction: str,
        entry: float,
        level: Level,
        atr: float,
        swing: float,
    ) -> tuple[float, float, float]:
        stop_pct = get_stop_loss_pct(self._settings)
        tp_pct = get_take_profit_pct(self._settings)
        if direction == "LONG":
            sl = min(swing, level.price) - 0.5 * atr
            tp1 = entry + 1.0 * atr
            tp2 = entry + 2.0 * atr
            if stop_pct:
                sl = entry * (1 - stop_pct)
            if tp_pct:
                tp1 = entry * (1 + tp_pct)
                tp2 = entry * (1 + 2 * tp_pct)
        else:
            sl = max(swing, level.price) + 0.5 * atr
            tp1 = entry - 1.0 * atr
            tp2 = entry - 2.0 * atr
            if stop_pct:
                sl = entry * (1 + stop_pct)
            if tp_pct:
                tp1 = entry * (1 - tp_pct)
                tp2 = entry * (1 - 2 * tp_pct)
        return sl, tp1, tp2

    # ------------------------------------------------------------------
    def compute_orders(self, signal: BreakoutSignalPayload) -> dict[str, Any]:
        broker = self._broker
        symbol = signal.symbol
        entry = signal.entry_price
        sl = signal.sl
        tp1 = signal.tp1
        tp2 = signal.tp2
        qty_target_src = "NONE"
        filters: dict[str, Any] = {}
        tick = 0.0
        step = 0.0
        min_qty = 0.0
        min_notional = 0.0
        if broker is not None:
            try:
                filters = broker.get_symbol_filters(symbol)
            except Exception:  # pragma: no cover - adapter compatibility
                filters = {}
        if filters:
            lot = filters.get("LOT_SIZE", {})
            min_qty = float(lot.get("minQty", 0.0))
            step = float(lot.get("stepSize", 0.0))
            price_filter = filters.get("PRICE_FILTER", {})
            tick = float(price_filter.get("tickSize", 0.0))
            min_notional = float(
                filters.get("MIN_NOTIONAL", {}).get("notional")
                or filters.get("MIN_NOTIONAL", {}).get("minNotional", 0.0)
            )

        entry_rounded = broker.round_price_to_tick(symbol, entry) if broker and tick else _snap_price(entry, tick, side=signal.action)
        sl_rounded = broker.round_price_to_tick(symbol, sl) if broker and tick else _snap_price(sl, tick, side="SELL" if signal.direction == "LONG" else "BUY")
        tp1_rounded = broker.round_price_to_tick(symbol, tp1) if broker and tick else _snap_price(tp1, tick, side=signal.action)
        tp2_rounded = broker.round_price_to_tick(symbol, tp2) if broker and tick else _snap_price(tp2, tick, side=signal.action)

        risk_distance = abs(entry_rounded - sl_rounded)
        risk_notional = float(getattr(self._settings, "RISK_NOTIONAL_USDT", 0.0) or 0.0)
        qty_target = 0.0
        if risk_notional > 0 and risk_distance > 0:
            qty_target = risk_notional / risk_distance
            qty_target_src = "NOTIONAL_RISK"
        elif risk_notional > 0 and entry_rounded > 0:
            qty_target = risk_notional / entry_rounded
            qty_target_src = "NOTIONAL"
        else:
            balance = 0.0
            try:
                if broker is not None:
                    balance = float(broker.get_available_balance_usdt())
            except Exception:  # pragma: no cover - adapter compatibility
                balance = 0.0
            risk_pct = float(getattr(self._settings, "RISK_PCT", 0.0) or 0.0)
            if risk_pct > 0 and balance > 0 and risk_distance > 0:
                qty_target = (balance * risk_pct) / risk_distance
                qty_target_src = "PCT_RISK"

        if step > 0:
            qty_target = math.floor(qty_target / step) * step
        qty_target = max(qty_target, min_qty)
        if entry_rounded and entry_rounded * qty_target < min_notional:
            required_qty = min_notional / entry_rounded if entry_rounded else 0.0
            if step > 0:
                required_qty = math.ceil(required_qty / step) * step
            qty_target = max(qty_target, required_qty)

        orders = {
            "symbol": symbol,
            "side": signal.action,
            "entry": entry_rounded,
            "stop_loss": sl_rounded,
            "take_profit_1": tp1_rounded,
            "take_profit_2": tp2_rounded,
            "qty": qty_target,
            "rr": signal.rr,
            "breakeven_on_tp1": True,
            "qty_target_src": qty_target_src,
            "timeframe_exec": signal.exec_tf,
            "level": {"price": signal.level.price, "type": signal.level.level_type},
        }
        logger.info(
            json.dumps(
                {
                    "action": "signal",
                    "strategy": "breakout_dual_tf",
                    "orders": orders,
                    "atr": signal.atr,
                    "volume_rel": signal.volume_rel,
                    "ema_fast": signal.ema_fast,
                    "ema_slow": signal.ema_slow,
                }
            )
        )
        return orders

    # ------------------------------------------------------------------
    def generate_signal(self, now: datetime) -> Signal | None:
        symbol = get_symbol(self._settings)
        level_tf = getattr(self._settings, "INTERVAL", "1h")
        exec_tf = downscale_interval(level_tf)
        self._exec_tf = exec_tf
        self._bar_ms = _interval_to_minutes(exec_tf) * 60_000

        levels = self.get_levels(symbol, level_tf)
        if not levels:
            self._log_reject("no_levels")
            return None

        exec_limit = max(120, int(self._config["LEVEL_WINDOW"]) * 3)
        exec_candles = self._fetch_candles(symbol, exec_tf, exec_limit)
        if len(exec_candles) < 55:
            self._log_reject("not_enough_exec_candles", data={"len": len(exec_candles)})
            return None

        context = {
            "symbol": symbol,
            "exec_tf": exec_tf,
            "level_tf": level_tf,
            "now": now,
            "exec_candles": exec_candles,
        }
        payload = self.should_trigger_breakout(exec_candles[-1], levels, context)
        if payload is None:
            return None
        self._last_payload = payload
        return Signal(action=payload.action, price=payload.entry_price, time=now)

    # ------------------------------------------------------------------
    def run(
        self,
        exchange: BrokerPort | None = None,
        market_data: MarketDataPort | None = None,
        settings: SettingsProvider | None = None,
        now_utc: datetime | None = None,
        event: Any | None = None,
    ) -> dict[str, Any]:
        if exchange is not None:
            self._broker = exchange
        if market_data is not None:
            self._market_data = market_data
        if settings is not None:
            self._settings = settings

        now = now_utc or datetime.utcnow()
        signal = self.generate_signal(now)
        if signal is None or self._last_payload is None:
            return {"status": "no_signal", "strategy": "breakout_dual_tf"}
        orders = self.compute_orders(self._last_payload)
        return {
            "status": "signal",
            "strategy": "breakout_dual_tf",
            "signal": signal,
            "orders": orders,
        }


__all__ = ["BreakoutDualTFStrategy", "downscale_interval", "Level"]

