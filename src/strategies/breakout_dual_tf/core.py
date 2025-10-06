"""Dual timeframe breakout strategy implementation.

This module introduces a breakout strategy that analyses support and
resistance levels on a higher timeframe and executes trades on a lower
timeframe.  The implementation follows the requirements described in the
user story and keeps the logic self-contained so it can be easily reused by
other orchestration layers.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence
from uuid import uuid4

from common.symbols import normalize_symbol
from common.utils import sanitize_client_order_id
from config.settings import get_stop_loss_pct, get_take_profit_pct
from config.utils import parse_bool
from core.domain.models.Signal import Signal
from core.ports.broker import BrokerPort
from core.ports.market_data import MarketDataPort
from core.ports.settings import SettingsProvider, get_symbol
from core.ports.strategy import Strategy
from utils.tp_store_s3 import load_tp_value, persist_tp_value
from .filters.ema_distance import ema_distance_filter


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


def _fmt(value: Any | None) -> Any | None:
    """Round numeric values to 6 decimals for logging purposes."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return None
    return value


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
        "RR_FILTER_ENABLED": False,
        "RETEST_TOL_ATR": 0.2,
        "RETEST_TIMEOUT": 6,
        "USE_RETEST": False,
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

        level_window_source = "default"
        base_level_window = self._config.get("LEVEL_WINDOW", self.DEFAULT_CONFIG["LEVEL_WINDOW"])
        try:
            effective_level_window = int(base_level_window)
        except (TypeError, ValueError):
            effective_level_window = int(self.DEFAULT_CONFIG["LEVEL_WINDOW"])

        env_value: str | None = None
        if "LEVEL_WINDOW" in os.environ:
            env_value = os.environ["LEVEL_WINDOW"]
            level_window_source = "env:LEVEL_WINDOW"
        elif "LOOKBACK_BARS" in os.environ:
            env_value = os.environ["LOOKBACK_BARS"]
            level_window_source = "env:LOOKBACK_BARS"

        if env_value is not None:
            try:
                effective_level_window = int(env_value)
            except (TypeError, ValueError):
                effective_level_window = int(self._config.get("LEVEL_WINDOW", self.DEFAULT_CONFIG["LEVEL_WINDOW"]))
                level_window_source = "default"

        if effective_level_window < 10:
            effective_level_window = 10
        elif effective_level_window > 500:
            effective_level_window = 500

        self._config["LEVEL_WINDOW"] = effective_level_window
        logger.info(
            {
                "cfg": "LEVEL_WINDOW",
                "effective": self._config["LEVEL_WINDOW"],
                "source": level_window_source,
            }
        )

        default_rr_enabled = self.DEFAULT_CONFIG["RR_FILTER_ENABLED"]
        rr_source: Any = default_rr_enabled
        settings_rr_value = getattr(settings, "RR_FILTER_ENABLED", None)
        if settings_rr_value is None and hasattr(settings, "get"):
            settings_rr_value = settings.get("RR_FILTER_ENABLED", None)
        if settings_rr_value is not None:
            rr_source = settings_rr_value
        if config and "RR_FILTER_ENABLED" in config:
            rr_source = config["RR_FILTER_ENABLED"]
        rr_filter_enabled = parse_bool(rr_source, default=default_rr_enabled)
        self._config["RR_FILTER_ENABLED"] = rr_filter_enabled

        default_rr_min = float(self.DEFAULT_CONFIG["RR_MIN"])
        rr_min_source: Any = default_rr_min
        settings_rr_min = getattr(settings, "RR_MIN", None)
        if settings_rr_min is None and hasattr(settings, "get"):
            settings_rr_min = settings.get("RR_MIN", None)
        if settings_rr_min is not None:
            rr_min_source = settings_rr_min
        if config and "RR_MIN" in config:
            rr_min_source = config["RR_MIN"]
        try:
            rr_min_value = float(rr_min_source)
        except (TypeError, ValueError):
            rr_min_value = default_rr_min
        self._config["RR_MIN"] = rr_min_value

        use_retest_source: Any = self.DEFAULT_CONFIG["USE_RETEST"]
        settings_use_retest = getattr(settings, "USE_RETEST", None)
        if settings_use_retest is None and hasattr(settings, "get"):
            settings_use_retest = settings.get("USE_RETEST", None)
        if settings_use_retest is not None:
            use_retest_source = settings_use_retest
        else:
            env_raw_os = os.getenv("USE_RETEST")
            if env_raw_os is not None:
                use_retest_source = env_raw_os
        if config and "USE_RETEST" in config:
            use_retest_source = config["USE_RETEST"]
        use_retest = parse_bool(use_retest_source, default=self.DEFAULT_CONFIG["USE_RETEST"])
        self._config["USE_RETEST"] = use_retest
        self._use_retest = use_retest
        #logger.info(
        #    json.dumps(
        #        {
        #            "action": "config",
        #            "key": "USE_RETEST",
        #            "value": self._config["USE_RETEST"],
        #            "source": "env-dominate",
        #        }
        #    )
        #)
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
    def _has_active_position_or_orders(
        self, symbol: str, side: str | None = None
    ) -> dict[str, Any] | None:
        """Return skip payload when there's an active position/order."""

        sides_to_check: tuple[str, ...]
        if side is None:
            sides_to_check = ("BUY", "SELL")
        else:
            sides_to_check = (side,)

        summary_total: dict[str, Any] = {
            "has_position": False,
            "open_orders_total": 0,
            "tp_reduce_only": False,
            "tp_restore_attempted": False,
            "tp_restore_success": False,
            "tp_expected_stop": None,
            "sides": [],
        }

        payload: dict[str, Any] | None = None
        for candidate in sides_to_check:
            payload_candidate = self._has_active_position_or_orders_single(
                symbol, candidate
            )
            side_summary = dict(
                getattr(self, "_poscheck_last_summary_single", {})
            )
            if not side_summary:
                side_summary = {
                    "side": str(candidate),
                    "has_position": False,
                    "open_orders_total": 0,
                    "tp_reduce_only": False,
                    "tp_restore_attempted": False,
                    "tp_restore_success": False,
                    "tp_expected_stop": None,
                }
            summary_total["sides"].append(side_summary)
            summary_total["has_position"] = (
                summary_total["has_position"] or side_summary.get("has_position", False)
            )
            summary_total["open_orders_total"] += side_summary.get("open_orders_total", 0)
            summary_total["tp_reduce_only"] = (
                summary_total["tp_reduce_only"]
                or side_summary.get("tp_reduce_only", False)
            )
            summary_total["tp_restore_attempted"] = (
                summary_total["tp_restore_attempted"]
                or side_summary.get("tp_restore_attempted", False)
            )
            summary_total["tp_restore_success"] = (
                summary_total["tp_restore_success"]
                or side_summary.get("tp_restore_success", False)
            )
            if (
                summary_total["tp_expected_stop"] is None
                and side_summary.get("tp_expected_stop") is not None
            ):
                summary_total["tp_expected_stop"] = side_summary.get("tp_expected_stop")
            if payload_candidate is not None and payload is None:
                payload = payload_candidate
                break

        setattr(self, "_poscheck_last_summary", summary_total)
        return payload

    # ------------------------------------------------------------------
    def _has_active_position_or_orders_single(
        self, symbol: str, side: str
    ) -> dict[str, Any] | None:
        """Return skip payload when there's an active position/order for a side."""

        summary: dict[str, Any] = {
            "side": "",
            "has_position": False,
            "open_orders_total": 0,
            "tp_reduce_only": False,
            "tp_restore_attempted": False,
            "tp_restore_success": False,
            "tp_expected_stop": None,
        }

        def _finalize(payload: dict[str, Any] | None) -> dict[str, Any] | None:
            setattr(self, "_poscheck_last_summary_single", summary.copy())
            return payload

        exchange = self._broker
        if exchange is None:
            return _finalize(None)

        if not hasattr(self, "_logger"):
            self._logger = logger

        settings_obj: SettingsProvider | None = getattr(self, "_settings", None)
        testnet = None
        trading_mode = None
        if settings_obj is not None:
            getter = getattr(settings_obj, "get", None)
            if callable(getter):
                testnet = getter("BINANCE_TESTNET", None)
                trading_mode = getter("TRADING_MODE", None)
            else:
                testnet = getattr(settings_obj, "BINANCE_TESTNET", None)
                trading_mode = getattr(settings_obj, "TRADING_MODE", None)

        self._logger.info(
            "bdtf.poscheck.entry %s",
            {
                "symbol_in": symbol,
                "side_in": side,
                "testnet": testnet,
                "trading_mode": trading_mode,
            },
        )

        norm_symbol = normalize_symbol(symbol)
        broker_symbol = norm_symbol
        if hasattr(exchange, "normalize_symbol"):
            try:
                broker_symbol = exchange.normalize_symbol(norm_symbol)
            except Exception:  # pragma: no cover - best effort fallback
                broker_symbol = norm_symbol

        self._logger.info("bdtf.poscheck.symbol %s", {"symbol_norm": broker_symbol})

        side_norm = str(side or "").strip().upper()
        side_norm = {"LONG": "BUY", "SHORT": "SELL"}.get(side_norm, side_norm)
        if side_norm not in {"BUY", "SELL"}:
            side_norm = "SELL" if side_norm.startswith("S") else "BUY"
        summary["side"] = side_norm

        def _coerce_bool(value: Any) -> bool:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "t", "yes", "y"}
            return bool(value)

        position_amt = 0.0
        position_side = None
        position_side_raw: str | None = None
        position_entry_price: float | None = None
        matched_entry_orders = 0
        has_position = False
        open_orders_total = 0
        open_orders_list: list[Any] | None = None

        def _ensure_open_orders() -> list[Any]:
            nonlocal open_orders_list
            if open_orders_list is None:
                open_orders_list = self._get_open_orders(broker_symbol, side_norm)
            return open_orders_list

        def tp_close_all_state(
            *,
            position_qty: float,
            position_side_value: str | None,
            entry_price: float | None,
            orders: list[Any] | None,
        ) -> None:
            pos_side_clean = str(position_side_value or "").upper()
            if math.isclose(position_qty, 0.0, abs_tol=1e-12) or not pos_side_clean:
                self._logger.info(
                    "tp_close_all_skip_no_position %s",
                    {"symbol": broker_symbol},
                )
                return

            target_side = pos_side_clean
            if target_side in {"BUY", "SELL"}:
                target_side = "LONG" if target_side == "BUY" else "SHORT"

            found_tp_close = False
            for order in orders or []:
                if not isinstance(order, dict):
                    continue
                order_type = str(
                    order.get("type")
                    or order.get("orderType")
                    or order.get("origType")
                    or ""
                ).upper()
                if order_type != "TAKE_PROFIT_MARKET":
                    continue
                order_pos_side = str(order.get("positionSide") or "").upper()
                if order_pos_side in {"BUY", "SELL"}:
                    order_pos_side = "LONG" if order_pos_side == "BUY" else "SHORT"
                if order_pos_side and target_side and order_pos_side != target_side:
                    continue
                close_position_flag = _coerce_bool(order.get("closePosition"))
                reduce_only_flag = _coerce_bool(order.get("reduceOnly")) or _coerce_bool(
                    order.get("reduce_only")
                )
                if not (close_position_flag and reduce_only_flag):
                    continue
                found_tp_close = True
                break

            if found_tp_close:
                summary["tp_reduce_only"] = True
                self._logger.info(
                    "tp_close_all_exists %s",
                    {"symbol": broker_symbol, "positionSide": target_side},
                )
                return

            expected_stop_price: float | None = None
            if entry_price is not None:
                try:
                    entry_val = float(entry_price)
                except (TypeError, ValueError):
                    entry_val = None
                if entry_val and entry_val > 0:
                    if target_side == "LONG":
                        expected = entry_val * (1 + 0.0032)
                        rounding_side = "SELL"
                    else:
                        expected = entry_val * (1 - 0.0032)
                        rounding_side = "BUY"
                    tick_size = 0.0
                    try:
                        context = self._build_order_context(exchange, broker_symbol)
                    except Exception:  # pragma: no cover - best effort
                        context = None
                    if isinstance(context, dict):
                        tick_size = float(context.get("tick_size", 0.0) or 0.0)
                    if tick_size:
                        expected = _snap_price(expected, tick_size, side=rounding_side)
                    else:
                        expected = round(expected, 8)
                    expected_stop_price = expected
            if expected_stop_price is not None:
                summary["tp_expected_stop"] = expected_stop_price

            self._logger.info(
                "tp_close_all_missing %s",
                {
                    "symbol": broker_symbol,
                    "positionSide": target_side,
                    "entry_price": entry_price,
                    "expected_stopPrice": expected_stop_price,
                    "note": "no tp_close_all detected",
                },
            )

            tp_loaded = load_tp_value(broker_symbol)
            if tp_loaded is None:
                self._logger.info(
                    "tp_s3_load.skip %s",
                    {
                        "symbol": broker_symbol,
                        "positionSide": target_side,
                        "reason": "tp_value_missing",
                    },
                )
                return

            if not hasattr(exchange, "place_tp_reduce_only"):
                self._logger.info(
                    "tp_s3_load.skip %s",
                    {
                        "symbol": broker_symbol,
                        "positionSide": target_side,
                        "reason": "place_tp_reduce_only_unavailable",
                    },
                )
                return

            qty_abs = abs(position_qty)
            if qty_abs <= 0:
                self._logger.info(
                    "tp_s3_load.skip %s",
                    {
                        "symbol": broker_symbol,
                        "positionSide": target_side,
                        "reason": "invalid_quantity",
                        "qty": qty_abs,
                    },
                )
                return

            try:
                qty_final = float(qty_abs)
            except (TypeError, ValueError):
                qty_final = 0.0

            if hasattr(exchange, "round_qty_to_step"):
                try:
                    qty_final = float(exchange.round_qty_to_step(broker_symbol, qty_final))
                except Exception:  # pragma: no cover - best effort rounding
                    pass

            if qty_final <= 0:
                self._logger.info(
                    "tp_s3_load.skip %s",
                    {
                        "symbol": broker_symbol,
                        "positionSide": target_side,
                        "reason": "qty_rounding_zero",
                    },
                )
                return

            tp_price = tp_loaded
            if hasattr(exchange, "round_price_to_tick"):
                try:
                    tp_price = float(exchange.round_price_to_tick(broker_symbol, tp_price))
                except Exception:  # pragma: no cover - best effort rounding
                    pass

            if tp_price <= 0:
                self._logger.info(
                    "tp_s3_load.skip %s",
                    {
                        "symbol": broker_symbol,
                        "positionSide": target_side,
                        "reason": "tp_price_invalid",
                        "value": tp_price,
                    },
                )
                return

            exit_side_order = "SELL" if target_side == "LONG" else "BUY"
            tp_cid = sanitize_client_order_id(f"bdtf-{uuid4().hex[:8]}-tp")

            summary["tp_restore_attempted"] = True
            try:
                order_resp = exchange.place_tp_reduce_only(
                    broker_symbol,
                    exit_side_order,
                    tp_price,
                    qty_final,
                    tp_cid,
                )
            except Exception as err:  # pragma: no cover - defensive logging
                self._logger.warning(
                    "tp_s3_load.error %s",
                    {
                        "symbol": broker_symbol,
                        "positionSide": target_side,
                        "reason": "order_failed",
                        "error": str(err),
                    },
                )
            else:
                summary["tp_restore_success"] = True
                self._logger.info(
                    "tp_s3_load.restore %s",
                    {
                        "symbol": broker_symbol,
                        "positionSide": target_side,
                        "tp_price": tp_price,
                        "qty": qty_final,
                        "clientOrderId": tp_cid,
                        "order": order_resp,
                    },
                )
        try:
            info: Any | None
            if hasattr(exchange, "get_position"):
                info = exchange.get_position(broker_symbol)
            elif hasattr(exchange, "futures_position_information"):
                info = exchange.futures_position_information(broker_symbol)
            elif hasattr(exchange, "position_information"):
                info = exchange.position_information(broker_symbol)
            else:
                info = None

            if info:
                entries: Iterable[dict[str, Any]]
                if isinstance(info, dict):
                    entries = [info]
                else:
                    entries = list(info)

                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    entry_symbol = normalize_symbol(str(entry.get("symbol", broker_symbol)))
                    if entry_symbol and entry_symbol != broker_symbol:
                        continue
                    raw_amt = (
                        entry.get("positionAmt")
                        or entry.get("position_amt")
                        or entry.get("position_amount")
                        or entry.get("qty")
                        or entry.get("quantity")
                    )
                    try:
                        amt = float(raw_amt)
                    except (TypeError, ValueError):
                        continue
                    if math.isclose(amt, 0.0, abs_tol=1e-12):
                        continue
                    raw_entry_price = (
                        entry.get("entryPrice")
                        or entry.get("avgEntryPrice")
                        or entry.get("avgPrice")
                        or entry.get("price")
                    )
                    try:
                        entry_price = float(raw_entry_price)
                    except (TypeError, ValueError):
                        entry_price = None
                    entry_side_raw = str(entry.get("positionSide", "")).upper()
                    if entry_side_raw in {"LONG", "SHORT"}:
                        entry_side = "BUY" if entry_side_raw == "LONG" else "SELL"
                    elif entry_side_raw == "BOTH":
                        entry_side = "BUY" if amt > 0 else "SELL"
                    elif entry_side_raw:
                        entry_side = entry_side_raw
                        if entry_side not in {"BUY", "SELL"}:
                            entry_side = "BUY" if amt > 0 else "SELL"
                    else:
                        entry_side = "BUY" if amt > 0 else "SELL"

                    position_amt = amt
                    position_side = entry_side
                    position_side_raw = entry_side_raw or entry_side
                    position_entry_price = entry_price
                    break
        except Exception as err:  # pragma: no cover - defensive
            self._logger.warning(
                "positioncheck.error %s",
                {
                    "strategy": "breakout_dual_tf",
                    "symbol": broker_symbol,
                    "side": side_norm,
                    "error": str(err),
                },
            )

        self._logger.info(
            "bdtf.poscheck.position %s",
            {
                "position_amt": position_amt,
                "entryPrice": position_entry_price,
            },
        )

        if position_side is not None and not math.isclose(position_amt, 0.0, abs_tol=1e-12):
            orders_for_tp = _ensure_open_orders()
            summary["open_orders_total"] = len(orders_for_tp or [])
            tp_close_all_state(
                position_qty=position_amt,
                position_side_value=position_side_raw or position_side,
                entry_price=position_entry_price,
                orders=orders_for_tp,
            )
            has_position = True
            summary["has_position"] = True
            payload = {
                "status": "skipped_existing_position",
                "strategy": "breakout_dual_tf",
                "symbol": broker_symbol,
                "side": position_side,
                "position_amt": position_amt,
                "positionAmt": position_amt,
            }
            if position_side_raw and position_side_raw != position_side:
                payload["positionSide"] = position_side_raw
            if side_norm != position_side:
                payload["requested_side"] = side_norm
            self._logger.info(json.dumps(payload))
            self._logger.info(
                "bdtf.poscheck.summary %s",
                {
                    "open_orders": open_orders_total,
                    "matched_entries": matched_entry_orders,
                    "position_amt": position_amt,
                    "result": "skip_position",
                },
            )
            return _finalize(payload)

        tp_close_all_state(
            position_qty=position_amt,
            position_side_value=position_side_raw or position_side,
            entry_price=position_entry_price,
            orders=None,
        )

        open_orders_list = _ensure_open_orders()

        active_statuses = {"NEW", "PARTIALLY_FILLED", "PENDING_NEW"}
        working_orders: list[Any] = []
        for order in open_orders_list or []:
            if not isinstance(order, dict):
                continue
            open_orders_total += 1
            order_id_val = (
                order.get("orderId")
                or order.get("order_id")
                or order.get("id")
                or ""
            )
            order_id = str(order_id_val) if order_id_val is not None else ""
            client_id = ""
            for key in (
                "clientOrderId",
                "client_order_id",
                "origClientOrderId",
                "orig_client_order_id",
                "clientId",
                "client_id",
            ):
                value = order.get(key)
                if value:
                    client_id = str(value)
                    break
            status = str(
                order.get("status")
                or order.get("orderStatus")
                or order.get("state")
                or ""
            ).upper()
            order_side = str(order.get("side") or "").upper()
            order_side = {"LONG": "BUY", "SHORT": "SELL"}.get(order_side, order_side)
            if not order_side and order.get("positionSide"):
                pos_side = str(order.get("positionSide") or "").upper()
                order_side = "BUY" if pos_side == "LONG" else "SELL" if pos_side == "SHORT" else order_side
            reduce_only_flag = _coerce_bool(order.get("reduceOnly")) or _coerce_bool(
                order.get("reduce_only")
            )
            close_position_flag = _coerce_bool(order.get("closePosition"))
            effective_reduce_only = reduce_only_flag or close_position_flag
            qty_val = (
                order.get("origQty")
                or order.get("orig_qty")
                or order.get("quantity")
                or order.get("qty")
            )
            self._logger.info(
                "bdtf.poscheck.order %s",
                {
                    "id": order_id,
                    "clientId": client_id,
                    "side": order_side,
                    "status": status,
                    "price": order.get("price"),
                    "qty": qty_val,
                    "reduceOnly": effective_reduce_only,
                },
            )
            if status not in active_statuses:
                self._logger.info(
                    "bdtf.poscheck.filter %s",
                    {"filter": "status_invalid", "orderId": order_id},
                )
                continue
            order_symbol_raw = (
                order.get("symbol")
                or order.get("symbolNormalized")
                or order.get("s")
                or ""
            )
            order_symbol_norm = (
                normalize_symbol(str(order_symbol_raw)) if order_symbol_raw else ""
            )
            if order_symbol_norm and order_symbol_norm != broker_symbol:
                self._logger.info(
                    "bdtf.poscheck.filter %s",
                    {"filter": "symbol_mismatch", "orderId": order_id},
                )
                continue
            if effective_reduce_only:
                self._logger.info(
                    "bdtf.poscheck.filter %s",
                    {"filter": "reduce_only", "orderId": order_id},
                )
                continue

            working_orders.append(order)

        matched_entry_orders = len(working_orders)
        if working_orders:
            summary["open_orders_total"] = open_orders_total
            payload = {
                "status": "skipped_existing_entry_orders",
                "strategy": "breakout_dual_tf",
                "symbol": broker_symbol,
                "count": len(working_orders),
            }
            payload["client_order_ids"] = [
                str(
                    order.get("clientOrderId")
                    or order.get("client_order_id")
                    or order.get("origClientOrderId")
                    or order.get("orig_client_order_id")
                    or order.get("clientId")
                    or order.get("client_id")
                    or ""
                )
                for order in working_orders
            ]
            self._logger.info(json.dumps(payload))
            self._logger.info(
                "bdtf.poscheck.summary %s",
                {
                    "open_orders": open_orders_total,
                    "matched_entries": matched_entry_orders,
                    "position_amt": position_amt,
                    "result": "skip_entry_orders",
                },
            )
            return _finalize(payload)

        self._logger.info(
            "bdtf.poscheck.summary %s",
            {
                "open_orders": open_orders_total,
                "matched_entries": matched_entry_orders,
                "position_amt": position_amt,
                "result": "none",
            },
        )
        summary["open_orders_total"] = open_orders_total
        return _finalize(None)

    def _get_open_orders(
        self, symbol: str, side_norm: str
    ) -> list[Any]:  # pragma: no cover - integration logic
        exchange = self._broker
        if exchange is None:
            return []

        cache_key = (symbol, side_norm)
        cache: dict[tuple[str, str], tuple[float, list[Any]]] | None = getattr(
            self, "_order_check_cache", None
        )
        now_ts = datetime.utcnow().timestamp()
        if cache:
            cached_entry = cache.get(cache_key)
            if cached_entry:
                cached_ts, cached_orders = cached_entry
                if now_ts - cached_ts <= 1.0:
                    self._logger.info(
                        "bdtf.poscheck.cache %s",
                        {"source": "reuse", "symbol": symbol, "side": side_norm},
                    )
                    return list(cached_orders)

        open_orders: Iterable[Any] = []
        fetch_methods = (
            "open_orders_perpetual",
            "futures_open_orders",
            "futures_get_open_orders",
            "open_orders",
        )
        for attr in fetch_methods:
            method = getattr(exchange, attr, None)
            if not callable(method):
                continue
            try:
                open_orders = method(symbol)
            except TypeError:
                open_orders = method(symbol=symbol)
            except Exception as err:  # pragma: no cover - defensive
                self._logger.warning(
                    "ordercheck.error %s",
                    {
                        "strategy": "breakout_dual_tf",
                        "symbol": symbol,
                        "side": side_norm,
                        "endpoint": attr,
                        "error": str(err),
                    },
                )
                continue
            else:
                break
        else:
            open_orders = []

        open_orders_list = list(open_orders or [])
        cache = cache or {}
        cache[cache_key] = (now_ts, list(open_orders_list))
        setattr(self, "_order_check_cache", cache)
        return open_orders_list

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
        logger.info({"lookback_effective_dual_tf": self._config["LEVEL_WINDOW"]})
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
        exec_closes = [float(c[4]) for c in exec_candles]
        ema_fast = _ema(exec_closes, 15)
        ema_slow = _ema(exec_closes, 50)
        ema7_exec = context.get("ema7_exec")
        ema25_exec = context.get("ema25_exec")
        if ema7_exec is None:
            ema7_exec = _ema(exec_closes, 7)
            context["ema7_exec"] = ema7_exec
        if ema25_exec is None:
            ema25_exec = _ema(exec_closes, 25)
            context["ema25_exec"] = ema25_exec
        exec_tf = context.get("exec_tf", self._exec_tf)
        bar_ms = _interval_to_minutes(exec_tf) * 60_000
        self._bar_ms = bar_ms

        rr_min = float(self._config["RR_MIN"])
        vol_threshold = float(self._config["VOL_REL_MIN"])

        sorted_levels = sorted(levels, key=lambda lv: (-lv.score, abs(close - lv.price)))

        max_retries = int(self._config["MAX_RETRIES"])
        use_retest = bool(self._config.get("USE_RETEST", False))
        retest_tol = float(self._config["RETEST_TOL_ATR"])
        retest_timeout = int(self._config["RETEST_TIMEOUT"])

        for level in sorted_levels:
            direction = "SHORT" if level.level_type == "R" else "LONG"
            action = "SELL" if direction == "SHORT" else "BUY"
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
            if self._config.get("RR_FILTER_ENABLED", False):
                if rr < rr_min:
                    tp_pct_value = get_take_profit_pct(self._settings)
                    if tp_pct_value:
                        tp_mode = "pct"
                    else:
                        tp_mode = "atr"
                    payload = {
                        "action": "reject",
                        "reason": "rr_filter",
                        "level": {"price": _fmt(level.price), "type": level.level_type},
                        "rr": _fmt(rr),
                        "min": _fmt(rr_min),
                        "side": direction,
                        "entry_price": _fmt(close),
                        "sl": _fmt(level.price),
                        "tp1": _fmt(tp1),
                        "tp2": _fmt(tp2),
                        "tp_mode": tp_mode if tp_mode else "unknown",
                    }
                    logger.info(json.dumps(payload))
                    continue
            else:
                logger.info(json.dumps({"action": "skip", "reason": "rr_filter_disabled"}))

            ema_filter_enabled = (
                os.getenv("EMA_DISTANCE_FILTER_ENABLED", "true").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            if ema_filter_enabled:
                filter_ctx = {
                    "ohlc": {
                        "open": float(candle_exec[1]),
                        "high": float(candle_exec[2]),
                        "low": float(candle_exec[3]),
                        "close": close,
                    },
                    "ema7": ema7_exec,
                    "ema25": ema25_exec,
                    "candles": exec_candles,
                    "direction": direction,
                    "atr": atr_exec,
                    "exec_tf": exec_tf,
                    "use_wick": True,
                }
                passed, meta = ema_distance_filter(filter_ctx)
                if not passed:
                    result = meta.get("result")
                    dist = meta.get("dist") or (result.dist_to_nearest_pct if result else 0.0)
                    max_allowed = meta.get("max_allowed")
                    tf_label = meta.get("tf_label")
                    policy = meta.get("policy")
                    nearest = meta.get("nearest")
                    if meta.get("reattach_blocked") and meta.get("touches") is not None:
                        logger.info(
                            "[EmaDistance] skip: dist=%s max=%s tf=%s reattach: touches=%s/%s lookback=%s",
                            f"{dist:.4f}" if isinstance(dist, float) else dist,
                            f"{max_allowed:.4f}" if isinstance(max_allowed, float) else max_allowed,
                            tf_label,
                            meta.get("touches"),
                            meta.get("min_touches"),
                            meta.get("lookback"),
                        )
                    else:
                        logger.info(
                            "[EmaDistance] skip: dist=%s max=%s tf=%s policy=%s nearest=%s",
                            f"{dist:.4f}" if isinstance(dist, float) else dist,
                            f"{max_allowed:.4f}" if isinstance(max_allowed, float) else max_allowed,
                            tf_label,
                            policy,
                            nearest,
                        )
                    continue
                else:
                    result = meta.get("result")
                    dist = meta.get("dist") or (result.dist_to_nearest_pct if result else 0.0)
                    max_allowed = meta.get("max_allowed")
                    tf_label = meta.get("tf_label")
                    policy = meta.get("policy")
                    nearest = meta.get("nearest")
                    if meta.get("reattach_pass") and meta.get("touches") is not None:
                        logger.info(
                            "[EmaDistance] reattach: touches=%s/%s -> pass",
                            meta.get("touches"),
                            meta.get("min_touches"),
                        )
                    else:
                        logger.info(
                            "[EmaDistance] pass: dist=%s max=%s tf=%s policy=%s nearest=%s use_wick=%s",
                            f"{dist:.4f}" if isinstance(dist, float) else dist,
                            f"{max_allowed:.4f}" if isinstance(max_allowed, float) else max_allowed,
                            tf_label,
                            policy,
                            nearest,
                            True,
                        )

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
    def _build_order_context(
        self, exchange: BrokerPort | None, symbol: str
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "tick_size": 0.0,
            "step_size": 0.0,
            "min_qty": 0.0,
            "min_notional": 0.0,
            "available_balance_usdt": 0.0,
        }
        if exchange is None:
            return context

        filters: dict[str, Any] | None = None
        if hasattr(exchange, "get_symbol_filters"):
            try:
                filters = exchange.get_symbol_filters(symbol)
            except Exception as err:  # pragma: no cover - defensive logging
                logger.warning(
                    "order_context.filters_error %s",
                    {
                        "strategy": "breakout_dual_tf",
                        "symbol": symbol,
                        "error": str(err),
                    },
                )
                filters = None

        if isinstance(filters, dict):
            lot = filters.get("LOT_SIZE", {}) or {}
            context["min_qty"] = float(lot.get("minQty", lot.get("min_qty", 0.0)) or 0.0)
            context["step_size"] = float(lot.get("stepSize", lot.get("step_size", 0.0)) or 0.0)

            price_filter = filters.get("PRICE_FILTER", {}) or {}
            context["tick_size"] = float(
                price_filter.get("tickSize", price_filter.get("tick_size", 0.0)) or 0.0
            )

            min_notional_filter = filters.get("MIN_NOTIONAL", {}) or {}
            context["min_notional"] = float(
                min_notional_filter.get("notional")
                or min_notional_filter.get("minNotional")
                or min_notional_filter.get("min_notional", 0.0)
                or 0.0
            )

        if hasattr(exchange, "get_available_balance_usdt"):
            try:
                context["available_balance_usdt"] = float(
                    exchange.get_available_balance_usdt()
                )
            except Exception as err:  # pragma: no cover - defensive logging
                logger.warning(
                    "order_context.balance_error %s",
                    {
                        "strategy": "breakout_dual_tf",
                        "symbol": symbol,
                        "error": str(err),
                    },
                )

        return context

    # ------------------------------------------------------------------
    def compute_orders(
        self,
        signal: BreakoutSignalPayload,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        symbol = signal.symbol
        entry = signal.entry_price
        sl = signal.sl
        tp1 = signal.tp1
        tp2 = signal.tp2
        qty_target_src = "NONE"

        ctx = context or {}
        tick = float(ctx.get("tick_size", 0.0) or 0.0)
        step = float(ctx.get("step_size", 0.0) or 0.0)
        min_qty = float(ctx.get("min_qty", 0.0) or 0.0)
        min_notional = float(ctx.get("min_notional", 0.0) or 0.0)
        balance_usdt = float(ctx.get("available_balance_usdt", 0.0) or 0.0)

        entry_rounded = _snap_price(entry, tick, side=signal.action)
        sl_side = "SELL" if signal.direction == "LONG" else "BUY"
        sl_rounded = _snap_price(sl, tick, side=sl_side)
        tp1_rounded = _snap_price(tp1, tick, side=signal.action)
        tp2_rounded = _snap_price(tp2, tick, side=signal.action)

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
            risk_pct = float(getattr(self._settings, "RISK_PCT", 0.0) or 0.0)
            if risk_pct > 0 and balance_usdt > 0 and risk_distance > 0:
                qty_target = (balance_usdt * risk_pct) / risk_distance
                qty_target_src = "PCT_RISK"

        if step > 0:
            qty_target = math.floor(qty_target / step) * step
        qty_target = max(qty_target, min_qty)
        if entry_rounded and entry_rounded * qty_target < min_notional:
            required_qty = min_notional / entry_rounded if entry_rounded else 0.0
            if step > 0:
                required_qty = math.ceil(required_qty / step) * step
            qty_target = max(qty_target, required_qty)

        qty_tp1 = 0.0
        qty_tp2 = 0.0
        if qty_target > 0:
            if step > 0:
                steps_total = max(int(round(qty_target / step)), 1)
                steps_tp1 = steps_total // 2
                qty_tp1 = steps_tp1 * step
                qty_tp1 = min(qty_tp1, qty_target)
                qty_tp2 = qty_target - qty_tp1
            else:
                qty_tp1 = qty_target / 2
                qty_tp2 = qty_target - qty_tp1
            qty_tp1 = round(qty_tp1, 10)
            qty_tp2 = round(qty_tp2, 10)

        orders = {
            "symbol": symbol,
            "side": signal.action,
            "entry": entry_rounded,
            "stop_loss": sl_rounded,
            "take_profit_1": tp1_rounded,
            "take_profit_2": tp2_rounded,
            "qty": qty_target,
            "qty_tp1": qty_tp1,
            "qty_tp2": qty_tp2,
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
    def place_orders(
        self,
        orders: dict[str, Any],
        *,
        exchange: BrokerPort | None = None,
    ) -> dict[str, Any]:
        broker = exchange or self._broker
        if broker is None:
            raise ValueError("Exchange not configured for order placement")

        symbol = str(orders.get("symbol") or get_symbol(self._settings))
        side = str(orders.get("side", "")).upper() or "BUY"
        qty = float(orders.get("qty", 0.0) or 0.0)
        entry_price = float(orders.get("entry", 0.0) or 0.0)
        stop_loss = float(orders.get("stop_loss", 0.0) or 0.0)
        exit_side = "SELL" if side == "BUY" else "BUY"

        trade_tag = f"bdtf-{uuid4().hex[:8]}"
        entry_cid = sanitize_client_order_id(f"{trade_tag}-entry")
        sl_cid = sanitize_client_order_id(f"{trade_tag}-sl")

        try:
            entry_resp = broker.place_entry_limit(
                symbol,
                side,
                entry_price,
                qty,
                entry_cid,
                timeInForce="GTC",
            )
        except TypeError:  # pragma: no cover - legacy brokers without timeInForce
            entry_resp = broker.place_entry_limit(symbol, side, entry_price, qty, entry_cid)

        stop_resp = broker.place_stop_reduce_only(
            symbol,
            exit_side,
            stop_loss,
            qty,
            sl_cid,
        )

        logger.debug("tp_atr_call_removed_from_place_orders")

        tp_close_price = 0.0
        try:
            tp_close_price = float(orders.get("take_profit_2", 0.0) or 0.0)
        except (TypeError, ValueError):  # pragma: no cover - defensive conversion
            tp_close_price = 0.0
        if tp_close_price > 0:
            persist_tp_value(
                symbol,
                tp_close_price,
                int(datetime.utcnow().timestamp()),
            )
        else:
            logger.info(
                "tp_s3_persist.skip %s",
                {"symbol": symbol, "reason": "invalid_tp_value", "value": tp_close_price},
            )

        logger.info(
            "tp_creation_skipped_breakout_dual_tf",
            extra={
                "symbol": symbol,
                "position_side": side,
                "reason": "disabled_for_flow",
            },
        )

        placement = {
            "status": "orders_placed",
            "symbol": symbol,
            "side": side,
            "entry": {"clientOrderId": entry_cid, "order": entry_resp},
            "stop": {"clientOrderId": sl_cid, "order": stop_resp},
            "take_profits": [],
        }
        return placement

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
        active_interval = getattr(self._settings, "INTERVAL", "1h")
        self._exec_tf = downscale_interval(active_interval)
        logger.info(
            "active_config { INTERVAL=%s, timeframe_exec=%s, RR_FILTER_ENABLED=%s, RR_MIN=%s }",
            active_interval,
            self._exec_tf,
            self._config.get("RR_FILTER_ENABLED"),
            self._config.get("RR_MIN"),
        )

        now = now_utc or datetime.utcnow()
        symbol = get_symbol(self._settings)
        skip_payload = self._has_active_position_or_orders(symbol, side=None)
        summary_pre = getattr(self, "_poscheck_last_summary", None)
        guard_log: dict[str, Any] = {
            "strategy": "breakout_dual_tf",
            "symbol": symbol,
            "result": "skip" if skip_payload is not None else "continue",
        }
        if isinstance(summary_pre, dict):
            guard_log.update(
                {
                    "has_position": summary_pre.get("has_position"),
                    "open_orders_total": summary_pre.get("open_orders_total"),
                    "tp_reduce_only": summary_pre.get("tp_reduce_only"),
                    "tp_restore_attempted": summary_pre.get("tp_restore_attempted"),
                    "tp_restore_success": summary_pre.get("tp_restore_success"),
                    "tp_expected_stop": summary_pre.get("tp_expected_stop"),
                }
            )
        logger.info("bdtf.poscheck.guard %s", guard_log)
        if skip_payload is not None:
            return skip_payload

        signal = self.generate_signal(now)

        if signal is None or self._last_payload is None:
            return {"status": "no_signal", "strategy": "breakout_dual_tf"}

        side = str(signal.action).upper()

        skip_payload = self._has_active_position_or_orders(symbol, side)
        if skip_payload is not None:
            return skip_payload

        logger.info(
            "pre-compute guard %s",
            {"strategy": "breakout_dual_tf", "symbol": symbol, "side": side},
        )

        broker = self._broker
        order_context = self._build_order_context(broker, symbol)
        orders = self.compute_orders(self._last_payload, context=order_context)

        logger.info(
            "post-compute guard %s",
            {"strategy": "breakout_dual_tf", "symbol": symbol, "side": side},
        )
        skip_payload = self._has_active_position_or_orders(symbol, side)
        if skip_payload is not None:
            return skip_payload

        result = {
            "status": "signal",
            "strategy": "breakout_dual_tf",
            "symbol": symbol,
            "side": side,
            "signal": signal,
            "orders": orders,
        }

        if broker is None:
            logger.info(
                "placing orders %s",
                {
                    "strategy": "breakout_dual_tf",
                    "symbol": symbol,
                    "side": side,
                    "status": "skipped_no_exchange",
                },
            )
            return result

        logger.info(
            "placing orders %s",
            {
                "strategy": "breakout_dual_tf",
                "symbol": symbol,
                "side": side,
                "qty": orders.get("qty"),
            },
        )
        placement = self.place_orders(orders, exchange=broker)
        result["placement"] = placement
        return result


__all__ = ["BreakoutDualTFStrategy", "downscale_interval", "Level"]

