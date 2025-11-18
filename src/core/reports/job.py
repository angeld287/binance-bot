"""Reporting job orchestrator implementation."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time as time_cls, timedelta, timezone
from importlib import import_module
from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple

from zoneinfo import ZoneInfo

from binance.client import Client

from common.symbols import normalize_symbol
from exporters.roundtrip_builder import enrich_before_persist
from repositories.dynamo_store import DailyActivityStore, ExecutionReportStore

Orchestrator = Callable[[dict | None, datetime | None], Any]

logger = logging.getLogger(__name__)

_DEFAULT_TZ_NAME = "America/Santo_Domingo"
_DEFAULT_TZ = ZoneInfo(_DEFAULT_TZ_NAME)
_EPSILON = 1e-12
_RETROSPECTIVE_BLOCK_MS = 2 * 60 * 60 * 1000  # 2 hours
_RETROSPECTIVE_MAX_BLOCKS = 24  # 48 hours total lookback cap

_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("core.reports.pipeline", "run"),
    ("core.reports.orchestrator", "run"),
    ("core.reports.jobs", "run"),
    ("core.reports.job_impl", "run"),
    ("exporters.daily_export", "run"),
    ("services.daily_export_handler", "run"),
)


@dataclass
class _PositionState:
    direction: str | None = None
    net_size: float = 0.0
    trades: list[dict[str, Any]] = field(default_factory=list)
    is_open: bool = False
    completed_roundtrips: int = 0

    def reset(self) -> None:
        self.direction = None
        self.net_size = 0.0
        self.trades.clear()
        self.is_open = False

    def mark_open(self, sign: int) -> None:
        self.direction = "LONG" if sign > 0 else "SHORT"
        self.is_open = True

    def is_flat(self) -> bool:
        return abs(self.net_size) <= _EPSILON

    def active_sign(self) -> int:
        if self.net_size > _EPSILON:
            return 1
        if self.net_size < -_EPSILON:
            return -1
        return 0


def _iter_orchestrators() -> Iterable[tuple[Orchestrator, str]]:
    for module_name, attr_name in _CANDIDATES:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            continue
        orchestrator = getattr(module, attr_name, None)
        if callable(orchestrator):
            yield orchestrator, f"{module_name}.{attr_name}"


def _fallback_orchestrator(event_in: dict | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Fallback orchestrator used when no concrete implementation is present."""

    return {
        "status": "noop",
        "reason": "no_reports_orchestrator_found",
        "event": event_in or {},
        "timestamp": (now or datetime.utcnow()).isoformat(),
    }


def resolve_orchestrator() -> Tuple[Orchestrator, str]:
    """Return the reports orchestrator and its dotted path."""

    for orchestrator, path in _iter_orchestrators():
        if orchestrator is not run:
            return orchestrator, path
    return run, "core.reports.job.run"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _now_local(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(tz=_DEFAULT_TZ)
    if now.tzinfo is None:
        return now.replace(tzinfo=_DEFAULT_TZ)
    return now.astimezone(_DEFAULT_TZ)


def _parse_datetime_value(value: Any, tz: ZoneInfo) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            dt = datetime.fromtimestamp(float(text) / 1000, tz=timezone.utc)
        else:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                dt = datetime.fromtimestamp(float(text) / 1000, tz=timezone.utc)
    else:  # pragma: no cover - defensive branch
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def _resolve_window(
    from_value: Any,
    to_value: Any,
    *,
    now: datetime | None,
) -> tuple[datetime, datetime]:
    local_now = _now_local(now)
    start_dt = _parse_datetime_value(from_value, _DEFAULT_TZ)
    end_dt = _parse_datetime_value(to_value, _DEFAULT_TZ)

    if start_dt is None and end_dt is None:
        target_day = (local_now - timedelta(days=1)).date()
        start_dt = datetime.combine(target_day, time_cls(0, 0), tzinfo=_DEFAULT_TZ)
        end_dt = start_dt + timedelta(days=1) - timedelta(milliseconds=1)
    elif start_dt is None and end_dt is not None:
        day = end_dt.astimezone(_DEFAULT_TZ).date()
        start_dt = datetime.combine(day, time_cls(0, 0), tzinfo=_DEFAULT_TZ)
    elif end_dt is None and start_dt is not None:
        day = start_dt.astimezone(_DEFAULT_TZ).date()
        end_dt = datetime.combine(day, time_cls(23, 59, 59, 999000), tzinfo=_DEFAULT_TZ)

    assert start_dt is not None and end_dt is not None  # for mypy
    if end_dt < start_dt:
        end_dt = start_dt

    return start_dt.astimezone(_DEFAULT_TZ), end_dt.astimezone(_DEFAULT_TZ)


def _epoch_ms(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def _parse_symbols(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        tokens = [normalize_symbol(token) for token in value.split(",")]
        return [token for token in tokens if token]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = []
        for item in value:
            if item is None:
                continue
            items.append(normalize_symbol(str(item)))
        return [token for token in items if token]
    return []


def _default_symbols() -> list[str]:
    env_symbols = _parse_symbols(os.getenv("REPORT_SYMBOLS"))
    if env_symbols:
        return env_symbols
    env_symbol = _parse_symbols(os.getenv("SYMBOL"))
    if env_symbol:
        return env_symbol
    return ["BTCUSDT"]


def _dedupe_order(seq: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in seq:
        if item in seen or not item:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _resolve_parameters(event_in: Mapping[str, Any] | None, now: datetime | None) -> dict[str, Any]:
    event_in = event_in or {}
    from_value = (
        event_in.get("fromTs")
        or event_in.get("from_ts")
        or event_in.get("from")
        or event_in.get("start")
    )
    to_value = event_in.get("toTs") or event_in.get("to_ts") or event_in.get("to") or event_in.get("end")

    start_dt, end_dt = _resolve_window(from_value, to_value, now=now)
    symbols_raw = event_in.get("symbols") or event_in.get("symbol")
    symbols = _parse_symbols(symbols_raw) or _default_symbols()
    symbols = _dedupe_order(symbols)

    origin = str(event_in.get("origin") or event_in.get("source") or "manual").lower()
    origin = "eventbridge" if origin == "aws.events" else origin
    if origin not in {"manual", "eventbridge"}:
        origin = "manual"

    dry_run = _parse_bool(event_in.get("dryRun") or event_in.get("dry_run"))

    return {
        "origin": origin,
        "dry_run": dry_run,
        "symbols": symbols,
        "from_dt": start_dt,
        "to_dt": end_dt,
        "from_ts": _epoch_ms(start_dt),
        "to_ts": _epoch_ms(end_dt),
        "from_iso": start_dt.isoformat(),
        "to_iso": end_dt.isoformat(),
        "request_meta": {k: event_in.get(k) for k in sorted(event_in.keys())},
    }


def _create_binance_client() -> Client:
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET environment variables must be set")
    testnet = _parse_bool(os.getenv("BINANCE_TESTNET"))
    logger.info("reports.job.client.init", extra={"testnet": testnet})
    return Client(api_key=api_key, api_secret=api_secret, testnet=testnet)


def _generate_run_id(start_dt: datetime) -> str:
    base = start_dt.astimezone(_DEFAULT_TZ).strftime("%Y%m%dT%H%M%S")
    return f"reports-{base}-{uuid.uuid4().hex[:8]}"


def _call_client_method(client: Any, names: Sequence[str], **params: Any) -> Any:
    for name in names:
        method = getattr(client, name, None)
        if callable(method):
            return method(**params)
    raise AttributeError(f"Client does not expose any of {', '.join(names)}")


def _log_fetch_count(
    symbol: str,
    category: str,
    payload: Any,
    *,
    from_ts: int | None = None,
    to_ts: int | None = None,
) -> None:
    """Emit a structured log message with the raw count for a fetch call."""

    try:
        count = len(payload)  # type: ignore[arg-type]
    except TypeError:
        count = 0

    if from_ts is not None or to_ts is not None:
        logger.info(
            "reports.job.fetch_counts symbol=%s category=%s count=%s from_ts=%s to_ts=%s",
            symbol,
            category,
            count,
            from_ts,
            to_ts,
        )
    else:
        logger.info(
            "reports.job.fetch_counts symbol=%s category=%s count=%s",
            symbol,
            category,
            count,
        )


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0


def _build_orders_lookup(orders_map: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for symbol, orders in orders_map.items():
        for order in orders or []:
            order_id = order.get("orderId")
            if order_id is None:
                continue
            lookup[(symbol, str(order_id))] = order
    return lookup


def _build_income_index(income_map: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    index: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for symbol, incomes in income_map.items():
        norm_symbol = normalize_symbol(symbol)
        for income in incomes or []:
            trade_id = income.get("tradeId")
            if trade_id is None:
                continue
            key = (norm_symbol, str(trade_id))
            index.setdefault(key, []).append(income)
    return index


def _slice_trade(trade: Mapping[str, Any], qty: float) -> dict[str, Any]:
    orig_qty = abs(_to_float(trade.get("qty") or trade.get("origQty")))
    ratio = qty / orig_qty if orig_qty > _EPSILON else 0.0
    sliced = dict(trade)
    sliced["qty"] = qty
    for field in ("quoteQty", "realizedPnl", "commission"):
        if field in trade:
            sliced[field] = _to_float(trade.get(field)) * ratio
    return sliced


def _summarize_roundtrip(
    symbol: str,
    trades: Sequence[Mapping[str, Any]],
    direction: str | None,
    income_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    orders_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not trades:
        return None

    first_trade = trades[0]
    if direction is None:
        side = str(first_trade.get("side") or "BUY").upper()
        direction = "LONG" if side == "BUY" else "SHORT"

    entry_side = "BUY" if direction == "LONG" else "SELL"
    exit_side = "SELL" if direction == "LONG" else "BUY"

    entry_trades = [t for t in trades if str(t.get("side") or "").upper() == entry_side]
    exit_trades = [t for t in trades if str(t.get("side") or "").upper() == exit_side]

    if not entry_trades or not exit_trades:
        return None

    entry_qty = sum(_to_float(t.get("qty")) for t in entry_trades)
    exit_qty = sum(_to_float(t.get("qty")) for t in exit_trades)
    if entry_qty <= _EPSILON or exit_qty <= _EPSILON:
        return None

    entry_value = sum(_to_float(t.get("price")) * _to_float(t.get("qty")) for t in entry_trades)
    exit_value = sum(_to_float(t.get("price")) * _to_float(t.get("qty")) for t in exit_trades)
    if abs(entry_value) <= _EPSILON:
        return None

    entry_price = entry_value / entry_qty
    exit_price = exit_value / exit_qty
    qty_closed = min(entry_qty, exit_qty)

    open_ts = min(int(_to_float(t.get("time"))) for t in trades if t.get("time") is not None)
    close_ts = max(int(_to_float(t.get("time"))) for t in trades if t.get("time") is not None)

    commission_total = sum(_to_float(t.get("commission")) for t in trades)
    realized_pnl = sum(_to_float(t.get("realizedPnl")) for t in trades)
    net_pnl = realized_pnl + commission_total
    total_pnl_with_income = net_pnl
    roi_pct = (realized_pnl / entry_value) * 100.0
    roi_net_pct = (net_pnl / entry_value) * 100.0

    income_breakdown: dict[str, float] = {}
    income_total = 0.0
    trade_ids: list[str] = []
    for trade in trades:
        trade_id = trade.get("id") or trade.get("tradeId")
        if trade_id is None:
            continue
        tid = str(trade_id)
        trade_ids.append(tid)
        for income in income_index.get((symbol, tid), []):
            income_value = _to_float(income.get("income"))
            income_total += income_value
            income_type = str(income.get("incomeType") or "UNKNOWN").upper()
            income_breakdown[income_type] = income_breakdown.get(income_type, 0.0) + income_value

    total_pnl_with_income += income_total

    order_ids = sorted({str(trade.get("orderId")) for trade in trades if trade.get("orderId") is not None})
    order_details = []
    for order_id in order_ids:
        order = orders_lookup.get((symbol, order_id))
        if not order:
            continue
        detail = {
            "orderId": order.get("orderId"),
            "type": order.get("type"),
            "status": order.get("status"),
            "avgPrice": _to_float(order.get("avgPrice")),
            "updateTime": order.get("updateTime"),
            "clientOrderId": order.get("clientOrderId"),
        }
        order_details.append(detail)

    duration_ms = max(close_ts - open_ts, 0)
    open_dt_local = datetime.fromtimestamp(open_ts / 1000, tz=timezone.utc).astimezone(_DEFAULT_TZ)
    close_dt_local = datetime.fromtimestamp(close_ts / 1000, tz=timezone.utc).astimezone(_DEFAULT_TZ)
    day_key = open_dt_local.strftime("%Y-%m-%d")

    roundtrip: dict[str, Any] = {
        "symbol": symbol,
        "direction": direction,
        "entryQty": entry_qty,
        "exitQty": exit_qty,
        "qty": qty_closed,
        "entryPrice": entry_price,
        "exitPrice": exit_price,
        "entryValue": entry_value,
        "exitValue": exit_value,
        "openAt": open_ts,
        "closeAt": close_ts,
        "openTimestamp": open_ts,
        "closeTimestamp": close_ts,
        "openAtIso": open_dt_local.isoformat(),
        "closeAtIso": close_dt_local.isoformat(),
        "durationMs": duration_ms,
        "roiPct": roi_pct,
        "roiNetPct": roi_net_pct,
        "realizedPnl": realized_pnl,
        "commissionPaid": commission_total,
        "netPnl": net_pnl,
        "pnl": total_pnl_with_income,
        "incomeRealized": income_total,
        "incomeBreakdown": income_breakdown,
        "tradeIds": sorted(set(trade_ids)),
        "orderIds": order_ids,
        "ordersReferenced": order_details,
        "tz": _DEFAULT_TZ_NAME,
        "day": day_key,
    }

    position_side = next((trade.get("positionSide") for trade in trades if trade.get("positionSide")), None)
    if position_side:
        roundtrip["positionSide"] = position_side

    return roundtrip


def _trade_sort_key(trade: Mapping[str, Any]) -> tuple[int, int, int]:
    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    trade_id = _as_int(trade.get("id") or trade.get("tradeId") or trade.get("orderId"))
    ts = _as_int(trade.get("time"))
    update_ts = _as_int(trade.get("updateTime"))
    return trade_id, ts, update_ts


def _should_enter_retrospective(trade: Mapping[str, Any]) -> bool:
    qty = abs(_to_float(trade.get("qty") or trade.get("origQty")))
    if qty <= _EPSILON:
        return False
    side = str(trade.get("side") or "BUY").upper()
    position_side = str(trade.get("positionSide") or "").upper()
    realized_pnl = abs(_to_float(trade.get("realizedPnl"))) > _EPSILON

    if realized_pnl:
        return True
    if position_side == "LONG" and side == "SELL":
        return True
    if position_side == "SHORT" and side == "BUY":
        return True
    if position_side in {"", "BOTH"} and side == "SELL":
        return True
    if position_side in {"", "BOTH"} and side == "BUY" and trade.get("reduceOnly"):
        return True
    return False


def _replay_trades_for_state(
    symbol: str,
    trades: Sequence[Mapping[str, Any]],
    income_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    orders_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[_PositionState, int | None, str | None, bool]:
    state = _PositionState()
    last_zero_ts: int | None = None
    last_zero_trade_id: str | None = None
    zero_seen = False

    for trade in trades:
        produced, _skipped = _consume_trade(
            state,
            trade,
            symbol=symbol,
            income_index=income_index,
            orders_lookup=orders_lookup,
        )
        if produced:
            zero_seen = True
            last_zero_ts = produced[-1].get("closeTimestamp")
            trade_id = trade.get("id") or trade.get("tradeId")
            if trade_id is not None:
                last_zero_trade_id = str(trade_id)

    return state, last_zero_ts, last_zero_trade_id, zero_seen


def _retrospective_search(
    symbol: str,
    *,
    client: Any,
    range_start: int,
    income_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    orders_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[_PositionState, dict[str, Any]]:
    lookback_trades: list[Mapping[str, Any]] = []
    seen: set[tuple[str | int | None, int]] = set()
    blocks_used = 0
    last_zero_ts: int | None = None
    last_zero_trade_id: str | None = None
    found_zero = False
    state: _PositionState | None = None

    logger.info("reports.job.retrospective.enter symbol=%s start_ts=%s", symbol, range_start)

    while blocks_used < _RETROSPECTIVE_MAX_BLOCKS:
        block_end = range_start - blocks_used * _RETROSPECTIVE_BLOCK_MS
        block_start = max(block_end - _RETROSPECTIVE_BLOCK_MS, 0)
        blocks_used += 1

        block = _call_client_method(
            client,
            ("futures_account_trades", "get_my_trades", "user_trades"),
            symbol=symbol,
            startTime=block_start,
            endTime=block_end,
            limit=1000,
        )
        _log_fetch_count(
            symbol,
            "account_trades_retrospective",
            block or [],
            from_ts=block_start,
            to_ts=block_end,
        )

        new_trades: list[Mapping[str, Any]] = []
        for trade in block or []:
            trade_id = trade.get("id") or trade.get("tradeId") or trade.get("orderId")
            key = (trade_id, int(_to_float(trade.get("time"))))
            if key in seen:
                continue
            seen.add(key)
            new_trades.append(trade)

        if new_trades:
            lookback_trades.extend(new_trades)

        if not lookback_trades and not new_trades:
            logger.info(
                "reports.job.retrospective.no_trades symbol=%s block=%s block_start=%s block_end=%s",
                symbol,
                blocks_used,
                block_start,
                block_end,
            )
            break

        sorted_history = sorted(lookback_trades, key=_trade_sort_key)
        state, last_zero_ts, last_zero_trade_id, zero_seen = _replay_trades_for_state(
            symbol,
            sorted_history,
            income_index,
            orders_lookup,
        )
        found_zero = found_zero or zero_seen
        logger.info(
            "reports.job.retrospective.progress symbol=%s block=%s accumulated_trades=%s net_after=%s found_zero=%s zero_trade_id=%s zero_ts=%s block_start=%s block_end=%s",
            symbol,
            blocks_used,
            len(sorted_history),
            state.net_size,
            found_zero,
            last_zero_trade_id,
            last_zero_ts,
            block_start,
            block_end,
        )

        if found_zero:
            break
        if not block:
            # Stop when server returned no more trades to avoid looping forever
            break

    if state is None:
        state = _PositionState()

    logger.info(
        "reports.job.retrospective.exit symbol=%s blocks_used=%s found_zero=%s zero_trade_id=%s zero_ts=%s net_at_start=%s",
        symbol,
        blocks_used,
        found_zero,
        last_zero_trade_id,
        last_zero_ts,
        state.net_size,
    )

    metadata = {
        "blocksUsed": blocks_used,
        "foundZero": found_zero,
        "zeroTradeId": last_zero_trade_id,
        "zeroTimestamp": last_zero_ts,
        "netAtStart": state.net_size,
    }
    return state, metadata


def _bootstrap_state(
    symbol: str,
    trades: Sequence[Mapping[str, Any]],
    *,
    client: Any | None,
    range_start: int | None,
    income_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    orders_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[_PositionState, dict[str, Any]]:
    state = _PositionState()
    meta: dict[str, Any] = {"retrospective": False}

    if not trades or client is None or range_start is None:
        return state, meta

    first_trade = trades[0]
    if not _should_enter_retrospective(first_trade):
        return state, meta

    logger.info(
        "reports.job.retrospective.trigger symbol=%s first_trade_id=%s first_trade_time=%s",
        symbol,
        first_trade.get("id") or first_trade.get("tradeId"),
        first_trade.get("time"),
    )
    state, metadata = _retrospective_search(
        symbol,
        client=client,
        range_start=range_start,
        income_index=income_index,
        orders_lookup=orders_lookup,
    )
    meta.update(metadata)
    meta["retrospective"] = True
    return state, meta


def _consume_trade(
    state: _PositionState,
    trade: Mapping[str, Any],
    *,
    symbol: str,
    income_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    orders_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Apply ``trade`` into ``state`` and return any completed roundtrips.

    The net position for the symbol is tracked in ``state.net_size``. Every time
    it moves 0 -> non-zero -> 0 we close a roundtrip, even if multiple positions
    share the same direction or size.
    """
    produced: list[dict[str, Any]] = []
    skipped = 0

    def _finalize_cycle() -> None:
        nonlocal skipped
        net_at_close = state.net_size
        trades_count = len(state.trades)
        roundtrip_index = state.completed_roundtrips + 1
        roundtrip = _summarize_roundtrip(symbol, state.trades, state.direction, income_index, orders_lookup)
        if roundtrip is None:
            skipped += 1
        else:
            logger.info(
                "reports.job.roundtrip_state symbol=%s idx=%s trades_count=%s net_at_close=%s open_ts=%s close_ts=%s pnl=%s qty=%s",
                symbol,
                roundtrip_index,
                trades_count,
                net_at_close,
                roundtrip.get("openTimestamp"),
                roundtrip.get("closeTimestamp"),
                roundtrip.get("pnl"),
                roundtrip.get("qty"),
            )
            produced.append(roundtrip)
            state.completed_roundtrips = roundtrip_index
        state.reset()

    side = str(trade.get("side") or "BUY").upper()
    sign = 1 if side == "BUY" else -1
    total_qty = abs(_to_float(trade.get("qty") or trade.get("origQty")))
    if total_qty <= _EPSILON:
        return produced, skipped

    remaining = total_qty

    while remaining > _EPSILON:
        if not state.is_open or state.is_flat():
            state.mark_open(sign)

        position_sign = state.active_sign()
        incoming_same_direction = position_sign == 0 or position_sign == sign

        qty_to_use = remaining if incoming_same_direction else min(remaining, abs(state.net_size))
        if qty_to_use <= _EPSILON:
            break

        partial_trade = _slice_trade(trade, qty_to_use)
        partial_trade["side"] = side
        net_before = state.net_size
        state.trades.append(partial_trade)
        state.net_size += sign * qty_to_use
        net_after = state.net_size
        remaining -= qty_to_use

        current_roundtrip_index = state.completed_roundtrips + 1
        trade_id = trade.get("id") or trade.get("tradeId")
        trade_time = trade.get("time")
        trade_position_side = trade.get("positionSide")
        logger.info(
            "reports.job.position_trace symbol=%s trade_id=%s time=%s side=%s qty=%s position_side=%s net_before=%s net_after=%s roundtrip_index=%s",
            symbol,
            trade_id,
            trade_time,
            side,
            qty_to_use,
            trade_position_side,
            net_before,
            net_after,
            current_roundtrip_index,
        )

        if state.is_flat():
            _finalize_cycle()

    return produced, skipped


def _build_roundtrips(
    symbols: Sequence[str],
    trades_map: Mapping[str, Sequence[Mapping[str, Any]]],
    income_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    orders_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    client: Any | None = None,
    range_start: int | None = None,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    roundtrips: list[dict[str, Any]] = []
    skipped = 0
    leftovers: list[dict[str, Any]] = []

    for symbol in symbols:
        trades = trades_map.get(symbol) or []
        if not trades:
            continue
        sorted_trades = sorted(trades, key=_trade_sort_key)
        if sorted_trades:
            first_tid = _trade_sort_key(sorted_trades[0])[0]
            last_tid = _trade_sort_key(sorted_trades[-1])[0]
            logger.info(
                "reports.job.trades_sorted symbol=%s first_trade_id=%s last_trade_id=%s total=%s",
                symbol,
                first_tid,
                last_tid,
                len(sorted_trades),
            )
        state, boot_meta = _bootstrap_state(
            symbol,
            sorted_trades,
            client=client,
            range_start=range_start,
            income_index=income_index,
            orders_lookup=orders_lookup,
        )
        if boot_meta.get("retrospective"):
            logger.info(
                "reports.job.retrospective.state_ready symbol=%s net_at_start=%s blocks_used=%s found_zero=%s zero_ts=%s",
                symbol,
                boot_meta.get("netAtStart"),
                boot_meta.get("blocksUsed"),
                boot_meta.get("foundZero"),
                boot_meta.get("zeroTimestamp"),
            )
        for trade in sorted_trades:
            produced, skipped_count = _consume_trade(
                state,
                trade,
                symbol=symbol,
                income_index=income_index,
                orders_lookup=orders_lookup,
            )
            roundtrips.extend(produced)
            skipped += skipped_count
        if state.trades:
            first_ts = min(int(_to_float(t.get("time"))) for t in state.trades if t.get("time") is not None)
            last_ts = max(int(_to_float(t.get("time"))) for t in state.trades if t.get("time") is not None)
            logger.info(
                "Position discarded: ended with net != 0 at end of range",
                extra={
                    "symbol": symbol,
                    "net": state.net_size,
                    "trades": len(state.trades),
                    "firstTs": first_ts,
                    "lastTs": last_ts,
                },
            )
            leftovers.append({
                "symbol": symbol,
                "direction": state.direction,
                "openTrades": len(state.trades),
                "openQty": abs(state.net_size),
                "firstTs": first_ts,
                "lastTs": last_ts,
            })

    return roundtrips, skipped, leftovers


def _format_iso(dt: datetime) -> str:
    iso = dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return iso


def run(event_in: dict | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Execute the reporting job pipeline."""

    start_utc = datetime.now(timezone.utc)

    params = _resolve_parameters(event_in, now)
    symbols = params["symbols"]
    dry_run = params["dry_run"]

    logger.info(
        "reports.job.start",
        extra={
            "origin": params["origin"],
            "fromTs": params["from_iso"],
            "toTs": params["to_iso"],
            "symbols": symbols,
            "dryRun": dry_run,
        },
    )

    client: Client | None = None
    daily_store: DailyActivityStore | None = None
    exec_store: ExecutionReportStore | None = None
    run_id: str | None = None
    processed = 0
    inserted = 0
    skipped = 0
    error_message: str | None = None
    fetch_stats: dict[str, Any] = {}
    roundtrips_payload: list[dict[str, Any]] = []

    try:
        client = _create_binance_client()

        orders_map: dict[str, list[Mapping[str, Any]]] = {}
        trades_map: dict[str, list[Mapping[str, Any]]] = {}
        income_map: dict[str, list[Mapping[str, Any]]] = {}
        api_calls = 0

        from_ts = params["from_ts"]
        to_ts = params["to_ts"]

        for symbol in symbols:
            fetch_args = {"symbol": symbol, "startTime": from_ts, "endTime": to_ts, "limit": 1000}
            orders = _call_client_method(client, ("futures_all_orders", "get_all_orders", "all_orders"), **fetch_args)
            _log_fetch_count(symbol, "orders", orders or [], from_ts=from_ts, to_ts=to_ts)
            trades = _call_client_method(client, ("futures_account_trades", "get_my_trades", "user_trades"), **fetch_args)
            _log_fetch_count(symbol, "account_trades", trades or [], from_ts=from_ts, to_ts=to_ts)
            income = []
            try:
                income = _call_client_method(
                    client,
                    ("futures_income_history",),
                    symbol=symbol,
                    startTime=from_ts,
                    endTime=to_ts,
                    limit=1000,
                )
                _log_fetch_count(symbol, "income", income or [], from_ts=from_ts, to_ts=to_ts)
            except AttributeError:
                income = []
            klines = _call_client_method(
                client,
                ("futures_klines", "get_klines", "klines"),
                symbol=symbol,
                interval="1m",
                startTime=from_ts,
                endTime=to_ts,
                limit=1500,
            )
            _log_fetch_count(symbol, "klines", klines or [], from_ts=from_ts, to_ts=to_ts)
            orders_map[symbol] = list(orders or [])
            trades_map[symbol] = list(trades or [])
            income_map[symbol] = list(income or [])
            api_calls += 4

            highs = [
                _to_float(candle[2])
                for candle in klines or []
                if isinstance(candle, (list, tuple)) and len(candle) > 2
            ]
            lows = [
                _to_float(candle[3])
                for candle in klines or []
                if isinstance(candle, (list, tuple)) and len(candle) > 3
            ]
            fetch_stats[symbol] = {
                "orders": len(orders_map[symbol]),
                "trades": len(trades_map[symbol]),
                "income": len(income_map[symbol]),
                "klines": len(klines or []),
                "dayHigh": max(highs) if highs else None,
                "dayLow": min(lows) if lows else None,
            }

        orders_lookup = _build_orders_lookup(orders_map)
        income_index = _build_income_index(income_map)
        base_roundtrips, skipped_roundtrips, leftovers = _build_roundtrips(
            symbols,
            trades_map,
            income_index,
            orders_lookup,
            client=client,
            range_start=from_ts,
        )
        skipped += skipped_roundtrips
        processed = len(base_roundtrips)

        if leftovers:
            logger.warning("reports.job.leftover_positions", extra={"positions": leftovers})

        run_id = _generate_run_id(params["from_dt"])
        start_ms = int(start_utc.timestamp() * 1000)

        if not dry_run:
            exec_store = ExecutionReportStore()
            daily_store = DailyActivityStore()
            exec_store.start_run(
                run_id,
                start_ms,
                _DEFAULT_TZ_NAME,
                {
                    "origin": params["origin"],
                    "symbols": symbols,
                    "from": params["from_iso"],
                    "to": params["to_iso"],
                    "dryRun": dry_run,
                    "request": params["request_meta"],
                    "fetchStats": fetch_stats,
                },
            )

        for idx, roundtrip in enumerate(base_roundtrips, start=1):
            rt = dict(roundtrip)
            pk = f"DAY#{rt['day']}#SYM#{rt['symbol']}"
            sk = f"RT#{run_id}#{idx:04d}"
            rt.update({
                "PK": pk,
                "SK": sk,
                "runId": run_id,
                "origin": params["origin"],
            })
            try:
                enriched = enrich_before_persist(rt, client=client, tz_name=_DEFAULT_TZ_NAME)
            except Exception as exc:  # pragma: no cover - analytics failures
                logger.exception(
                    "reports.job.enrich_failed",
                    extra={"symbol": rt.get("symbol"), "sk": sk},
                )
                skipped += 1
                continue

            roundtrips_payload.append(enriched)
            qty_closed = _to_float(enriched.get("qty"))
            if qty_closed <= _EPSILON:
                qty_closed = min(
                    _to_float(enriched.get("entryQty")),
                    _to_float(enriched.get("exitQty")),
                )
            open_ts_raw = enriched.get("openTimestamp")
            if open_ts_raw is None:
                open_ts_raw = enriched.get("openAt")
            open_ts = int(_to_float(open_ts_raw)) if open_ts_raw is not None else None
            close_ts_raw = enriched.get("closeTimestamp")
            if close_ts_raw is None:
                close_ts_raw = enriched.get("closeAt")
            close_ts = int(_to_float(close_ts_raw)) if close_ts_raw is not None else None
            pnl_value = enriched.get("pnl")
            if pnl_value is None:
                pnl_value = enriched.get("netPnl")
            if pnl_value is None:
                pnl_value = enriched.get("realizedPnl")
            pnl_value = _to_float(pnl_value)
            entry_price = _to_float(enriched.get("entryPrice"))
            exit_price = _to_float(enriched.get("exitPrice"))
            logger.info(
                "reports.job.roundtrip_debug symbol=%s side=%s qty=%s open_ts=%s close_ts=%s pnl=%s entry_price=%s exit_price=%s",
                enriched.get("symbol"),
                enriched.get("direction"),
                qty_closed,
                open_ts,
                close_ts,
                pnl_value,
                entry_price,
                exit_price,
            )
            if dry_run:
                logger.info(
                    "reports.job.dry_run.roundtrip",
                    extra={"symbol": enriched.get("symbol"), "sk": sk},
                )
                continue

            assert daily_store is not None
            try:
                daily_store.put_roundtrip(enriched)
                inserted += 1
            except Exception:  # pragma: no cover - persistence failures
                skipped += 1
                logger.exception(
                    "reports.job.persist_failed",
                    extra={"symbol": enriched.get("symbol"), "sk": sk},
                )

        if exec_store and not dry_run:
            try:
                exec_store.inc_metrics(
                    run_id,
                    {
                        "opsAnalyzed": processed,
                        "apiWeightsUsed": api_calls,
                        "symbolsProcessed": symbols,
                        "errorsCount": skipped,
                    },
                )
            except Exception:  # pragma: no cover - persistence failures
                logger.exception("reports.job.metrics_failed", extra={"runId": run_id})

    except Exception as exc:  # pragma: no cover - top level guard
        error_message = str(exc)
        logger.exception(
            "reports.job.failure",
            extra={
                "origin": params["origin"],
                "symbols": symbols,
                "fromTs": params["from_iso"],
                "toTs": params["to_iso"],
            },
        )
    finally:
        end_utc = datetime.now(timezone.utc)
        end_ms = int(end_utc.timestamp() * 1000)
        duration_ms = int((end_utc - start_utc).total_seconds() * 1000)
        status = "error" if error_message else "success"

        if exec_store and run_id and not dry_run:
            totals = {
                "opsClosed": inserted,
                "opsAnalyzed": processed,
                "errorsCount": skipped,
            }
            try:
                exec_store.finalize_run(
                    run_id,
                    end_ms,
                    totals,
                    status.upper(),
                    error_message,
                )
            except Exception:  # pragma: no cover - finalize failures
                logger.exception("reports.job.finalize_failed", extra={"runId": run_id})

        summary = {
            "status": status,
            "origin": params["origin"],
            "processed": processed,
            "inserted": inserted if not dry_run else 0,
            "skipped": skipped,
            "fromTs": params["from_iso"],
            "toTs": params["to_iso"],
            "symbols": symbols,
            "dryRun": dry_run,
            "start": _format_iso(start_utc),
            "end": _format_iso(end_utc),
            "duration_ms": duration_ms,
            "roundtrips": roundtrips_payload if dry_run else [],
            "fetchStats": fetch_stats,
        }
        if error_message:
            summary["error"] = error_message

    logger.info(
        "reports.job.completed",
        extra={
            "status": summary.get("status"),
            "processed": summary.get("processed"),
            "inserted": summary.get("inserted"),
            "skipped": summary.get("skipped"),
        },
    )

    return summary
