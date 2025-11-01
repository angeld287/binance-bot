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
    position: float = 0.0
    trades: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.direction = None
        self.position = 0.0
        self.trades.clear()


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

    open_ts = min(int(_to_float(t.get("time"))) for t in trades if t.get("time") is not None)
    close_ts = max(int(_to_float(t.get("time"))) for t in trades if t.get("time") is not None)

    commission_total = sum(_to_float(t.get("commission")) for t in trades)
    realized_pnl = sum(_to_float(t.get("realizedPnl")) for t in trades)
    net_pnl = realized_pnl + commission_total
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
        "entryPrice": entry_price,
        "exitPrice": exit_price,
        "entryValue": entry_value,
        "exitValue": exit_value,
        "openAt": open_ts,
        "closeAt": close_ts,
        "openAtIso": open_dt_local.isoformat(),
        "closeAtIso": close_dt_local.isoformat(),
        "durationMs": duration_ms,
        "roiPct": roi_pct,
        "roiNetPct": roi_net_pct,
        "realizedPnl": realized_pnl,
        "commissionPaid": commission_total,
        "netPnl": net_pnl,
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


def _consume_trade(
    state: _PositionState,
    trade: Mapping[str, Any],
    *,
    symbol: str,
    income_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    orders_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    produced: list[dict[str, Any]] = []
    skipped = 0

    side = str(trade.get("side") or "BUY").upper()
    sign = 1 if side == "BUY" else -1
    total_qty = abs(_to_float(trade.get("qty") or trade.get("origQty")))
    if total_qty <= _EPSILON:
        return produced, skipped

    remaining = total_qty

    while remaining > _EPSILON:
        if state.direction is None:
            state.direction = "LONG" if sign > 0 else "SHORT"

        same_direction = (state.direction == "LONG" and sign > 0) or (state.direction == "SHORT" and sign < 0)

        if same_direction or abs(state.position) <= _EPSILON:
            qty_to_use = remaining
        else:
            qty_to_use = min(remaining, abs(state.position))
            if qty_to_use <= _EPSILON:
                state.reset()
                state.direction = "LONG" if sign > 0 else "SHORT"
                qty_to_use = remaining

        partial_trade = _slice_trade(trade, qty_to_use)
        partial_trade["side"] = side
        state.trades.append(partial_trade)
        state.position += sign * qty_to_use
        remaining -= qty_to_use

        if abs(state.position) <= _EPSILON:
            roundtrip = _summarize_roundtrip(symbol, state.trades, state.direction, income_index, orders_lookup)
            if roundtrip is None:
                skipped += 1
            else:
                produced.append(roundtrip)
            state.reset()
            if remaining > _EPSILON:
                state.direction = "LONG" if sign > 0 else "SHORT"

    return produced, skipped


def _build_roundtrips(
    symbols: Sequence[str],
    trades_map: Mapping[str, Sequence[Mapping[str, Any]]],
    income_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    orders_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    roundtrips: list[dict[str, Any]] = []
    skipped = 0
    leftovers: list[dict[str, Any]] = []

    for symbol in symbols:
        trades = trades_map.get(symbol) or []
        if not trades:
            continue
        sorted_trades = sorted(trades, key=lambda t: int(_to_float(t.get("time"))))
        state = _PositionState()
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
            leftovers.append({
                "symbol": symbol,
                "direction": state.direction,
                "openTrades": len(state.trades),
                "openQty": abs(state.position),
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

        for symbol in symbols:
            fetch_args = {"symbol": symbol, "startTime": params["from_ts"], "endTime": params["to_ts"], "limit": 1000}
            orders = _call_client_method(client, ("futures_all_orders", "get_all_orders", "all_orders"), **fetch_args)
            trades = _call_client_method(client, ("futures_account_trades", "get_my_trades", "user_trades"), **fetch_args)
            income = []
            try:
                income = _call_client_method(
                    client,
                    ("futures_income_history",),
                    symbol=symbol,
                    startTime=params["from_ts"],
                    endTime=params["to_ts"],
                    limit=1000,
                )
            except AttributeError:
                income = []
            klines = _call_client_method(
                client,
                ("futures_klines", "get_klines", "klines"),
                symbol=symbol,
                interval="1m",
                startTime=params["from_ts"],
                endTime=params["to_ts"],
                limit=1500,
            )
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
        base_roundtrips, skipped_roundtrips, leftovers = _build_roundtrips(symbols, trades_map, income_index, orders_lookup)
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
