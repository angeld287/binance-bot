"""Breakout trading strategy implementation.

This class wraps the previous breakout logic and exposes it through the
:class:`core.ports.strategy.Strategy` interface. Network or exchange access is
performed exclusively via the injected ports.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
import logging
import json
import os
from zoneinfo import ZoneInfo
from math import ceil

from common.utils import sanitize_client_order_id, is_in_blackout
from core.domain.models.Signal import Signal
from core.ports.broker import BrokerPort
from core.ports.market_data import MarketDataPort
from core.ports.strategy import Strategy
from core.ports.settings import SettingsProvider, get_symbol
from .cr_hook import run_cr_on_open_position

logger = logging.getLogger("bot.strategy.breakout")


def _interval_to_minutes(interval: str) -> int:
    units = {"m": 1, "h": 60, "d": 1440}
    return int(interval[:-1]) * units[interval[-1]]


class BreakoutStrategy(Strategy):
    """Simple breakout strategy based on the latest two candles."""

    def __init__(
        self,
        market_data: MarketDataPort,
        broker: BrokerPort,
        settings: SettingsProvider,
        repositories: Any | None = None,
    ) -> None:
        self._market_data = market_data
        self._broker = broker
        self._settings = settings
        self._repositories = repositories

    # ------------------------------------------------------------------
    def generate_signal(self, now: datetime) -> Signal | None:
        """Generate a trading signal at ``now``.

        The strategy compares the latest candle close with the previous
        candle's high and low. A breakout above the previous high produces a
        ``BUY`` signal; a breakdown below the previous low produces a ``SELL``
        signal. If neither condition is met, ``None`` is returned.
        """

        symbol = get_symbol(self._settings)
        interval = self._settings.get("INTERVAL", "1h")

        interval_min = _interval_to_minutes(interval)
        candles = self._market_data.get_klines(
            symbol=symbol, interval=interval, lookback_min=interval_min * 2
        )
        logger.debug("Candles fetched: %s", candles)
        if len(candles) < 2:
            logger.info("🔎 OBS: %s", "skip:not_enough_candles")
            return None

        prev, last = candles[-2], candles[-1]
        prev_high = float(prev[2])
        prev_low = float(prev[3])
        last_close = float(last[4])

        supports = [(prev_low, 1.0)]
        resistances = [(prev_high, 1.0)]
        supports_str = ", ".join([f"{p:.6f} (score {s:.2f})" for p, s in supports])
        resistances_str = ", ".join([f"{p:.6f} (score {s:.2f})" for p, s in resistances])
        logger.info("🛡️ Soportes estimados: %s", supports_str)
        logger.info("📚 Resistencias estimadas: %s", resistances_str)

        support_price, support_score = supports[0]
        resistance_price, resistance_score = resistances[0]
        support_dist = abs((last_close - support_price) / support_price * 100)
        resistance_dist = abs((resistance_price - last_close) / resistance_price * 100)
        logger.info(
            "🛡️ Próximo soporte: %.6f (score %.2f, dist≈%.2f%%) | razones: %s",
            support_price,
            support_score,
            support_dist,
            "prev_low",
        )
        logger.info(
            "🧱 Próxima resistencia: %.6f (score %.2f, dist≈%.2f%%) | razones: %s",
            resistance_price,
            resistance_score,
            resistance_dist,
            "prev_high",
        )

        action: str | None = None
        if last_close > prev_high:
            action = "BUY"
        elif last_close < prev_low:
            action = "SELL"

        if action is None:
            logger.info("🔎 OBS: %s", "no_signal")
            return None

        price = self._market_data.get_price(symbol)
        return Signal(action=action, price=price, time=now)

    # ------------------------------------------------------------------
    def run(
        self,
        exchange: BrokerPort | None = None,
        market_data: MarketDataPort | None = None,
        settings: SettingsProvider | None = None,
        now_utc: datetime | None = None,
        event: Any | None = None,
    ) -> dict[str, Any]:
        """Execute the strategy once and place an order if a signal emerges."""

        exch = exchange or self._broker
        settings = settings or self._settings
        market_data = market_data or self._market_data
        now = now_utc or datetime.utcnow()

        symbol = get_symbol(settings)
        ctx = {
            "exchange": exch,
            "settings": settings,
            "market_data": market_data,
            "now": now,
        }

        position_amt = 0.0
        entry_price = 0.0
        try:
            info = exch.get_position(symbol)
            if info is None:
                logger.info("get_position: no data for %s", symbol)
            else:
                position_amt = float(info.get("positionAmt", 0.0))
                entry_price = float(info.get("entryPrice", 0.0))
        except Exception as err:  # pragma: no cover - defensive
            logger.warning(
                "positioncheck.pre: ERROR {symbol=%s, error=%s}",
                symbol,
                err,
            )
        logger.info("position_amt: %s", position_amt)
        if abs(position_amt) > 0:
            side = "BUY" if position_amt > 0 else "SELL"
            logger.info(
                "positioncheck.pre: OPEN {symbol=%s, side=%s, qty=%s, entry=%s}",
                symbol,
                side,
                position_amt,
                entry_price,
            )
            logger.info(
                "breakout: posición abierta detectada; ejecutando CR hook antes de retornar"
            )
            current_position = {
                "positionAmt": position_amt,
                "entryPrice": entry_price,
                "side": side,
            }
            run_cr_on_open_position(
                ctx=ctx,
                symbol=symbol,
                position=current_position,
                logger=logger,
            )
            return {
                "status": "skipped_existing_position",
                "symbol": symbol,
                "positionAmt": position_amt,
                "entryPrice": entry_price,
            }
        else:
            logger.info(
                "positioncheck.pre: NONE {symbol=%s}",
                symbol,
            )

        try:
            open_orders = exch.open_orders(symbol)
            working = [
                o
                for o in open_orders
                if o.get("type") == "LIMIT"
                and o.get("status") in {"NEW", "PARTIALLY_FILLED"}
                and not o.get("reduceOnly")
            ]
            buys: list[float] = []
            sells: list[float] = []
            for o in working:
                try:
                    price = float(o.get("price", 0.0))
                except (TypeError, ValueError):
                    continue
                if o.get("side") == "BUY":
                    buys.append(price)
                elif o.get("side") == "SELL":
                    sells.append(price)
            buy_prices = sorted(buys)[:5]
            sell_prices = sorted(sells)[:5]
            n_buy = len(buys)
            n_sell = len(sells)
            total = len(working)
            if total:
                logger.info(
                    "ordercheck.pre: OPEN {symbol=%s, buys=%s, buy_prices=%s, sells=%s, sell_prices=%s, total=%s}",
                    symbol,
                    n_buy,
                    buy_prices,
                    n_sell,
                    sell_prices,
                    total,
                )
                logger.info(
                    "skip: existing working order {symbol=%s, buys=%s, buy_prices=%s, sells=%s, sell_prices=%s, total=%s}",
                    symbol,
                    n_buy,
                    buy_prices,
                    n_sell,
                    sell_prices,
                    total,
                )
                return {
                    "status": "skipped_existing_order",
                    "symbol": symbol,
                    "open_orders_total": total,
                    "open_orders_buys": n_buy,
                    "open_orders_sells": n_sell,
                }
            else:
                logger.info(
                    "ordercheck.pre: NONE {symbol=%s, total=0}",
                    symbol,
                )
        except Exception as err:
            logger.warning(
                "ordercheck.pre: ERROR {symbol=%s, error=%s}",
                symbol,
                err,
            )

        tz = os.getenv("BLACKOUT_TZ", "America/New_York")
        windows = os.getenv("BLACKOUT_WINDOWS", "")
        if is_in_blackout(now, tz, windows):
            local_time = (
                now if now.tzinfo else now.replace(tzinfo=ZoneInfo("UTC"))
            ).astimezone(ZoneInfo(tz)).strftime("%H:%M")
            logger.info(
                json.dumps(
                    {
                        "status": "skipped_blackout",
                        "strategy": "breakout",
                        "tz": tz,
                        "local_time": local_time,
                        "windows": windows,
                    }
                )
            )
            return {"status": "skipped_blackout"}

        signal = self.generate_signal(now)
        if signal is None:
            return {"status": "no_signal"}

        price_prev = signal.price
        price_norm = exch.round_price_to_tick(symbol, price_prev)

        filters = exch.get_symbol_filters(symbol)
        lot = filters.get("LOT_SIZE", {})
        step_size = float(lot.get("stepSize", 0.0))
        qty_min = float(lot.get("minQty", 0.0))
        tick_size = float(filters.get("PRICE_FILTER", {}).get("tickSize", 0.0))
        min_notional = float(
            filters.get("MIN_NOTIONAL", {}).get("notional")
            or filters.get("MIN_NOTIONAL", {}).get("minNotional", 0.0)
        )

        risk_notional = float(getattr(settings, "RISK_NOTIONAL_USDT", 0.0) or 0.0)
        qty_target_src = "NONE"
        if risk_notional > 0 and price_norm > 0:
            qty_target = risk_notional / price_norm
            qty_target_src = "NOTIONAL"
        else:
            balance = 0.0
            try:
                balance = float(exch.get_available_balance_usdt())
            except Exception:
                balance = 0.0
            risk_pct = float(getattr(settings, "RISK_PCT", 0.0) or 0.0)
            if risk_pct > 0 and balance > 0 and price_norm > 0:
                qty_target = (balance * risk_pct) / price_norm
                qty_target_src = "PCT"
            else:
                qty_target = 0.0

        def _ceil_to_step(x: float, step: float) -> float:
            return ceil(x / step) * step if step else x

        qty_min_by_notional = _ceil_to_step(min_notional / price_norm if price_norm else 0.0, step_size)
        qty_norm = exch.round_qty_to_step(symbol, max(qty_target, qty_min, qty_min_by_notional))

        if price_norm * qty_norm < min_notional and price_norm > 0:
            qty_norm = exch.round_qty_to_step(
                symbol, _ceil_to_step(min_notional / price_norm, step_size)
            )

        logger.info(
            json.dumps(
                {
                    "sizing_trace": {
                        "price_prev": price_prev,
                        "price_norm": price_norm,
                        "stepSize": step_size,
                        "tickSize": tick_size,
                        "minQty": qty_min,
                        "minNotional": min_notional,
                        "qty_target_src": qty_target_src,
                        "qty_target": qty_target,
                        "qty_norm": qty_norm,
                        "notional": price_norm * qty_norm,
                    }
                }
            )
        )

        cid = sanitize_client_order_id(f"breakout-{int(now.timestamp())}")

        try:
            order = exch.place_entry_limit(
                symbol,
                signal.action,
                price_norm,
                qty_norm,
                cid,
                timeInForce="GTC",
            )
        except TypeError:  # pragma: no cover - legacy brokers
            order = exch.place_entry_limit(
                symbol,
                signal.action,
                price_norm,
                qty_norm,
                cid,
            )

        return {
            "status": "order_placed",
            "side": signal.action,
            "price": price_norm,
            "qty": qty_norm,
            "clientOrderId": cid,
            "order": order,
        }
