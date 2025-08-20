import math
import statistics
from typing import List, Dict, Optional

try:
    from binance.client import Client
except Exception:  # pragma: no cover - library may not be available in tests
    Client = None  # type: ignore

from bot_trading import log


def _round_to_tick(price: float, tick_size: float, precision: int) -> float:
    """Round *price* respecting tick size and precision."""
    try:
        if tick_size and tick_size > 0:
            steps = round(float(price) / tick_size)
            return round(steps * tick_size, precision)
    except Exception:
        pass
    return round(float(price), precision)


def _fmt_price(price: float, precision: int) -> str:
    return f"{price:.{precision}f}"


def _is_round(level: float, precision: int) -> bool:
    mult = 10 ** precision
    try:
        val = int(round(level * mult))
        last_two = val % 100
        return last_two in {0, 25, 50, 75}
    except Exception:
        return False


def _get_client() -> Optional[Client]:
    if Client is None:  # pragma: no cover
        return None
    try:
        return Client(None, None)
    except Exception:  # pragma: no cover
        return None


def next_resistances(symbol: str, interval: str = "5m", limit: int = 500) -> List[Dict]:
    """Estimate next resistance levels above current price."""
    client = _get_client()
    if client is None:
        log("📚 Resistencias estimadas: cliente no disponible")
        return []

    sym = symbol.replace("/", "")

    try:
        ticker = client.futures_symbol_ticker(symbol=sym)
        price = float(ticker["price"])
    except Exception:
        log("📚 Resistencias estimadas: ticker no disponible")
        return []

    # tick size & precision
    tick_size = 1e-06
    price_precision = 6
    try:
        info = client.futures_exchange_info()
        sym_info = next((s for s in info.get("symbols", []) if s.get("symbol") == sym), None)
        if sym_info:
            price_precision = int(sym_info.get("pricePrecision", price_precision))
            for f in sym_info.get("filters", []):
                if f.get("filterType") == "PRICE_FILTER":
                    ts = float(f.get("tickSize", tick_size))
                    if ts > 0:
                        tick_size = ts
                    break
    except Exception:
        pass

    # klines
    try:
        klines = client.futures_klines(symbol=sym, interval=interval, limit=limit)
    except Exception:
        log("📚 Resistencias estimadas: klines no disponibles")
        return []
    if not klines:
        return []

    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    times = [int(k[0]) for k in klines]
    L = R = 3

    swing_highs = []
    swing_lows = []
    for i in range(L, len(highs) - R):
        if highs[i] > max(highs[i - L:i]) and highs[i] >= max(highs[i + 1:i + R + 1]):
            swing_highs.append({"index": i, "price": highs[i], "time": times[i]})
        if lows[i] < min(lows[i - L:i]) and lows[i] <= min(lows[i + 1:i + R + 1]):
            swing_lows.append({"index": i, "price": lows[i], "time": times[i]})

    if not swing_highs:
        log("📚 Resistencias estimadas: sin swings")
        return []

    # clustering
    clusters = []
    for sh in swing_highs:
        placed = False
        for cl in clusters:
            radius = max(3 * tick_size, 0.001 * cl["level"])
            if abs(sh["price"] - cl["level"]) <= radius:
                cl["highs"].append(sh)
                cl["level"] = statistics.median([h["price"] for h in cl["highs"]])
                placed = True
                break
        if not placed:
            clusters.append({"highs": [sh], "level": sh["price"]})

    # pivots diarios
    pivots = {}
    try:
        daily = client.futures_klines(symbol=sym, interval="1d", limit=2)
        if len(daily) >= 2:
            prev = daily[-2]
            H, Lw, C = float(prev[2]), float(prev[3]), float(prev[4])
            P = (H + Lw + C) / 3.0
            pivots = {
                "R1": 2 * P - Lw,
                "R2": P + (H - Lw),
                "R3": H + 2 * (P - Lw),
            }
    except Exception:
        pivots = {}

    # fib extensions
    fib_exts = {}
    if swing_lows and swing_highs:
        last_low = swing_lows[-1]
        prev_highs = [h for h in swing_highs if h["index"] < last_low["index"]]
        if prev_highs:
            last_high = prev_highs[-1]
            diff = last_high["price"] - last_low["price"]
            base = last_low["price"]
            fib_exts = {
                "1.0": base + diff * 1.0,
                "1.272": base + diff * 1.272,
                "1.618": base + diff * 1.618,
            }

    # order book
    asks = []
    try:
        depth = client.futures_order_book(symbol=sym, limit=100)
        asks = [(float(p), float(q)) for p, q in depth.get("asks", [])]
    except Exception:
        asks = []
    median_qty = statistics.median([q for _, q in asks]) if asks else 0

    start_time = times[0]
    end_time = times[-1]
    outputs: List[Dict] = []
    for cl in clusters:
        level = _round_to_tick(cl["level"], tick_size, price_precision)
        if level <= price:
            continue
        touches = len(cl["highs"])
        w_touches = 2.0 * math.log1p(touches)
        cluster_time = max(h["time"] for h in cl["highs"])
        age_norm = (
            (end_time - cluster_time) / (end_time - start_time)
            if end_time != start_time
            else 0
        )
        recency_factor = 1 - age_norm
        w_recency = 1.5 * recency_factor

        reasons = [f"toques={touches}"]
        confluences = 0
        pivot_match = None
        for name, p in pivots.items():
            if p > price and abs(level - p) / level <= 0.0008:
                pivot_match = name
                dist = abs(level - p) / level * 100
                reasons.append(f"{name} a {dist:.2f}%")
                confluences += 1
                break

        fib_match = None
        for name, f_level in fib_exts.items():
            if f_level > price and abs(level - f_level) / level <= 0.0008:
                fib_match = name
                reasons.append(f"fib {name}")
                confluences += 1
                break

        round_match = _is_round(level, price_precision)
        if round_match:
            reasons.append(f"round {_fmt_price(level, price_precision)}")
            confluences += 1

        w_confluence = 0.8 * confluences

        ask_sum_ratio = 0.0
        if asks:
            radius = 2 * tick_size
            liquidity = sum(q for p, q in asks if abs(p - level) <= radius)
            if median_qty > 0:
                ask_sum_ratio = liquidity / median_qty
        w_liquidity = min(1.5, ask_sum_ratio)
        if ask_sum_ratio:
            reasons.append(f"liq x{ask_sum_ratio:.1f}")

        distance_pct = (level - price) / price * 100
        w_distance = -min(1.0, distance_pct / 2.0)
        reasons.append(f"dist {distance_pct:.2f}%")

        score = w_touches + w_recency + w_confluence + w_liquidity + w_distance
        outputs.append(
            {
                "level": level,
                "score": score,
                "reasons": reasons,
                "touches": touches,
                "confluence": {
                    "pivot": pivot_match,
                    "fib": fib_match,
                    "round": round_match,
                    "liquidity_ratio": ask_sum_ratio,
                },
                "distance_pct": distance_pct,
            }
        )

    outputs.sort(key=lambda x: x["score"], reverse=True)
    top3 = outputs[:3]
    if top3:
        log(
            "📚 Resistencias estimadas: "
            + ", ".join(
                [
                    f"{_fmt_price(l['level'], price_precision)} (score {l['score']:.2f})"
                    for l in top3
                ]
            )
        )
    else:
        log("📚 Resistencias estimadas: ninguna")
    return top3
