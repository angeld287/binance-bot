"""Breakout-based trading strategy utilities."""

from analysis.pattern_detection import detect_patterns


ANALYSIS_WINDOW = 12


def _last_swing_high(ohlcv, window=ANALYSIS_WINDOW):
    highs = [c[2] for c in ohlcv]
    for i in range(len(highs) - window - 1, window, -1):
        local = highs[i - window : i + window + 1]
        if highs[i] == max(local):
            return highs[i]
    return None


def _last_swing_low(ohlcv, window=ANALYSIS_WINDOW):
    lows = [c[3] for c in ohlcv]
    for i in range(len(lows) - window - 1, window, -1):
        local = lows[i - window : i + window + 1]
        if lows[i] == min(local):
            return lows[i]
    return None


def generate_signal(exchange, symbol, window=ANALYSIS_WINDOW):
    """Busca rupturas de los últimos máximos o mínimos en 15m y 30m."""
    for tf in ["15m", "30m"]:
        try:
            klines = exchange.futures_klines(
                symbol=symbol.replace("/", ""), interval=tf, limit=50
            )
            ohlcv = [
                [k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
                for k in klines
            ]
            if len(ohlcv) < window * 2 + 2:
                continue
            price = ohlcv[-1][4]
            last_high = _last_swing_high(ohlcv[:-1], window=window)
            last_low = _last_swing_low(ohlcv[:-1], window=window)
            if last_high and price > last_high:
                patterns = detect_patterns(ohlcv)
                price_range = (last_low if last_low else price * 0.99, last_high)
                return "buy", last_high, patterns, price_range
            if last_low and price < last_low:
                patterns = detect_patterns(ohlcv)
                price_range = (last_low, last_high if last_high else price * 1.01)
                return "sell", last_low, patterns, price_range
        except Exception:
            continue
    return None, None, [], (None, None)

