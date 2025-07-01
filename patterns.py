import numpy as np


def _local_maxima(values, order=2):
    maxima = []
    for i in range(order, len(values) - order):
        window = values[i - order : i + order + 1]
        if values[i] == max(window):
            maxima.append(i)
    return maxima


def _local_minima(values, order=2):
    minima = []
    for i in range(order, len(values) - order):
        window = values[i - order : i + order + 1]
        if values[i] == min(window):
            minima.append(i)
    return minima


# --- Chart pattern detectors -------------------------------------------------

def detect_double_top(ohlcv, tol=0.005):
    highs = [c[2] for c in ohlcv]
    peaks = _local_maxima(highs)
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        h1, h2 = highs[p1], highs[p2]
        if abs(h1 - h2) / ((h1 + h2) / 2) < tol and p2 - p1 > 1:
            return True
    return False


def detect_double_bottom(ohlcv, tol=0.005):
    lows = [c[3] for c in ohlcv]
    troughs = _local_minima(lows)
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        l1, l2 = lows[t1], lows[t2]
        if abs(l1 - l2) / ((l1 + l2) / 2) < tol and t2 - t1 > 1:
            return True
    return False


def detect_triple_top(ohlcv, tol=0.005):
    highs = [c[2] for c in ohlcv]
    peaks = _local_maxima(highs)
    if len(peaks) >= 3:
        p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
        h = [highs[p] for p in (p1, p2, p3)]
        if max(h) - min(h) <= max(h) * tol:
            return True
    return False


def detect_triple_bottom(ohlcv, tol=0.005):
    lows = [c[3] for c in ohlcv]
    troughs = _local_minima(lows)
    if len(troughs) >= 3:
        t1, t2, t3 = troughs[-3], troughs[-2], troughs[-1]
        l = [lows[t] for t in (t1, t2, t3)]
        if max(l) - min(l) <= max(l) * tol:
            return True
    return False


def detect_head_shoulders(ohlcv, tol=0.01):
    highs = [c[2] for c in ohlcv]
    peaks = _local_maxima(highs)
    if len(peaks) >= 3:
        left, head, right = peaks[-3], peaks[-2], peaks[-1]
        if highs[head] > highs[left] and highs[head] > highs[right]:
            if abs(highs[left] - highs[right]) / highs[head] < tol * 2:
                return True
    return False


def detect_inverted_head_shoulders(ohlcv, tol=0.01):
    lows = [c[3] for c in ohlcv]
    troughs = _local_minima(lows)
    if len(troughs) >= 3:
        left, head, right = troughs[-3], troughs[-2], troughs[-1]
        if lows[head] < lows[left] and lows[head] < lows[right]:
            if abs(lows[left] - lows[right]) / abs(lows[head]) < tol * 2:
                return True
    return False


def _slopes(values, length):
    x = np.arange(len(values))
    y = np.array(values)
    slope, _ = np.polyfit(x[-length:], y[-length:], 1)
    return slope


def detect_symmetrical_triangle(ohlcv, length=10, tol=1e-4):
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]
    hi_slope = _slopes(highs, length)
    lo_slope = _slopes(lows, length)
    if hi_slope < -tol and lo_slope > tol:
        return True
    return False


def detect_ascending_triangle(ohlcv, length=10, tol=1e-4):
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]
    hi_slope = _slopes(highs, length)
    lo_slope = _slopes(lows, length)
    if abs(hi_slope) <= tol and lo_slope > tol:
        return True
    return False


def detect_descending_triangle(ohlcv, length=10, tol=1e-4):
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]
    hi_slope = _slopes(highs, length)
    lo_slope = _slopes(lows, length)
    if abs(lo_slope) <= tol and hi_slope < -tol:
        return True
    return False


def detect_rectangle(ohlcv, length=10, tol=1e-4):
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]
    hi_slope = _slopes(highs, length)
    lo_slope = _slopes(lows, length)
    if abs(hi_slope) <= tol and abs(lo_slope) <= tol:
        return True
    return False


def detect_rising_wedge(ohlcv, length=10):
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]
    hi_slope = _slopes(highs, length)
    lo_slope = _slopes(lows, length)
    if hi_slope > 0 and lo_slope > 0 and hi_slope < lo_slope:
        return True
    return False


def detect_falling_wedge(ohlcv, length=10):
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]
    hi_slope = _slopes(highs, length)
    lo_slope = _slopes(lows, length)
    if hi_slope < 0 and lo_slope < 0 and hi_slope > lo_slope:
        return True
    return False


def detect_broadening_triangle(ohlcv, length=10):
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]
    hi_slope = _slopes(highs, length)
    lo_slope = _slopes(lows, length)
    if hi_slope > 0 and lo_slope < 0:
        return True
    return False


def detect_flag(ohlcv, length=10, tol=0.02):
    closes = [c[4] for c in ohlcv]
    if len(closes) < length + 1:
        return False
    move = abs(closes[-length - 1] - closes[-length - 2]) / closes[-length - 2]
    if move > tol:
        slope = _slopes(closes, length)
        if abs(slope) < tol / 2:
            return True
    return False


def detect_cup_handle(ohlcv, length=40):
    closes = [c[4] for c in ohlcv]
    if len(closes) < length:
        return False
    segment = closes[-length:]
    mid = int(len(segment) / 2)
    left = segment[:mid]
    right = segment[mid:]
    if min(left) == min(segment) and min(right) >= min(segment):
        if segment[-1] <= segment[0] * 1.05:
            return True
    return False


def detect_patterns(ohlcv):
    patterns = []
    if detect_head_shoulders(ohlcv):
        patterns.append("Head & Shoulders")
    if detect_inverted_head_shoulders(ohlcv):
        patterns.append("Inverted Head & Shoulders")
    if detect_double_top(ohlcv):
        patterns.append("Double Top")
    if detect_double_bottom(ohlcv):
        patterns.append("Double Bottom")
    if detect_triple_top(ohlcv):
        patterns.append("Triple Top")
    if detect_triple_bottom(ohlcv):
        patterns.append("Triple Bottom")
    if detect_falling_wedge(ohlcv):
        patterns.append("Falling Wedge")
    if detect_broadening_triangle(ohlcv):
        patterns.append("Broadening Triangle")
    if detect_symmetrical_triangle(ohlcv):
        patterns.append("Symmetrical Triangle")
    if detect_rising_wedge(ohlcv):
        patterns.append("Rising Wedge")
    if detect_rectangle(ohlcv):
        patterns.append("Rectangle")
    if detect_flag(ohlcv):
        patterns.append("Flag")
    if detect_descending_triangle(ohlcv):
        patterns.append("Descending Triangle")
    if detect_ascending_triangle(ohlcv):
        patterns.append("Ascending Triangle")
    if detect_cup_handle(ohlcv):
        patterns.append("Cup & Handle")
    return patterns
