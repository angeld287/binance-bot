import time
from typing import Dict, List

from resistance_levels import next_resistances
from support_levels import next_supports


def get_sr_levels(symbol: str, timeframe: str) -> Dict[str, List[float]]:
    """Obtiene los 3 principales niveles de soporte y resistencia."""
    sup = next_supports(symbol, interval=timeframe)
    res = next_resistances(symbol, interval=timeframe)
    s_vals = [s.get("level") for s in sup][:3]
    r_vals = [r.get("level") for r in res][:3]
    return {"S": s_vals, "R": r_vals, "asof": int(time.time() * 1000)}
