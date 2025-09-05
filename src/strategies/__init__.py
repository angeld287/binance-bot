from __future__ import annotations

from importlib import util as _importlib_util
from pathlib import Path

from .breakout import BreakoutStrategy

STRATEGY_REGISTRY: dict[str, type] = {}
STRATEGY_REGISTRY["breakout"] = BreakoutStrategy

# Optional: load LiquiditySweepStrategy if present
_liq_path = Path(__file__).with_name("liquidity-sweep").joinpath("__init__.py")
if _liq_path.exists():  # pragma: no cover - optional stub
    spec = _importlib_util.spec_from_file_location("strategies.liquidity_sweep", _liq_path)
    if spec and spec.loader:
        _module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(_module)
        if hasattr(_module, "LiquiditySweepStrategy"):
            STRATEGY_REGISTRY["liquidity-sweep"] = getattr(_module, "LiquiditySweepStrategy")

__all__ = ["STRATEGY_REGISTRY", "BreakoutStrategy"]
