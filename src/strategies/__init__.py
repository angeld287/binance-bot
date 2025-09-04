import importlib
import os
from typing import Type, Tuple

_supported = {
    "breakout": {
        "aliases": ["breakout"],
        "module": "strategies.breakout",
        "class": "FuturesBot",
    },
    "random_open": {
        "aliases": ["random_open", "random"],
        "module": "strategies.random_open",
        "class": "FuturesBot",
    },
}

_supported.update({
    "liquidity_sweep": {
        "aliases": ["liquidity_sweep", "liquidsweep", "LiquidSweep"],
        "module": "strategies.liquidity_sweep.strategy",
        "class": "LiquiditySweepStrategy",
    }
})


def _normalize(name: str) -> str:
    return name.replace(" ", "").lower()


def _get_entry(name: str) -> Tuple[object, Type]:
    name_n = _normalize(name or os.getenv("STRATEGY", "breakout"))
    for key, cfg in _supported.items():
        aliases = [key] + cfg.get("aliases", [])
        if name_n in [_normalize(a) for a in aliases]:
            mod = importlib.import_module(cfg["module"])
            cls = getattr(mod, cfg["class"])
            return mod, cls
    raise ValueError(f"Unknown strategy: {name}")


def get_strategy_class(name: str | None = None) -> Type:
    _, cls = _get_entry(name or os.getenv("STRATEGY", "breakout"))
    return cls


def generateSignal(context, name: str | None = None):
    mod, _ = _get_entry(name or os.getenv("STRATEGY", "breakout"))
    fn = getattr(mod, "generateSignal")
    return fn(context)


__all__ = ["get_strategy_class", "generateSignal"]
