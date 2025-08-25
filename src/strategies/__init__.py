import os
from typing import Any
from . import breakout as _breakout
from . import random_open as _random

_SUPPORTED = {
    "breakout": _breakout,
    "random_open": _random,
    "random": _random,  # alias opcional
}

def _strategy_name() -> str:
    return os.getenv("STRATEGY_NAME", "breakout").strip().lower()

def generate_signal(*args: Any, **kwargs: Any):
    """
    Punto único de entrada de estrategia.
    Despacha en cada llamada según STRATEGY_NAME (runtime, no build).
    """
    mod = _SUPPORTED.get(_strategy_name(), _breakout)
    fn = getattr(mod, "generate_signal")
    return fn(*args, **kwargs)

__all__ = ["generate_signal"]
