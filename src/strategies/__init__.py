# strategies/__init__.py
import os
import importlib
from typing import Any, Tuple, Type

# clave -> (módulo, nombre_de_clase)
_SUPPORTED = {
    "breakout":    ("breakout", "FuturesBot"),
    "random_open": ("random_open", "FuturesBot"),
    "random":      ("random_open", "FuturesBot"),  # alias
}

def _strategy_name() -> str:
    return os.getenv("STRATEGY_NAME", "breakout").strip().lower()

def _load() -> Tuple[object, Type]:
    mod_name, cls_name = _SUPPORTED.get(_strategy_name(), ("breakout", "FuturesBot"))
    mod = importlib.import_module(f"{__name__}.{mod_name}")
    cls = getattr(mod, cls_name)
    return mod, cls

# === Señales (igual que ya tienes) ===
def generate_signal(*args: Any, **kwargs: Any):
    mod, _ = _load()
    fn = getattr(mod, "generate_signal")
    return fn(*args, **kwargs)


# === Iteración ===
def _run_iteration(*args: Any, **kwargs: Any):
    """Delegates to the underlying strategy's iteration helper."""
    mod, _ = _load()
    fn = getattr(mod, "_run_iteration")
    return fn(*args, **kwargs)

# === Factory de la clase ===
def bot_class() -> Type:
    """Devuelve la clase (no instancia) para la estrategia activa."""
    _, cls = _load()
    return cls

def create_bot(*args: Any, **kwargs: Any):
    """Crea la instancia del bot de la estrategia activa (runtime)."""
    return bot_class()(*args, **kwargs)

# === Alias opcional del nombre de clase ===
# Permite: `from strategies import FuturesBot`
# Nota: se resuelve una sola vez al momento del import.
def __getattr__(name: str):
    if name == "FuturesBot":
        return bot_class()
    raise AttributeError(name)

__all__ = ["generate_signal", "bot_class", "create_bot", "FuturesBot", "_run_iteration"]
