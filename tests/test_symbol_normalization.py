import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from core.ports.settings import get_symbol

class DummySettings:
    def __init__(self, symbol):
        self._symbol = symbol

    def get(self, key, default=None):
        if key == "SYMBOL":
            return self._symbol
        return default


def test_get_symbol_normalizes_various_inputs():
    settings = DummySettings(" sol/usdt ")
    assert get_symbol(settings) == "SOLUSDT"
    settings = DummySettings("sol-usdt")
    assert get_symbol(settings) == "SOLUSDT"
    settings = DummySettings("SOLUSDT")
    assert get_symbol(settings) == "SOLUSDT"


def test_get_symbol_defaults_when_empty():
    settings = DummySettings("__--")
    assert get_symbol(settings) == "BTCUSDT"
    settings = DummySettings("")
    assert get_symbol(settings) == "BTCUSDT"
