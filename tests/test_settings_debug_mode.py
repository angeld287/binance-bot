import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from core.ports.settings import get_debug_mode


class DummySettings:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, key, default=None):
        return self._values.get(key, default)


def test_get_debug_mode_accepts_truthy_strings():
    for value in ["1", "true", "TRUE", "True"]:
        settings = DummySettings({"DEBUG_MODE": value})
        assert get_debug_mode(settings) is True


def test_get_debug_mode_falsey_values_and_default():
    falsey_cases = ["0", "false", None]
    for value in falsey_cases:
        values = {"DEBUG_MODE": value} if value is not None else {}
        settings = DummySettings(values)
        assert get_debug_mode(settings) is False
