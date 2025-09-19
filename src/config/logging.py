import logging
import sys
from typing import Literal


def setup_logging(level: str = "INFO", mode: Literal["plain", "json"] = "plain") -> None:
    """Configure basic logging for the application."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if mode == "json":
        formatter = logging.Formatter(
            '{"level":"%(levelname)s","time":"%(asctime)s","name":"%(name)s","msg":"%(message)s"}',
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    else:
        formatter = logging.Formatter("%(levelname)s - %(message)s")

    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)
