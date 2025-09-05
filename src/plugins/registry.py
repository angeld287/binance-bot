"""Simple plugin registry for strategies, brokers and data sources."""

from __future__ import annotations

from typing import Dict, Type

from core.ports.strategy import Strategy
from core.ports.broker import BrokerPort
from core.ports.market_data import MarketDataPort


STRATEGIES: Dict[str, Type[Strategy]] = {}
BROKERS: Dict[str, Type[BrokerPort]] = {}
DATASOURCES: Dict[str, Type[MarketDataPort]] = {}


def register_strategy(name: str, cls: Type[Strategy]) -> None:
    STRATEGIES[name] = cls


def register_broker(name: str, cls: Type[BrokerPort]) -> None:
    BROKERS[name] = cls


def register_datasource(name: str, cls: Type[MarketDataPort]) -> None:
    DATASOURCES[name] = cls
