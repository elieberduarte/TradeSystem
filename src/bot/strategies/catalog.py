"""Catálogo de estratégias disponíveis, indexadas pelo nome do config."""

from src.bot.strategies.base import BaseStrategy
from src.bot.strategies.ema_cross import EmaCrossStrategy
from src.bot.strategies.opening_range import OpeningRangeStrategy

REGISTRY: dict[str, type[BaseStrategy]] = {
    "ema_cross": EmaCrossStrategy,
    "opening_range": OpeningRangeStrategy,
}


def get_strategy(name: str, params: dict | None = None) -> BaseStrategy:
    if name not in REGISTRY:
        raise ValueError(f"Estratégia desconhecida: '{name}' (disponíveis: {list(REGISTRY)})")
    return REGISTRY[name](params)
