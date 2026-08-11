"""Interface base para estratégias de trading."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    """Sinal gerado por uma estratégia para um ativo."""

    symbol: str
    type: SignalType
    # Preço sugerido de entrada (None = a mercado)
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    # Confiança do sinal entre 0 e 1, usada pelo risk manager no sizing
    confidence: float = 1.0


class BaseStrategy(ABC):
    """Toda estratégia recebe candles e devolve um sinal.

    A estratégia não conhece corretora nem executa ordens — apenas analisa
    dados e opina. Execução e risco são responsabilidade de outros módulos.
    """

    # day_trade = posições zeradas no fim do pregão | swing_trade = overnight
    mode: str = "day_trade"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    @abstractmethod
    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        """Analisa os candles e retorna um sinal.

        `candles` deve conter as colunas: open, high, low, close, volume,
        indexadas por timestamp em ordem crescente.
        """
        raise NotImplementedError
