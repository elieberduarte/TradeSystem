"""Filtro de regime: só deixa a estratégia operar no mercado certo.

Envolve qualquer estratégia e veta os sinais quando o regime vigente
não está na lista de regimes em que ela funciona — a defesa contra
"operar setup de tendência num dia lateral".
"""

import pandas as pd

from src.bot.analysis.regime import Regime, classify
from src.bot.strategies.base import BaseStrategy, Signal, SignalType


class RegimeFilteredStrategy(BaseStrategy):
    def __init__(
        self,
        inner: BaseStrategy,
        allowed: set[Regime],
        adx_period: int = 14,
        trend_threshold: float = 25.0,
    ):
        super().__init__(inner.params)
        self.inner = inner
        self.allowed = allowed
        self.adx_period = adx_period
        self.trend_threshold = trend_threshold

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        regime = classify(candles, self.adx_period, self.trend_threshold)
        if regime not in self.allowed:
            return Signal(symbol=symbol, type=SignalType.HOLD)
        return self.inner.generate_signal(symbol, candles)
