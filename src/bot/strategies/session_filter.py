"""Filtro de sessão: restringe a estratégia a faixas de horário.

O estudo do WIN mostrou que a volatilidade das 9h–11h é o dobro da
tarde. Setups calibrados para movimento precisam da manhã; setups de
reversão em range tendem a preferir a tarde. Este filtro deixa isso
explícito e testável, em vez de embutido na estratégia.
"""

from datetime import time

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType


class SessionFilteredStrategy(BaseStrategy):
    def __init__(self, inner: BaseStrategy, start: time, end: time):
        super().__init__(inner.params)
        self.inner = inner
        self.mode = inner.mode
        self.start = start
        self.end = end

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        now = candles.index[-1].time()
        if not (self.start <= now <= self.end):
            return Signal(symbol=symbol, type=SignalType.HOLD)
        return self.inner.generate_signal(symbol, candles)
