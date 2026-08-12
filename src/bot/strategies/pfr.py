"""PFR — Preço de Fechamento de Reversão.

Padrão de exaustão do cânone brasileiro (popularizado por Stormer):
o candle faz mínima menor que as duas anteriores mas fecha acima do
fechamento anterior — quem vendeu na baixa foi rejeitado.

Regra original: entrada no rompimento da máxima do candle de reversão,
stop na mínima dele, alvo em 1x o risco. Aqui a entrada é no fechamento
do candle (o motor decide a cada barra fechada), o que é conservador:
não assume que o rompimento aconteceu.
"""

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType

DEFAULTS = {"rr": 1.0}


class PfrStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        if len(candles) < 3:
            return hold

        low0, low1, low2 = candles["low"].iloc[-1], candles["low"].iloc[-2], candles["low"].iloc[-3]
        high0, high1, high2 = candles["high"].iloc[-1], candles["high"].iloc[-2], candles["high"].iloc[-3]
        close0, close1 = candles["close"].iloc[-1], candles["close"].iloc[-2]
        entry = float(close0)
        rr = self.params["rr"]

        # PFR de compra: mínima menor que as duas anteriores, fecha acima
        if low0 < low1 and low0 < low2 and close0 > close1:
            stop = float(low0)
            if stop >= entry:
                return hold
            return Signal(
                symbol=symbol, type=SignalType.BUY, entry_price=entry,
                stop_loss=stop, take_profit=entry + rr * (entry - stop),
            )

        # PFR de venda: máxima maior que as duas anteriores, fecha abaixo
        if high0 > high1 and high0 > high2 and close0 < close1:
            stop = float(high0)
            if stop <= entry:
                return hold
            return Signal(
                symbol=symbol, type=SignalType.SELL, entry_price=entry,
                stop_loss=stop, take_profit=entry - rr * (stop - entry),
            )

        return hold
