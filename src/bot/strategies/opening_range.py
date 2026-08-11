"""Rompimento da abertura (Opening Range Breakout).

Setup clássico de day trade: o range dos primeiros candles do dia define
suporte e resistência iniciais; o rompimento desse range a favor indica
a direção do dia. Entrada no candle que FECHA fora do range (evita
violinos de pavio), stop no lado oposto do range.
"""

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType

DEFAULTS = {
    # Candles que formam o range de abertura (3 x 5min = 15 minutos)
    "range_bars": 3,
    "rr": 1.5,
}


class OpeningRangeStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)

        today = candles.index[-1].normalize()
        day = candles[candles.index.normalize() == today]
        # Range ainda em formação, ou dia sem candle anterior ao atual
        if len(day) <= p["range_bars"]:
            return hold

        opening = day.iloc[: p["range_bars"]]
        range_high = float(opening["high"].max())
        range_low = float(opening["low"].min())

        close_now = float(day["close"].iloc[-1])
        close_prev = float(day["close"].iloc[-2])

        # Só o candle do rompimento gera sinal (anterior dentro, atual fora)
        if close_prev <= range_high and close_now > range_high:
            entry, stop = close_now, range_low
            target = entry + p["rr"] * (entry - stop)
            return Signal(
                symbol=symbol,
                type=SignalType.BUY,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
            )
        if close_prev >= range_low and close_now < range_low:
            entry, stop = close_now, range_high
            target = entry - p["rr"] * (stop - entry)
            return Signal(
                symbol=symbol,
                type=SignalType.SELL,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
            )
        return hold
