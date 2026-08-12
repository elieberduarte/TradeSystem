"""Rompimento de canal de Donchian (swing, candles diários).

Seguidor de tendência clássico: compra quando o preço rompe a máxima
dos últimos N pregões, vende no rompimento da mínima. É a família das
"tartarugas" e o teste mais direto de continuação no horizonte diário
— onde, ao contrário do intradiário, o regime se mostrou persistente
(94,7% de permanência em 1 dia, 68,8% em 10).
"""

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType
from src.bot.strategies.swing_reversion import atr

DEFAULTS = {
    "channel": 20,
    "stop_atr": 2.0,
    "rr": 3.0,
    "atr_period": 14,
}


class DonchianStrategy(BaseStrategy):
    mode = "swing_trade"

    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        if len(candles) < p["channel"] + p["atr_period"] + 2:
            return hold

        # Canal formado pelos candles ANTERIORES ao atual
        window = candles.iloc[-(p["channel"] + 1) : -1]
        upper, lower = float(window["high"].max()), float(window["low"].min())
        close_now = float(candles["close"].iloc[-1])

        stop_distance = float(atr(candles, p["atr_period"]).iloc[-1]) * p["stop_atr"]
        if stop_distance <= 0:
            return hold

        if close_now > upper:
            return Signal(
                symbol=symbol, type=SignalType.BUY, entry_price=close_now,
                stop_loss=close_now - stop_distance,
                take_profit=close_now + p["rr"] * stop_distance,
            )
        if close_now < lower:
            return Signal(
                symbol=symbol, type=SignalType.SELL, entry_price=close_now,
                stop_loss=close_now + stop_distance,
                take_profit=close_now - p["rr"] * stop_distance,
            )
        return hold
