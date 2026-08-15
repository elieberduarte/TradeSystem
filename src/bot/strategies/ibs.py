"""IBS — Internal Bar Strength (swing, candles diários).

Onde o candle fechou dentro do próprio range: IBS = (close − low) /
(high − low). Perto de 0 = fechou na mínima (pressão vendedora
exaurida); perto de 1 = fechou na máxima.

É a família de reversão à média com maior taxa de acerto documentada
em índices — o que interessa quando o objetivo é uma curva mais suave,
e não o maior lucro possível. A saída não é um alvo fixo: é a própria
condição contrária, o que evita o problema de escolher alvo arbitrário.
"""

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType
from src.bot.strategies.swing_reversion import atr

DEFAULTS = {
    "entry_low": 0.15,   # abaixo disso, compra
    "entry_high": 0.85,  # acima disso, vende
    # Alvo em fração do range recente (mais próximo = mais acerto)
    "target_atr": 1.0,
    "stop_atr": 2.0,
    "atr_period": 14,
    # 0 desativa; senão só compra acima da média de N períodos
    "trend_filter": 0,
}


def ibs(candles: pd.DataFrame) -> pd.Series:
    span = candles["high"] - candles["low"]
    return ((candles["close"] - candles["low"]) / span.where(span > 0)).fillna(0.5)


class IbsStrategy(BaseStrategy):
    mode = "swing_trade"

    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        needed = max(p["atr_period"], p["trend_filter"]) + 5
        if len(candles) < needed:
            return hold

        value = float(ibs(candles).iloc[-1])
        entry = float(candles["close"].iloc[-1])
        volatility = float(atr(candles, p["atr_period"]).iloc[-1])
        if volatility <= 0:
            return hold

        above_trend = True
        below_trend = True
        if p["trend_filter"]:
            trend = float(candles["close"].rolling(p["trend_filter"]).mean().iloc[-1])
            if pd.isna(trend):
                return hold
            above_trend, below_trend = entry > trend, entry < trend

        trend_note = (f", com preço acima da média de {p['trend_filter']} pregões "
                      f"(filtro de tendência)") if p["trend_filter"] else ""
        if value <= p["entry_low"] and above_trend:
            return Signal(
                symbol=symbol, type=SignalType.BUY, entry_price=entry,
                stop_loss=entry - p["stop_atr"] * volatility,
                take_profit=entry + p["target_atr"] * volatility,
                reason=(f"IBS {value:.2f}: fechou colado na MÍNIMA do próprio candle "
                        f"(limiar {p['entry_low']}) — pressão vendedora exaurida, aposta "
                        f"em reversão de curtíssimo prazo{trend_note}; só preço, sem volume"),
            )
        if value >= p["entry_high"] and below_trend:
            return Signal(
                symbol=symbol, type=SignalType.SELL, entry_price=entry,
                stop_loss=entry + p["stop_atr"] * volatility,
                take_profit=entry - p["target_atr"] * volatility,
                reason=(f"IBS {value:.2f}: fechou colado na MÁXIMA do próprio candle "
                        f"(limiar {p['entry_high']}) — pressão compradora exaurida, aposta "
                        f"em reversão de curtíssimo prazo{trend_note}; só preço, sem volume"),
            )
        return hold
