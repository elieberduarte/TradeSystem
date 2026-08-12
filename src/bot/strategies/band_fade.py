"""Reversão à média por bandas — "fechou fora, fechou dentro" (FFFD).

Setup de reversão mais difundido no mercado brasileiro (conhecido como
Bollinger FFFD / "estratégia da Paula"): o preço fecha FORA da banda
(exagero) e o candle seguinte fecha DENTRO dela (rejeição) — entra a
favor da volta para a média.

Suporta banda de desvio-padrão (Bollinger) ou de ATR (Keltner). O ATR
tende a se adaptar melhor à sazonalidade intradiária, em que a
volatilidade da manhã é o dobro da tarde.
"""

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType

DEFAULTS = {
    "period": 20,
    "mult": 2.0,
    # "bollinger" = desvio-padrão do preço típico | "keltner" = ATR
    "band": "bollinger",
    # Alvo: "mid" = média central | número = múltiplo do stop
    "target": "mid",
    # Stop = range do candle de gatilho x este fator
    "stop_mult": 1.0,
}


def bands(candles: pd.DataFrame, period: int, mult: float, kind: str):
    """Retorna (média, banda superior, banda inferior)."""
    if kind == "keltner":
        mid = candles["close"].rolling(period).mean()
        prev_close = candles["close"].shift(1)
        true_range = pd.concat(
            [
                candles["high"] - candles["low"],
                (candles["high"] - prev_close).abs(),
                (candles["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        width = true_range.rolling(period).mean() * mult
    else:
        typical = (candles["high"] + candles["low"] + candles["close"]) / 3
        mid = typical.rolling(period).mean()
        width = typical.rolling(period).std(ddof=0) * mult
    return mid, mid + width, mid - width


class BandFadeStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        if len(candles) < p["period"] + 2:
            return hold

        mid, upper, lower = bands(candles, p["period"], p["mult"], p["band"])
        if pd.isna(mid.iloc[-1]) or pd.isna(mid.iloc[-2]):
            return hold

        close_now, close_prev = candles["close"].iloc[-1], candles["close"].iloc[-2]
        candle = candles.iloc[-1]
        candle_range = float(candle["high"] - candle["low"])
        if candle_range <= 0:
            return hold
        stop_distance = candle_range * p["stop_mult"]
        entry = float(close_now)
        center = float(mid.iloc[-1])

        # Compra: fechou abaixo da banda inferior e voltou para dentro
        if close_prev < lower.iloc[-2] and close_now > lower.iloc[-1]:
            stop = entry - stop_distance
            target = center if p["target"] == "mid" else entry + float(p["target"]) * stop_distance
            if target <= entry:
                return hold
            return Signal(
                symbol=symbol, type=SignalType.BUY,
                entry_price=entry, stop_loss=stop, take_profit=target,
            )

        # Venda: fechou acima da banda superior e voltou para dentro
        if close_prev > upper.iloc[-2] and close_now < upper.iloc[-1]:
            stop = entry + stop_distance
            target = center if p["target"] == "mid" else entry - float(p["target"]) * stop_distance
            if target >= entry:
                return hold
            return Signal(
                symbol=symbol, type=SignalType.SELL,
                entry_price=entry, stop_loss=stop, take_profit=target,
            )

        return hold
