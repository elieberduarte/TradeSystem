"""Estratégia de cruzamento de médias móveis exponenciais (EMA).

Compra quando a EMA rápida cruza a lenta para cima; vende quando cruza
para baixo. Filtro opcional de tendência: só opera a favor da EMA longa.
Stop no extremo recente; alvo em múltiplo do risco (relação R:R).
"""

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType

DEFAULTS = {
    "fast": 9,
    "slow": 21,
    # 0 desativa o filtro de tendência
    "trend": 80,
    # Candles olhados para trás na definição do stop
    "stop_lookback": 10,
    # Alvo = risco x rr
    "rr": 2.0,
}


class EmaCrossStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        min_bars = max(p["slow"], p["trend"], p["stop_lookback"]) + 2
        if len(candles) < min_bars:
            return Signal(symbol=symbol, type=SignalType.HOLD)

        close = candles["close"]
        fast = close.ewm(span=p["fast"], adjust=False).mean()
        slow = close.ewm(span=p["slow"], adjust=False).mean()

        crossed_up = fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
        crossed_down = fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]
        if not crossed_up and not crossed_down:
            return Signal(symbol=symbol, type=SignalType.HOLD)

        if p["trend"]:
            trend = close.ewm(span=p["trend"], adjust=False).mean()
            if crossed_up and close.iloc[-1] < trend.iloc[-1]:
                return Signal(symbol=symbol, type=SignalType.HOLD)
            if crossed_down and close.iloc[-1] > trend.iloc[-1]:
                return Signal(symbol=symbol, type=SignalType.HOLD)

        entry = float(close.iloc[-1])
        recent = candles.iloc[-p["stop_lookback"] :]
        trend_note = f", com preço acima da EMA {p['trend']} (tendência)" if p["trend"] else ""
        if crossed_up:
            stop = float(recent["low"].min())
            if stop >= entry:
                return Signal(symbol=symbol, type=SignalType.HOLD)
            target = entry + p["rr"] * (entry - stop)
            return Signal(
                symbol=symbol,
                type=SignalType.BUY,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
                reason=(f"EMA {p['fast']} cruzou para CIMA da EMA {p['slow']}{trend_note}; "
                        f"stop na mínima dos últimos {p['stop_lookback']} candles"),
            )

        stop = float(recent["high"].max())
        if stop <= entry:
            return Signal(symbol=symbol, type=SignalType.HOLD)
        target = entry - p["rr"] * (stop - entry)
        return Signal(
            symbol=symbol,
            type=SignalType.SELL,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            reason=(f"EMA {p['fast']} cruzou para BAIXO da EMA {p['slow']}"
                    + (f", com preço abaixo da EMA {p['trend']} (tendência)" if p["trend"] else "")
                    + f"; stop na máxima dos últimos {p['stop_lookback']} candles"),
        )
