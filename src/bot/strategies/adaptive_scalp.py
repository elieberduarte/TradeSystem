"""Scalp com alvo, stop e filtro proporcionais à volatilidade.

Corrige o defeito das primeiras estratégias de scalping, que usavam
alvo fixo em pontos: em mercado grande o alvo era pequeno demais para
o movimento, em mercado pequeno era grande demais para ser atingido.

Três consequências de amarrar tudo ao ATR:

1. Alvo e stop acompanham o tamanho do mercado.
2. A mão se ajusta sozinha (o motor divide risco pela distância do
   stop): mercado grande → stop maior → menos contratos.
3. O filtro de fricção passa a existir. A fricção é FIXA (~12,5 pts);
   se o ATR está baixo, ela come o alvo. Só operar acima de um piso de
   volatilidade é o que torna o alvo curto viável — ou revela que não é.
"""

import pandas as pd

from src.bot.analysis.volatility import atr, market_size
from src.bot.strategies.base import BaseStrategy, Signal, SignalType


class AdaptiveScalpStrategy(BaseStrategy):
    DEFAULTS = {
        "atr_period": 14,
        # Alvo e stop em múltiplos do ATR corrente
        "target_atr": 1.0,
        "stop_atr": 1.0,
        # Piso de volatilidade: só opera se o ATR valer ao menos N vezes
        # a fricção. Abaixo disso o custo domina o alvo.
        "min_atr_friction": 3.0,
        "friction_points": 12.5,
        # Faixa de "tamanho de mercado" relativo ao normal daquele
        # horário (ATR dessazonalizado). Fora dela o bot não opera:
        # mercado pequeno demais é dominado pela fricção; grande demais
        # costuma ser notícia/choque, onde stop é furado e o slippage
        # real explode. 0 no teto desativa o limite superior.
        "min_size": 0.0,
        "max_size": 0.0,
        # Gatilho: movimento recente em múltiplos do ATR
        "lookback": 5,
        "trigger_atr": 1.5,
        # "fade" aposta na devolução; "follow" aposta na continuação
        "direction": "fade",
    }

    def __init__(self, params: dict | None = None):
        super().__init__({**self.DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        if len(candles) < p["atr_period"] + p["lookback"] + 2:
            return hold

        volatility = float(atr(candles, p["atr_period"]).iloc[-1])
        if volatility <= 0:
            return hold

        # Filtro de fricção: mercado pequeno demais não paga o custo
        if volatility < p["min_atr_friction"] * p["friction_points"]:
            return hold

        # Faixa de tamanho de mercado, medida contra o normal do horário
        if p["min_size"] or p["max_size"]:
            size = float(market_size(candles, p["atr_period"]).iloc[-1])
            if p["min_size"] and size < p["min_size"]:
                return hold
            if p["max_size"] and size > p["max_size"]:
                return hold

        close = float(candles["close"].iloc[-1])
        past = float(candles["close"].iloc[-1 - p["lookback"]])
        move = close - past
        if abs(move) < p["trigger_atr"] * volatility:
            return hold

        target = p["target_atr"] * volatility
        stop = p["stop_atr"] * volatility

        # Movimento de queda: "fade" compra a devolução, "follow" segue a queda
        bullish = move < 0 if p["direction"] == "fade" else move > 0
        if bullish:
            return Signal(
                symbol=symbol, type=SignalType.BUY, entry_price=close,
                stop_loss=close - stop, take_profit=close + target,
            )
        return Signal(
            symbol=symbol, type=SignalType.SELL, entry_price=close,
            stop_loss=close + stop, take_profit=close - target,
        )
