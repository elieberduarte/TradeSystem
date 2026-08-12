"""Estratégias de scalping para execução passiva (ordem limitada).

Desenhadas para o único regime onde a aritmética fecha no WIN: alvo
curto (~20 pontos), entrada sem cruzar o spread, posição de poucos
minutos. Todas geram sinal com preço de referência; o motor coloca a
ordem limitada atrás do preço e espera ser procurado.
"""

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType


class MicroFadeStrategy(BaseStrategy):
    """Devolve o movimento curto: compra após queda rápida, vende após alta.

    A lógica passiva por trás: um movimento abrupto de poucos minutos
    costuma ser desequilíbrio de fluxo, não informação. Quem provê
    liquidez contra ele é pago pelo serviço — desde que o alvo seja
    curto e a posição não vire aposta direcional.
    """

    DEFAULTS = {
        "lookback": 5,       # minutos do movimento avaliado
        "min_move": 60.0,    # pontos mínimos para considerar exagero
        "target": 20.0,      # alvo em pontos
        "stop": 40.0,        # stop em pontos
    }

    def __init__(self, params: dict | None = None):
        super().__init__({**self.DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        if len(candles) < p["lookback"] + 2:
            return hold

        close = float(candles["close"].iloc[-1])
        past = float(candles["close"].iloc[-1 - p["lookback"]])
        move = close - past
        if abs(move) < p["min_move"]:
            return hold

        if move < 0:  # caiu rápido → compra a devolução
            return Signal(
                symbol=symbol, type=SignalType.BUY, entry_price=close,
                stop_loss=close - p["stop"], take_profit=close + p["target"],
            )
        return Signal(
            symbol=symbol, type=SignalType.SELL, entry_price=close,
            stop_loss=close + p["stop"], take_profit=close - p["target"],
        )


class RangeScalpStrategy(BaseStrategy):
    """Opera dentro do range recente: compra perto do piso, vende no teto.

    Só atua quando o mercado está comprimido (range estreito comparado
    ao histórico) — em expansão de volatilidade, prover liquidez contra
    o movimento é como catar moedas na frente do trator.
    """

    DEFAULTS = {
        "window": 30,          # minutos que formam o range
        "max_range": 250.0,    # acima disso o mercado está expandindo: fora
        "edge_pct": 0.25,      # quão perto da borda para agir
        "target": 20.0,
        "stop": 40.0,
    }

    def __init__(self, params: dict | None = None):
        super().__init__({**self.DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        if len(candles) < p["window"] + 2:
            return hold

        window = candles.iloc[-p["window"] :]
        high, low = float(window["high"].max()), float(window["low"].min())
        span = high - low
        if span <= 0 or span > p["max_range"]:
            return hold

        close = float(candles["close"].iloc[-1])
        position = (close - low) / span

        if position <= p["edge_pct"]:
            return Signal(
                symbol=symbol, type=SignalType.BUY, entry_price=close,
                stop_loss=close - p["stop"], take_profit=close + p["target"],
            )
        if position >= 1 - p["edge_pct"]:
            return Signal(
                symbol=symbol, type=SignalType.SELL, entry_price=close,
                stop_loss=close + p["stop"], take_profit=close - p["target"],
            )
        return hold
