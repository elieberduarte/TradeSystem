"""Classificação do regime de mercado.

"Qual é o mercado de hoje?" — tendência ou lateralização — é a pergunta
que decide qual setup pode operar. Aqui ela é respondida com o ADX de
Wilder: ADX alto = mercado direcional (a direção vem do +DI/-DI);
ADX baixo = mercado lateral.
"""

from enum import Enum

import pandas as pd


class Regime(Enum):
    TREND_UP = "tendencia_alta"
    TREND_DOWN = "tendencia_baixa"
    RANGE = "lateral"


def adx(candles: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX, +DI e -DI de Wilder. Retorna DataFrame com as três colunas."""
    high, low, close = candles["high"], candles["low"], candles["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Suavização de Wilder = média móvel exponencial com alpha 1/period
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr

    denominator = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / denominator.where(denominator != 0)
    adx_series = dx.ewm(alpha=alpha, adjust=False).mean()

    return pd.DataFrame({"adx": adx_series, "plus_di": plus_di, "minus_di": minus_di})


def classify(
    candles: pd.DataFrame, period: int = 14, trend_threshold: float = 25.0
) -> Regime:
    """Classifica o regime vigente no último candle."""
    if len(candles) < period * 3:
        return Regime.RANGE
    indicators = adx(candles, period).iloc[-1]
    if pd.isna(indicators["adx"]) or indicators["adx"] < trend_threshold:
        return Regime.RANGE
    if indicators["plus_di"] >= indicators["minus_di"]:
        return Regime.TREND_UP
    return Regime.TREND_DOWN
