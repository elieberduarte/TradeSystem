"""Descrição do estado do mercado em variáveis mensuráveis.

Para procurar padrões sem partir de um modelo pronto, primeiro é
preciso descrever cada instante do mercado por números. Cada coluna
aqui é uma pergunta simples sobre "como está o mercado agora" —
nenhuma delas pressupõe uma estratégia.

O alvo (`forward_return`) é o que aconteceu depois. Cruzar os dois é
o que permite perguntar aos dados, em vez de perguntar a um livro.
"""

import numpy as np
import pandas as pd

from src.bot.analysis.regime import adx
from src.bot.analysis.volatility import atr, market_size


def build(candles: pd.DataFrame, intraday: bool = True) -> pd.DataFrame:
    """Constrói a tabela de estado do mercado, candle a candle."""
    close = candles["close"]
    high, low = candles["high"], candles["low"]
    volatility = atr(candles, 14)

    frame = pd.DataFrame(index=candles.index)

    # ── Onde o candle fechou dentro do próprio range ──
    span = (high - low).replace(0, np.nan)
    frame["ibs"] = ((close - low) / span).clip(0, 1)

    # ── Movimento recente, normalizado pela volatilidade ──
    for periods in (1, 3, 5, 10, 20):
        frame[f"ret_{periods}"] = (close - close.shift(periods)) / volatility

    # ── Volatilidade: absoluta e relativa ao normal do horário ──
    frame["atr"] = volatility
    frame["tamanho"] = market_size(candles, 14) if intraday else volatility / volatility.rolling(60).median()

    # ── Posição no range recente ──
    for window in (20, 60):
        top = high.rolling(window).max()
        bottom = low.rolling(window).min()
        width = (top - bottom).replace(0, np.nan)
        frame[f"pos_{window}"] = ((close - bottom) / width).clip(0, 1)

    # ── Força direcional ──
    indicators = adx(candles, 14)
    frame["adx"] = indicators["adx"]
    frame["di_diff"] = indicators["plus_di"] - indicators["minus_di"]

    # ── Volume relativo ──
    if "volume" in candles:
        media = candles["volume"].rolling(60).median().replace(0, np.nan)
        frame["volume_rel"] = candles["volume"] / media

    # ── Contexto de calendário ──
    frame["hora"] = candles.index.hour
    frame["dia_semana"] = candles.index.dayofweek

    # ── Sequência: quantos candles seguidos na mesma direção ──
    direction = np.sign(close.diff())
    streak = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
    frame["sequencia"] = streak * direction

    return frame


def forward_return(candles: pd.DataFrame, horizon: int, normalize: bool = True) -> pd.Series:
    """Retorno futuro em `horizon` candles, normalizado pela volatilidade.

    Normalizar é essencial: sem isso, o mapa encontraria apenas
    "períodos voláteis rendem mais em pontos", que é trivial e não
    operável.
    """
    future = candles["close"].shift(-horizon) - candles["close"]
    if not normalize:
        return future
    return future / atr(candles, 14)
