"""Volatilidade intradiária dessazonalizada por hora.

O WIN anda o dobro às 10h do que às 15h. Um ATR bruto, portanto, não
mede "o mercado está agitado" — mede "que horas são". Usá-lo como
filtro de regime faz o bot simplesmente detectar a manhã.

A correção é dividir a volatilidade corrente pela volatilidade típica
daquele horário. O resultado é adimensional: 1,0 significa "normal
para este horário", 2,0 significa "o dobro do normal para este
horário". É esse número que diz se o mercado está grande ou pequeno.

A literatura brasileira mostra o tamanho do efeito: ao remover a
sazonalidade horária, a curtose do Ibovespa intradiário cai de 7,48
para 2,16 — ou seja, a maior parte da "cauda gorda" aparente é
horário do dia, não risco genuíno.
"""

import pandas as pd


def true_range(candles: pd.DataFrame) -> pd.Series:
    prev_close = candles["close"].shift(1)
    return pd.concat(
        [
            candles["high"] - candles["low"],
            (candles["high"] - prev_close).abs(),
            (candles["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR de Wilder, em pontos."""
    return true_range(candles).ewm(alpha=1 / period, adjust=False).mean()


def hourly_profile(candles: pd.DataFrame, min_samples: int = 30) -> pd.Series:
    """Volatilidade típica de cada hora do pregão, em pontos."""
    ranges = true_range(candles)
    frame = pd.DataFrame({"tr": ranges, "hour": candles.index.hour})
    profile = frame.groupby("hour")["tr"].median()
    counts = frame.groupby("hour")["tr"].size()
    return profile[counts >= min_samples]


def deseasonalized_atr(
    candles: pd.DataFrame, period: int = 14, profile: pd.Series | None = None
) -> pd.Series:
    """ATR dividido pela volatilidade típica do horário.

    Retorna a razão: 1,0 = normal para o horário; 2,0 = dobro do normal.
    Quando o horário não tem perfil conhecido, cai para a mediana geral.
    """
    values = atr(candles, period)
    reference = hourly_profile(candles) if profile is None else profile
    if reference.empty:
        return pd.Series(1.0, index=candles.index)
    fallback = float(reference.median())
    expected = pd.Series(candles.index.hour, index=candles.index).map(reference)
    expected = expected.fillna(fallback).replace(0, fallback)
    return values / expected


def market_size(
    candles: pd.DataFrame, period: int = 14, profile: pd.Series | None = None
) -> pd.Series:
    """Alias legível: 'tamanho do mercado' relativo ao normal do horário."""
    return deseasonalized_atr(candles, period, profile)
