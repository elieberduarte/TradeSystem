"""Estudo do horizonte de swing (candles diários).

A pergunta: o regime de mercado, que se mostrou imprevisível no
intradiário, é persistente na escala de dias? E os retornos diários
têm autocorrelação explorável?

Uso: python scripts/swing_study.py [SIMBOLO]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.bot.analysis.profile import swing_regime_persistence
from src.bot.data.history import HistoryStore


def autocorrelation(candles: pd.DataFrame, lags=(1, 2, 5, 10, 20)) -> pd.DataFrame:
    returns = candles["close"].pct_change().dropna()
    return pd.DataFrame(
        [
            {
                "lag_dias": lag,
                "autocorrelacao": round(float(returns.corr(returns.shift(-lag))), 4),
                "amostras": len(returns) - lag,
            }
            for lag in lags
        ]
    )


def momentum_study(candles: pd.DataFrame, lookbacks=(5, 10, 20, 60)) -> pd.DataFrame:
    """O retorno passado prevê o retorno futuro de mesmo prazo?

    Base do time series momentum, a anomalia mais robusta da literatura.
    Correlação positiva = momentum; negativa = reversão.
    """
    close = candles["close"]
    rows = []
    for lookback in lookbacks:
        past = close.pct_change(lookback)
        future = close.pct_change(lookback).shift(-lookback)
        pairs = pd.concat([past, future], axis=1).dropna()
        pairs.columns = ["passado", "futuro"]
        if len(pairs) < 30:
            continue
        # Acerto direcional: seguir o sinal do passado acerta a direção?
        hit = ((pairs["passado"] > 0) == (pairs["futuro"] > 0)).mean()
        rows.append(
            {
                "janela_dias": lookback,
                "correlacao": round(float(pairs["passado"].corr(pairs["futuro"])), 4),
                "acerto_direcional": round(float(hit), 3),
                "amostras": len(pairs),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    symbols = sys.argv[1:] or ["WIN$N", "WDO$N"]
    store = HistoryStore()

    for symbol in symbols:
        candles = store.load(symbol, "1d")
        if candles is None:
            print(f"{symbol}: sem acervo diário\n")
            continue
        print(f"═══ {symbol} diário ═══")
        print(f"{len(candles)} pregões ({candles.index.min().date()} → {candles.index.max().date()})\n")

        print("── Persistência do regime (a premissa do ciclo) ──")
        print(swing_regime_persistence(candles).to_string(index=False))
        print()

        print("── Autocorrelação dos retornos diários ──")
        print(autocorrelation(candles).to_string(index=False))
        print()

        print("── Momentum: retorno passado prevê o futuro? ──")
        print(momentum_study(candles).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
