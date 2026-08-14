"""Lead-lag entre contratos: um mercado antecipa o outro? (Eykyn E-21)

A tese do pit de Chicago era que o operador via o S&P grande mover e
corria para o mini. Em mercado eletrônico, qualquer defasagem
explorável em barras de 5-15 minutos deveria aparecer como correlação
entre o retorno defasado de A e o retorno seguinte de B.

Medição: correlação em lag 1 e taxa de acerto do sinal (o retorno de
A na barra anterior prevê o SINAL do retorno de B nesta barra?), com
p-valor binomial contra 50%. Sem estratégia, sem stop: se nem a
correlação bruta existir, não há o que operar.

Uso: python scripts/lead_lag.py
"""

import sys
from math import erf, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.data.history import HistoryStore


def sign_test(hits: int, n: int) -> float:
    """p-valor bicaudal (aproximação normal) para acertos contra 50%."""
    if n == 0:
        return 1.0
    z = (hits - n / 2) / (sqrt(n) / 2)
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def returns_of(candles: pd.DataFrame) -> pd.Series:
    return candles["close"].pct_change()


def evaluate(name: str, lead: pd.Series, lag: pd.Series) -> dict | None:
    """O retorno de `lead` na barra t-1 diz algo sobre `lag` na barra t?"""
    frame = pd.DataFrame({"lead": lead, "lag": lag}).dropna()
    if len(frame) < 500:
        return None
    x = frame["lead"].shift(1)
    y = frame["lag"]
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    pair = pair[(pair["x"] != 0) & (pair["y"] != 0)]

    corr = float(pair["x"].corr(pair["y"]))
    hits = int((np.sign(pair["x"]) == np.sign(pair["y"])).sum())
    n = len(pair)
    return {
        "par": name, "n": n, "corr_lag1": corr,
        "acerto_sinal": hits / n, "p": sign_test(hits, n),
    }


def main() -> None:
    store = HistoryStore()
    rows = []

    # Intraday: WIN × WDO nos dois sentidos, 5m e 15m
    for timeframe in ("5m", "15m"):
        win = store.load("WIN$N", timeframe)
        wdo = store.load("WDO$N", timeframe)
        if win is None or wdo is None:
            continue
        r_win, r_wdo = returns_of(win), returns_of(wdo)
        rows.append(evaluate(f"WDO→WIN {timeframe}", r_wdo, r_win))
        rows.append(evaluate(f"WIN→WDO {timeframe}", r_win, r_wdo))

    # Diário: os mercados que "deveriam" liderar o índice
    win_d = store.load("WIN$N", "1d")
    r_win_d = returns_of(win_d)
    for leader in ("DI1F27", "T10$N", "WSP$N", "WDO$N", "DOL$N"):
        candles = store.load(leader, "1d")
        if candles is None:
            continue
        rows.append(evaluate(f"{leader}→WIN 1d", returns_of(candles), r_win_d))

    print("═══ Lead-lag entre contratos · retorno defasado de A → retorno de B ═══\n")
    print(f"{'par':<16} {'barras':>8} {'corr lag-1':>11} {'acerto sinal':>13} {'p':>8}")
    print("-" * 62)
    for row in rows:
        if row is None:
            continue
        print(f"{row['par']:<16} {row['n']:>8,} {row['corr_lag1']:>11.4f} "
              f"{row['acerto_sinal']:>12.2%} {row['p']:>8.4f}")

    print("\nReferência: acerto de sinal 50% = moeda. Correlação precisa ser")
    print("grande o bastante para pagar spread + emolumentos em cada barra.")


if __name__ == "__main__":
    main()
