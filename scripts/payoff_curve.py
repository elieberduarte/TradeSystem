"""Acerto e payoff são independentes?

Taxa de acerto isolada não determina desempenho: 30% de acerto com
payoff 3:1 é lucrativo. A pergunta que importa é outra — ao afastar o
alvo para melhorar o payoff, quanto o acerto cai em troca?

Num passeio aleatório sem deriva, a probabilidade de tocar o alvo
antes do stop é L/(L+G): exatamente o que zera a expectativa. Se o
mercado se comportar assim, mudar o payoff não cria vantagem, apenas
troca acerto por tamanho de ganho — e a fricção continua sendo o
único saldo.

Este script mede o acerto REAL para vários payoffs e compara com a
probabilidade teórica do passeio aleatório. A diferença entre as duas
colunas é o edge que existe (ou não) no ativo.

Uso: python scripts/payoff_curve.py [SIMBOLO] [timeframe]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.analysis.volatility import atr
from src.bot.data.history import HistoryStore

POINT_VALUE = {"WIN$N": 0.20, "WDO$N": 10.00}


def measure(candles: pd.DataFrame, ratios, stop_atr: float, max_bars: int, step: int,
            friction: float, point_value: float) -> pd.DataFrame:
    volatility = atr(candles, 14).to_numpy()
    high = candles["high"].to_numpy()
    low = candles["low"].to_numpy()
    close = candles["close"].to_numpy()

    rows = []
    for ratio in ratios:
        wins = losses = unresolved = 0
        for i in range(30, len(close) - max_bars - 1, step):
            vol = volatility[i]
            if not np.isfinite(vol) or vol <= 0:
                continue
            stop_distance = stop_atr * vol
            target_distance = ratio * stop_distance
            entry = close[i]
            up, down = entry + target_distance, entry - stop_distance
            resolved = False
            for j in range(i + 1, i + 1 + max_bars):
                # Stop primeiro: pior caso quando os dois cabem no candle
                if low[j] <= down:
                    losses += 1
                    resolved = True
                    break
                if high[j] >= up:
                    wins += 1
                    resolved = True
                    break
            if not resolved:
                unresolved += 1

        total = wins + losses
        if total == 0:
            continue
        actual = wins / total
        # Passeio aleatório: P(alvo antes do stop) = stop / (stop + alvo)
        theoretical = 1 / (1 + ratio)
        gain = ratio - friction / (stop_atr * float(np.nanmedian(volatility)))
        # Expectativa em múltiplos do stop, já com fricção
        median_stop = stop_atr * float(np.nanmedian(volatility))
        gain_pts = ratio * median_stop - friction
        loss_pts = median_stop + friction
        expectancy = actual * gain_pts - (1 - actual) * loss_pts
        breakeven = loss_pts / (gain_pts + loss_pts) if gain_pts + loss_pts else 0
        rows.append(
            {
                "payoff": f"{ratio:.1f}:1",
                "acerto_real": round(actual * 100, 1),
                "acerto_aleatorio": round(theoretical * 100, 1),
                "edge_pp": round((actual - theoretical) * 100, 1),
                "acerto_p_empatar": round(breakeven * 100, 1),
                "expectativa_pts": round(expectancy, 1),
                "expectativa_R$": round(expectancy * point_value, 2),
                "resolvidos": total,
                "sem_resolucao": unresolved,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "WIN$N"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "1m"
    candles = HistoryStore().load(symbol, timeframe)
    if candles is None:
        raise SystemExit(f"Sem acervo {timeframe} para {symbol}")

    point_value = POINT_VALUE.get(symbol, 1.0)
    intraday = timeframe.endswith("m")
    friction = 12.5 if intraday else 12.5
    max_bars = 60 if intraday else 40
    step = 10 if intraday else 2

    print(f"═══ Acerto × payoff · {symbol} {timeframe} ═══")
    print(f"{len(candles):,} candles · stop fixo em 1×ATR · alvo variável")
    print(f"Fricção {friction} pts · até {max_bars} barras para resolver\n")

    frame = measure(
        candles, ratios=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
        stop_atr=1.0, max_bars=max_bars, step=step,
        friction=friction, point_value=point_value,
    )
    print(frame.to_string(index=False))
    print()
    print("acerto_aleatorio = probabilidade de um passeio aleatório sem deriva")
    print("edge_pp = quanto o ativo entrega ACIMA do acaso, em pontos percentuais")
    print("Se edge_pp ≈ 0 em todos os payoffs, mudar o alvo não cria vantagem —")
    print("só troca acerto por tamanho, e a fricção continua sendo o saldo.")


if __name__ == "__main__":
    main()
