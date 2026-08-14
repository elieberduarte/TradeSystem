"""Nowcast de regime: quantificar o "tipo de dia" enquanto ele acontece.

O trader olha o gráfico às 10h30 e "sente" que o dia é de tendência:
andou muito sem devolver, colou na média, retrações rasas. Cada uma
dessas impressões visuais tem uma tradução numérica calculável em
tempo real, sem olhar o futuro:

  eficiência      razão de Kaufman: |deslocamento| / caminho percorrido.
                  1,0 = linha reta; perto de 0 = serrote.
  microcanal      maior sequência de fechamentos do mesmo lado da EMA.
  lado_ema        fração das barras fechando do lado dominante da EMA.
  lado_vwap       idem, contra o VWAP da sessão.
  retracao        maior devolução contra a direção da sessão, como
                  fração do avanço (tendência limpa = retração rasa).

A pergunta que o estudo responde: essas medidas, lidas no MEIO do
pregão, preveem que a direção continua até o fechamento? E por
quantos pontos — acima ou abaixo da fricção?
"""

import numpy as np
import pandas as pd


def kaufman_efficiency(closes: pd.Series) -> float:
    """|deslocamento líquido| dividido pela soma dos passos absolutos."""
    if len(closes) < 3:
        return float("nan")
    path = float(closes.diff().abs().sum())
    if path <= 0:
        return 0.0
    return float(abs(closes.iloc[-1] - closes.iloc[0]) / path)


def microchannel(closes: pd.Series, reference: pd.Series) -> int:
    """Maior sequência de fechamentos consecutivos do MESMO lado da média."""
    sides = np.sign(closes.to_numpy(dtype=float) - reference.to_numpy(dtype=float))
    best = run = 0
    previous = 0.0
    for side in sides:
        if side != 0 and side == previous:
            run += 1
        else:
            run = 1 if side != 0 else 0
        previous = side
        best = max(best, run)
    return best


def side_consistency(closes: pd.Series, reference: pd.Series) -> float:
    """Fração das barras do lado dominante da referência (0,5 = equilíbrio)."""
    if len(closes) == 0:
        return float("nan")
    above = float((closes.to_numpy() > reference.to_numpy()).mean())
    return max(above, 1.0 - above)


def session_vwap(bars: pd.DataFrame) -> pd.Series:
    """VWAP acumulado da sessão (preço típico ponderado por volume)."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    volume = bars["volume"].replace(0, np.nan).ffill().fillna(1.0)
    return (typical * volume).cumsum() / volume.cumsum()


def max_retrace_fraction(closes: pd.Series, direction: float) -> float:
    """Maior retração contra a direção da sessão ÷ avanço total.

    Perto de 0 = tendência que não devolveu nada; acima de 1 = o
    "avanço" já foi inteiramente devolvido em algum momento.
    """
    if len(closes) < 3 or direction == 0:
        return float("nan")
    signed = closes * direction        # vira sempre "alta"
    advance = float(signed.max() - signed.iloc[0])
    if advance <= 0:
        return float("nan")
    retrace = float((signed.cummax() - signed).max())
    return retrace / advance
