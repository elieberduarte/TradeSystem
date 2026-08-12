"""Onde fica a faixa útil de volatilidade?

Hipótese a testar: em mercado pequeno demais a fricção domina, e em
mercado grande demais as regras perdem precisão (choque de notícia,
stop furado, slippage real maior). Se for verdade, deve existir uma
faixa intermediária com resultado melhor que as pontas.

Este script não assume a faixa — mede. Divide o histórico em quintis
de "tamanho de mercado" (ATR dessazonalizado pela hora) e mede, em
cada um, o que uma entrada com alvo e stop proporcionais entrega.

Uso: python scripts/volatility_bands.py [SIMBOLO]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.analysis.volatility import atr, market_size
from src.bot.data.history import HistoryStore

FRICTION_POINTS = 12.5   # spread + emolumentos, entrada a mercado
MAX_BARS = 15
STEP = 5


def outcomes(candles: pd.DataFrame, target_atr: float = 1.0) -> pd.DataFrame:
    """Para cada ponto amostrado: tamanho do mercado e o que aconteceu.

    Alvo e stop simétricos em múltiplos do ATR — exatamente o desenho
    proposto: mercado grande, alvo grande; mercado pequeno, alvo pequeno.
    """
    volatility = atr(candles, 14).to_numpy()
    size = market_size(candles, 14).to_numpy()
    high = candles["high"].to_numpy()
    low = candles["low"].to_numpy()
    close = candles["close"].to_numpy()

    rows = []
    for i in range(30, len(close) - MAX_BARS - 1, STEP):
        vol = volatility[i]
        if not np.isfinite(vol) or vol <= 0 or not np.isfinite(size[i]):
            continue
        distance = target_atr * vol
        entry = close[i]
        up, down = entry + distance, entry - distance
        result = None
        for j in range(i + 1, i + 1 + MAX_BARS):
            if high[j] >= up:
                result = 1
                break
            if low[j] <= down:
                result = 0
                break
        if result is None:
            continue
        # Ganho e perda em pontos, já com fricção fixa
        gain = distance - FRICTION_POINTS
        loss = distance + FRICTION_POINTS
        rows.append(
            {
                "tamanho": size[i],
                "atr": vol,
                "acertou": result,
                "pnl_pts": gain if result else -loss,
            }
        )
    return pd.DataFrame(rows)


def report(frame: pd.DataFrame, column: str, label: str, bins: int = 5) -> pd.DataFrame:
    frame = frame.copy()
    frame["faixa"] = pd.qcut(frame[column], bins, duplicates="drop")
    grouped = frame.groupby("faixa", observed=True)
    out = pd.DataFrame(
        {
            f"{label}_min": grouped[column].min().round(2),
            f"{label}_max": grouped[column].max().round(2),
            "amostras": grouped.size(),
            "acerto": (grouped["acertou"].mean() * 100).round(1),
            "pnl_medio_pts": grouped["pnl_pts"].mean().round(2),
        }
    )
    out["pnl_medio_R$"] = (out["pnl_medio_pts"] * 0.20).round(2)
    return out.reset_index(drop=True)


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "WIN$N"
    candles = HistoryStore().load(symbol, "1m")
    if candles is None:
        raise SystemExit(f"Sem acervo de 1 minuto para {symbol}")

    print(f"═══ Faixa útil de volatilidade · {symbol} 1min ═══")
    print(f"{len(candles):,} candles · alvo e stop = 1×ATR · fricção fixa de {FRICTION_POINTS} pts")
    print("Entrada a mercado (a limitada foi descartada por seleção adversa)\n")

    frame = outcomes(candles)
    if frame.empty:
        raise SystemExit("Sem amostras suficientes")

    print("── Por TAMANHO DE MERCADO (ATR dividido pelo normal do horário) ──")
    print("1,0 = típico para o horário · acima = mercado grande · abaixo = pequeno")
    print(report(frame, "tamanho", "tamanho").to_string(index=False))
    print()

    print("── Por ATR ABSOLUTO (pontos) ──")
    print(report(frame, "atr", "atr").to_string(index=False))
    print()

    positive = frame[frame["pnl_pts"] > 0]
    print(f"Resultado geral: acerto {frame['acertou'].mean():.1%} · "
          f"expectativa {frame['pnl_pts'].mean():.2f} pts "
          f"(R$ {frame['pnl_pts'].mean() * 0.20:.2f}) em {len(frame):,} amostras")
    print(f"Amostras com resultado positivo: {len(positive):,} ({len(positive) / len(frame):.1%})")


if __name__ == "__main__":
    main()
