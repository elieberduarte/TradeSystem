"""Busca cega por padrões — com a régua do acaso ao lado.

Varre todas as variáveis de estado do mercado procurando faixas onde o
retorno futuro destoa, e compara cada achado com o que o puro ruído
produziria na mesma varredura.

Uso: python scripts/pattern_search.py [SIMBOLO] [timeframe] [horizonte]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.analysis.discovery import heatmap, permutation_baseline, scan_all
from src.bot.analysis.features import build, forward_return
from src.bot.data.history import HistoryStore


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "WIN$N"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "1d"
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    candles = HistoryStore().load(symbol, timeframe)
    if candles is None:
        raise SystemExit(f"Sem acervo {timeframe} para {symbol}")

    intraday = timeframe.endswith("m")
    features = build(candles, intraday=intraday)
    target = forward_return(candles, horizon)

    # Alinha e descarta o rabo sem futuro conhecido
    valid = target.notna()
    features, target = features[valid], target[valid]

    print(f"═══ Busca de padrões · {symbol} {timeframe} ═══")
    print(f"{len(features):,} observações · {len(features.columns)} variáveis de estado")
    print(f"Alvo: retorno dos próximos {horizon} candles, dividido pelo ATR\n")

    patterns = scan_all(features, target)
    if not patterns:
        raise SystemExit("Nenhuma faixa com amostra suficiente")

    print("── Os 12 padrões mais fortes encontrados ──")
    for pattern in patterns[:12]:
        print(f"  {pattern}")
    print(f"\nTotal de faixas testadas: {len(patterns)}")

    print("\n── A régua do acaso (teste de permutação) ──")
    print("Embaralhando o alvo e repetindo a MESMA busca 20 vezes...")
    baseline = permutation_baseline(features, target, rounds=20)
    threshold_95 = float(np.percentile(baseline, 95))
    print(f"  Melhor |t| no ruído: mediana {np.median(baseline):.2f} · "
          f"máximo {max(baseline):.2f} · percentil 95 = {threshold_95:.2f}")

    survivors = [p for p in patterns if abs(p.t_stat) > threshold_95]
    print(f"\n── Veredito ──")
    print(f"Padrões que superam a régua do acaso: {len(survivors)} de {len(patterns)}")
    if survivors:
        for pattern in survivors[:8]:
            print(f"  ✓ {pattern}")
    else:
        print("  Nenhum. Tudo que a varredura achou é compatível com ruído.")

    print("\n── Mapa de calor: retorno médio cruzando duas condições ──")
    for a, b in (("pos_20", "adx"), ("ibs", "tamanho")):
        if a in features and b in features:
            grid = heatmap(features[a], features[b], target)
            if not grid.empty:
                print(f"\n{a} (linhas) × {b} (colunas):")
                print(grid.round(3).to_string())


if __name__ == "__main__":
    main()
