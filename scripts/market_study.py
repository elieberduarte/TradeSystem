"""Estudo do comportamento do WIN — onde existe assimetria explorável.

Uso: python scripts/market_study.py [SIMBOLO] [timeframe]

Imprime o relatório e exporta web/study.json para o painel.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.analysis.profile import (
    by_hour,
    daily_regimes,
    gap_study,
    opening_range_study,
    summarize,
)
from src.bot.data.history import HistoryStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = sys.argv[1:]
    symbol = args[0] if args else "WIN$N"
    timeframe = args[1] if len(args) > 1 else "5m"

    candles = HistoryStore().load(symbol, timeframe)
    if candles is None:
        raise SystemExit(f"Sem acervo para {symbol} {timeframe}")
    print(f"═══ Estudo de mercado · {symbol} {timeframe} ═══")
    print(f"{len(candles)} candles ({candles.index.min()} → {candles.index.max()})\n")

    hours = by_hour(candles)
    print("── Por hora do pregão (pontos) ──")
    print(hours.to_string())
    print()

    regimes = daily_regimes(candles)
    counts = regimes["regime"].value_counts()
    print("── Regime dos pregões ──")
    for regime, n in counts.items():
        print(f"  {regime}: {n} dias ({n / len(regimes):.1%})")
    print(f"  amplitude média do dia: {regimes['amplitude'].mean():.0f} pts")
    print(f"  em dias de tendência: {regimes[regimes.regime != 'lateral']['amplitude'].mean():.0f} pts")
    print(f"  em dias laterais: {regimes[regimes.regime == 'lateral']['amplitude'].mean():.0f} pts")
    print()

    breakouts = opening_range_study(candles)
    if not breakouts.empty:
        print("── Rompimento da abertura: continuação ou reversão? ──")
        print(f"  {len(breakouts)} rompimentos (1º de cada dia)")
        print(f"  máximo a favor (MFE) médio: {breakouts['mfe'].mean():.0f} pts")
        print(f"  máximo contra (MAE) médio:  {breakouts['mae'].mean():.0f} pts")
        print(f"  razão MFE/MAE: {breakouts['mfe'].mean() / breakouts['mae'].mean():.2f}")
        print("  por hora do rompimento:")
        per_hour = breakouts.groupby("hora").agg(
            n=("mfe", "size"), mfe=("mfe", "mean"), mae=("mae", "mean")
        )
        per_hour["razao"] = (per_hour["mfe"] / per_hour["mae"]).round(2)
        print(per_hour[per_hour["n"] >= 20].round(0).to_string())
        print()

    gaps = gap_study(candles)
    big = gaps[gaps["gap"].abs() > gaps["gap"].abs().quantile(0.75)]
    fechou = ((big["gap"] > 0) & (big["dia"] < 0)) | ((big["gap"] < 0) & (big["dia"] > 0))
    print("── Gap de abertura ──")
    print(f"  gap médio (abs): {gaps['gap'].abs().mean():.0f} pts")
    print(f"  em gaps grandes (top 25%): dia reverteu o gap em {fechou.mean():.1%} das vezes")
    print()

    summary = summarize(candles)
    print("── Resumo ──")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "periodo": [str(candles.index.min()), str(candles.index.max())],
        "summary": summary,
        "by_hour": [
            {"hora": int(h), **{k: float(v) for k, v in row.items()}}
            for h, row in hours.iterrows()
        ],
        "regimes": {k: int(v) for k, v in counts.items()},
    }
    out = ROOT / "web" / "study.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
