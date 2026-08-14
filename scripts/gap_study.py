"""O gap fecha ou continua? (Bill Eykyn, cap. de gaps)

Eykyn afirma três faixas, em ticks de T-Bond de 2003:
    |gap| ≤ 8      → "Gap to Close": espera-se o fechamento do gap
    8 < |gap| < 10 → indefinido
    |gap| ≥ 10–12  → "Gap to Follow": espera-se continuação

Isto não é um setup, é uma AFIRMAÇÃO FALSIFICÁVEL — e a mais barata de
testar de todo o material dos livros, porque não tem grau de liberdade
de saída: a variável dependente é binária (o gap fechou no mesmo dia?).

Normalizamos o gap pelo ATR, já que 8 ticks de T-Bond não significam
nada no WIN. Se a probabilidade de fechamento cair de forma monotônica
com o tamanho relativo do gap, o autor tem razão; se a curva for
plana, cai toda a família "gap to follow" da literatura.

Uso: python scripts/gap_study.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.data.history import HistoryStore
from src.bot.strategies.swing_reversion import atr
from src.bot.universe import EXPANDED, FUTUROS

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = sorted(set(EXPANDED) | set(FUTUROS))


def gaps_of(candles: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Cada abertura com gap e o que aconteceu no dia."""
    volatility = atr(candles, 14)
    prev_close = candles["close"].shift(1)
    gap = candles["open"] - prev_close

    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "gap": gap,
            "gap_atr": gap / volatility,
            "open": candles["open"],
            "high": candles["high"],
            "low": candles["low"],
            "close": candles["close"],
            "prev_close": prev_close,
        }
    ).dropna()
    frame = frame[frame["gap"] != 0]
    if frame.empty:
        return frame

    up = frame["gap"] > 0
    # Fechou o gap = o preço voltou ao fechamento anterior no mesmo dia
    frame["fechou"] = np.where(up, frame["low"] <= frame["prev_close"],
                               frame["high"] >= frame["prev_close"])
    # Continuou = o dia terminou além da abertura, na direção do gap
    frame["continuou"] = np.where(up, frame["close"] > frame["open"],
                                  frame["close"] < frame["open"])
    # Excursão máxima na direção do gap, em ATR
    frame["extensao_atr"] = np.where(
        up, (frame["high"] - frame["open"]), (frame["open"] - frame["low"])
    ) / (frame["gap"] / frame["gap_atr"]).abs()
    return frame


def main() -> None:
    store = HistoryStore()
    rows = []
    for symbol in UNIVERSE:
        candles = store.load(symbol, "1d")
        if candles is None or len(candles) < 300:
            continue
        frame = gaps_of(candles, symbol)
        if not frame.empty:
            rows.append(frame)

    if not rows:
        raise SystemExit("Sem dados")
    data = pd.concat(rows, ignore_index=True)
    data["gap_abs"] = data["gap_atr"].abs()
    data = data[np.isfinite(data["gap_abs"])]

    print("═══ O gap fecha ou continua? ═══")
    print(f"{len(data):,} aberturas com gap · {data['symbol'].nunique()} instrumentos\n")

    print("── Por tamanho do gap (em múltiplos do ATR) ──")
    data["faixa"] = pd.cut(
        data["gap_abs"],
        bins=[0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, np.inf],
        labels=["<0,10", "0,10-0,25", "0,25-0,50", "0,50-0,75",
                "0,75-1,00", "1,00-1,50", ">1,50"],
    )
    grouped = data.groupby("faixa", observed=True)
    table = pd.DataFrame(
        {
            "amostras": grouped.size(),
            "fechou_%": (grouped["fechou"].mean() * 100).round(1),
            "continuou_%": (grouped["continuou"].mean() * 100).round(1),
        }
    )
    print(table.to_string())

    print("\n── A afirmação de Eykyn ──")
    print("Ele diz: gap PEQUENO fecha, gap GRANDE continua.")
    small = data[data["gap_abs"] <= 0.25]
    large = data[data["gap_abs"] >= 1.0]
    if len(small) > 30 and len(large) > 30:
        print(f"  gaps pequenos (≤0,25 ATR): fecham em {small['fechou'].mean():.1%} "
              f"({len(small):,} casos)")
        print(f"  gaps grandes (≥1,00 ATR):  fecham em {large['fechou'].mean():.1%} "
              f"({len(large):,} casos)")
        delta = small["fechou"].mean() - large["fechou"].mean()
        print(f"  diferença: {delta:+.1%}")
        if delta > 0.10:
            print("  → CONSISTENTE com a afirmação: gap pequeno fecha mais")
        elif delta < -0.10:
            print("  → CONTRADIZ a afirmação")
        else:
            print("  → curva praticamente plana: a afirmação NÃO se sustenta")

    # Monotonicidade: a probabilidade de fechar cai conforme o gap cresce?
    correlation = float(data["gap_abs"].corr(data["fechou"].astype(float)))
    print(f"\nCorrelação entre tamanho do gap e fechar no dia: {correlation:+.3f}")

    print("\n── Direção: o gap prevê o resto do dia? ──")
    print("(continuou = o dia fechou além da abertura, na direção do gap)")
    for label, subset in (("gap de alta", data[data["gap"] > 0]),
                          ("gap de baixa", data[data["gap"] < 0])):
        print(f"  {label:<14} continuação em {subset['continuou'].mean():.1%} "
              f"({len(subset):,} casos)")
    print("  50% = moeda. Diferença relevante exigiria algo acima de ~53%.")

    print("\n── Por classe de ativo ──")
    def classify(symbol: str) -> str:
        if symbol.startswith("DI1"):
            return "juros"
        if symbol in ("WIN$N", "IND$N", "WDO$N", "DOL$N", "WSP$N", "T10$N"):
            return "futuro índice/câmbio"
        if symbol.endswith("$N"):
            return "commodity"
        if symbol.endswith("11"):
            return "ETF"
        return "ação"

    data["classe"] = data["symbol"].map(classify)
    by_class = data.groupby("classe").agg(
        amostras=("fechou", "size"),
        fechou=("fechou", "mean"),
        continuou=("continuou", "mean"),
    )
    by_class["fechou"] = (by_class["fechou"] * 100).round(1)
    by_class["continuou"] = (by_class["continuou"] * 100).round(1)
    print(by_class.to_string())

    out = ROOT / "web" / "gap_study.json"
    out.write_text(
        json.dumps(
            {
                "amostras": int(len(data)),
                "instrumentos": int(data["symbol"].nunique()),
                "correlacao_tamanho_fechamento": round(correlation, 3),
                "por_faixa": table.reset_index().astype(str).to_dict("records"),
            },
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
