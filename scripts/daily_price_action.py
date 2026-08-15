"""Price action no DIÁRIO: o que cada tipo de dia diz sobre o seguinte?

As perguntas do usuário, com a magnitude que a média esconde. Tudo
normalizado pelo ATR(14) do dia anterior (r = variação do fechamento
em ATRs; d = deslocamento abertura→fechamento em ATRs):

  1. Depois de uma QUEDA, o "dia de alta" seguinte recupera TUDO ou
     só fecha um tico acima? (P(sobe) esconde a fração recuperada)
  2. Duas altas seguidas → e o terceiro dia?
  3. Dia de DESLOCAMENTO DE VERDADE (|abre→fecha| >= 1,5 ATR) → o
     que o dia seguinte faz?

Condições pré-declaradas (e só estas — sem pescaria):
  quedas/altas fortes (|r| >= 1 ATR) e fracas; sequências de 2 dias
  (qualquer e fortes >= 0,5 cada); deslocamento comprador/vendedor.

Régua de relevância: no diário do WIN a fricção (~30 pts) vale
~0,01 ATR — efeito médio acima disso é operável em tese.

Uso: python scripts/daily_price_action.py
"""

import json
import sys
from math import erf, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.bot.data.history import HistoryStore
from src.bot.strategies.swing_reversion import atr

ROOT = Path(__file__).resolve().parents[1]


def binomial_p(hits: int, n: int, base: float) -> float:
    if n == 0:
        return 1.0
    z = (hits - n * base) / sqrt(n * base * (1 - base))
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def study(symbol: str, store: HistoryStore) -> list[dict]:
    daily = store.load(symbol, "1d")
    volatility = atr(daily, 14).shift(1)          # ATR até ontem (sem o próprio dia)
    close, open_ = daily["close"], daily["open"]

    r = (close - close.shift(1)) / volatility      # variação diária em ATRs
    d = (close - open_) / volatility               # deslocamento do dia em ATRs

    frame = pd.DataFrame({
        "r": r, "d": d,
        "r_next": r.shift(-1),
        "close": close, "close_prev": close.shift(1),
        "close_next": close.shift(-1),
    }).dropna()

    base_up = float((frame["r_next"] > 0).mean())

    conditions = [
        ("incondicional", pd.Series(True, index=frame.index)),
        ("queda forte (r ≤ −1)", frame["r"] <= -1),
        ("queda fraca (−1 < r < 0)", (frame["r"] > -1) & (frame["r"] < 0)),
        ("alta fraca (0 < r < 1)", (frame["r"] > 0) & (frame["r"] < 1)),
        ("alta forte (r ≥ 1)", frame["r"] >= 1),
        ("2 altas seguidas", (frame["r"] > 0) & (frame["r"].shift(1) > 0)),
        ("2 altas FORTES (≥0,5 cada)", (frame["r"] >= 0.5) & (frame["r"].shift(1) >= 0.5)),
        ("2 quedas seguidas", (frame["r"] < 0) & (frame["r"].shift(1) < 0)),
        ("2 quedas FORTES (≤−0,5 cada)", (frame["r"] <= -0.5) & (frame["r"].shift(1) <= -0.5)),
        ("deslocamento comprador (d ≥ 1,5)", frame["d"] >= 1.5),
        ("deslocamento vendedor (d ≤ −1,5)", frame["d"] <= -1.5),
    ]

    rows = []
    print(f"\n═══ {symbol} · diário · o dia seguinte, condicionado ═══")
    print(f"{'condição':<32} {'n':>5} {'P(sobe)':>8} {'p':>7} {'média (ATR)':>12} "
          f"{'mediana':>8} {'recupera tudo':>13}")
    print("-" * 92)
    for label, mask in conditions:
        part = frame[mask.fillna(False)]
        n = len(part)
        if n < 30:
            continue
        ups = int((part["r_next"] > 0).sum())
        p_up = ups / n
        p_value = binomial_p(ups, n, base_up) if label != "incondicional" else 1.0
        mean_next = float(part["r_next"].mean())
        median_next = float(part["r_next"].median())

        # "Recuperou tudo": só faz sentido após queda — o fechamento de
        # amanhã volta ACIMA do fechamento de anteontem?
        recovered = None
        if "queda" in label or "vendedor" in label:
            recovered = float((part["close_next"] > part["close_prev"]).mean())

        rows.append({"condicao": label, "n": n, "p_sobe": round(p_up, 3),
                     "p_valor": round(p_value, 4), "media_atr": round(mean_next, 3),
                     "recupera": round(recovered, 3) if recovered is not None else None})
        rec_txt = f"{recovered:.1%}" if recovered is not None else "—"
        flag = " ◀" if p_value < 0.05 and label != "incondicional" else ""
        print(f"{label:<32} {n:>5} {p_up:>7.1%} {p_value:>7.3f} {mean_next:>+12.3f} "
              f"{median_next:>+8.3f} {rec_txt:>13}{flag}")
    return rows


def main() -> None:
    store = HistoryStore()
    payload = {}
    for symbol in ("WIN$N", "WDO$N"):
        payload[symbol] = study(symbol, store)

    out = ROOT / "web" / "daily_price_action.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nLeitura: média em ATRs do dia seguinte · fricção diária do WIN ≈ 0,01 ATR")
    print(f"'Recupera tudo' = fechamento de amanhã acima do fechamento de ANTEONTEM")
    print(f"◀ = P(sobe) difere da incondicional com p < 0,05 (antes de correção)")
    print(f"Exportado para {out}")


if __name__ == "__main__":
    main()
