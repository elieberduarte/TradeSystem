"""O "tipo de dia" dá para ler no meio do pregão? Medição honesta.

Para cada sessão do WIN 5m (3,5 anos), o relógio para às 10h, 10h30,
11h, 12h, 13h e 14h. Com o que existia ATÉ ALI calculam-se as medidas
de tendenciosidade (eficiência de Kaufman, microcanal, lado da EMA,
lado do VWAP, retração máxima, ADX, expansão de range). Depois
pergunta-se ao resto do dia:

  1. a direção da sessão continuou até o fechamento?
  2. quantos pontos rendeu seguir a favor (entrada no relógio,
     saída no fechamento)?

Cada medida é avaliada pelo contraste entre o quintil mais
"tendencioso" e o menos: se a leitura visual do trader tem conteúdo,
o quintil alto precisa continuar mais e render mais — e o rendimento
precisa superar a fricção (12,5 pts na melhor hipótese; ~30 pts a
mercado nas duas pontas).

Referências já medidas: persistência manhã→tarde incondicional de
55,1% (estudo do payroll) e matriz de transição de regime com linhas
idênticas (classificação em 3 estados). Este estudo é a versão
contínua e multi-medida das duas.

Uso: python scripts/regime_nowcast.py
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from math import erf, sqrt

import numpy as np
import pandas as pd


def spearmanr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Correlação de Spearman com p-valor pela aproximação normal."""
    rho = float(x.rank().corr(y.rank()))
    n = len(x)
    if n < 10 or np.isnan(rho):
        return rho, 1.0
    z = rho * sqrt(n - 1)
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return rho, p_value

from src.bot.analysis.nowcast import (
    kaufman_efficiency,
    max_retrace_fraction,
    microchannel,
    session_vwap,
    side_consistency,
)
from src.bot.analysis.regime import adx
from src.bot.data.history import HistoryStore

ROOT = Path(__file__).resolve().parents[1]
CLOCKS = [time(10, 0), time(10, 30), time(11, 0), time(12, 0), time(13, 0), time(14, 0)]
FRICTION_PTS = 12.5

# Medidas em que MAIOR = mais tendencioso; a retração é invertida
FEATURES = ["eficiencia", "microcanal", "lado_ema", "lado_vwap",
            "retracao_inv", "adx", "expansao"]


def collect(candles: pd.DataFrame) -> pd.DataFrame:
    ema9 = candles["close"].ewm(span=9, adjust=False).mean()
    adx14 = adx(candles)["adx"]
    rows = []

    for day, bars in candles.groupby(candles.index.normalize()):
        if len(bars) < 60:
            continue
        for clock in CLOCKS:
            so_far = bars[bars.index.time <= clock]
            rest = bars[bars.index.time > clock]
            if len(so_far) < 8 or len(rest) < 12:
                continue
            open_session = float(so_far["open"].iloc[0])
            close_now = float(so_far["close"].iloc[-1])
            close_end = float(rest["close"].iloc[-1])
            direction = np.sign(close_now - open_session)
            if direction == 0:
                continue

            closes = so_far["close"]
            ema_slice = ema9.loc[so_far.index]
            vwap_slice = session_vwap(so_far)
            move = (close_end - close_now) * direction

            rows.append({
                "day": day, "clock": clock.strftime("%H:%M"),
                "eficiencia": kaufman_efficiency(closes),
                "microcanal": microchannel(closes, ema_slice),
                "lado_ema": side_consistency(closes, ema_slice),
                "lado_vwap": side_consistency(closes, vwap_slice),
                "retracao_inv": -max_retrace_fraction(closes, direction),
                "adx": float(adx14.loc[so_far.index[-1]]),
                "range_pts": float(so_far["high"].max() - so_far["low"].min()),
                "continuou": np.sign(close_end - close_now) == direction,
                "move": move,
            })

    frame = pd.DataFrame(rows)
    # Expansão de range: o range até T dividido pela mediana do range
    # até o MESMO T (compara o dia com dias na mesma altura do pregão)
    frame["expansao"] = frame["range_pts"] / frame.groupby("clock")["range_pts"].transform("median")
    return frame


def main() -> None:
    store = HistoryStore()
    candles = store.load("WIN$N", "5m")
    frame = collect(candles)

    print("═══ Nowcast de regime · WIN 5m · a direção da sessão continua? ═══")
    print(f"Fricção de referência: {FRICTION_PTS} pts (limitada) / ~30 pts (a mercado)\n")

    report = {}
    for clock, group in frame.groupby("clock"):
        base_cont = group["continuou"].mean()
        base_move = group["move"].mean()
        print(f"── {clock} · {len(group)} dias · continuação incondicional "
              f"{base_cont:.1%} · rende {base_move:+.0f} pts ──")
        print(f"{'medida':<14} {'Q5 cont.':>9} {'Q5 pts':>8} {'Q1 cont.':>9} "
              f"{'Q1 pts':>8} {'spearman':>9} {'p':>8}")

        clock_report = {"dias": len(group), "base_cont": round(float(base_cont), 3),
                        "base_move": round(float(base_move), 1), "medidas": {}}
        for feature in FEATURES:
            values = group[feature]
            valid = group[values.notna()]
            if len(valid) < 100:
                continue
            q_top = valid[valid[feature] >= valid[feature].quantile(0.8)]
            q_bottom = valid[valid[feature] <= valid[feature].quantile(0.2)]
            rho, p_value = spearmanr(valid[feature], valid["move"])
            print(f"{feature:<14} {q_top['continuou'].mean():>8.1%} "
                  f"{q_top['move'].mean():>+8.0f} {q_bottom['continuou'].mean():>8.1%} "
                  f"{q_bottom['move'].mean():>+8.0f} {rho:>9.3f} {p_value:>8.4f}")
            clock_report["medidas"][feature] = {
                "q5_cont": round(float(q_top["continuou"].mean()), 3),
                "q5_move": round(float(q_top["move"].mean()), 1),
                "q1_cont": round(float(q_bottom["continuou"].mean()), 3),
                "q1_move": round(float(q_bottom["move"].mean()), 1),
                "spearman": round(float(rho), 3),
                "p": round(float(p_value), 4),
            }
        report[clock] = clock_report
        print()

    out = ROOT / "web" / "nowcast.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Exportado para {out}")


if __name__ == "__main__":
    main()
