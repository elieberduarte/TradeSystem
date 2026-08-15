"""A mesma bateria H1-H10, agora nas 13 AÇÕES — o comportamento muda?

Prior declarado antes de rodar: ações têm a reversão de curto prazo
mais documentada de todas as classes (fluxo de varejo, papel a papel
menos arbitrado) e deriva positiva estrutural — as células de
reversão (H1, H2 inferior, H8) devem sair MAIS FORTES que nos
futuros; as de continuação, iguais ou piores.

Duas réguas: estatística agrupada (com o aviso de que eventos do
mesmo dia em papéis correlacionados não são independentes — o
Ibovespa move todos juntos) e a régua da casa: em QUANTAS das 13
ações a média fica do lado previsto (replicação).

Uso: python scripts/pa_battery_stocks.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.pa_battery import ORDER, cell_stats, collect_cells
from src.bot.data.history import HistoryStore
from src.bot.universe import ACOES_BR

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    store = HistoryStore()
    pooled: dict[str, list[float]] = {}
    per_stock_mean: dict[str, list[float]] = {}

    used = 0
    for symbol in ACOES_BR:
        df = store.load(symbol, "1d")
        if df is None or len(df) < 700:
            continue
        used += 1
        for name, values in collect_cells(df).items():
            clean = [v for v in values if not np.isnan(v)]
            pooled.setdefault(name, []).extend(clean)
            if len(clean) >= 15:
                per_stock_mean.setdefault(name, []).append(float(np.mean(clean)))

    print(f"═══ Bateria H1-H10 · {used} ações · eventos agrupados + replicação ═══")
    print(f"{'célula':<38} {'n':>6} {'P(favor)':>9} {'p':>7} {'média ATR':>10} "
          f"{'p':>7} {'replica':>9}")
    print("-" * 94)

    payload = {}
    for name in ORDER:
        values = pooled.get(name)
        if not values:
            continue
        s = cell_stats(np.array(values))
        means = per_stock_mean.get(name, [])
        replication = (f"{sum(1 for m in means if m > 0)}/{len(means)}"
                       if means else "—")
        if s.get("sem_amostra"):
            print(f"{name:<38} {s['n']:>6} {'— sem amostra —':>34} {replication:>9}")
            continue
        passed = s["p_media"] < 0.005 and abs(s["media_atr"]) >= 0.03
        strong_rep = means and sum(1 for m in means if m > 0) / len(means) >= 0.7
        flag = ""
        if passed and strong_rep:
            flag = " ✅ PASSA + REPLICA"
        elif passed:
            flag = " ⚠️ passa agrupado, replicação fraca"
        elif strong_rep and len(means) >= 8:
            flag = " ◆ replica sem significância"
        print(f"{name:<38} {s['n']:>6} {s['favor']:>8.1%} {s['p_favor']:>7.3f} "
              f"{s['media_atr']:>+10.3f} {s['p_media']:>7.3f} {replication:>9}{flag}")
        payload[name] = {**s, "replica": replication}

    out = ROOT / "web" / "pa_battery_stocks.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nAviso de dependência: eventos do mesmo dia em papéis correlacionados")
    print(f"não são independentes — o p agrupado é otimista; a replicação corrige.")
    print(f"Exportado para {out}")


if __name__ == "__main__":
    main()
