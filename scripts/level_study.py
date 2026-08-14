"""Os níveis objetivos existem? E rompem mesmo após 2-3 testes?

Mede a taxa de repique em toques de níveis clássicos (YH, YL, pivô,
R1/S1, R2/S2, redondos) contra níveis PLACEBO deslocados, em WIN e
WDO intraday. Depois mede P(rompimento | k-ésimo toque) para testar
a regra do Madang.

Uso: python scripts/level_study.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.bot.analysis.levels import study_levels, two_proportion_z
from src.bot.data.history import HistoryStore

ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    ("WIN$N", "5m", 1000.0),
    ("WIN$N", "15m", 1000.0),
    ("WDO$N", "5m", 10.0),
    ("WDO$N", "15m", 10.0),
]


def bounce_stats(frame: pd.DataFrame) -> tuple[int, float]:
    if frame.empty:
        return 0, float("nan")
    return len(frame), float((~frame["broke"]).mean())


def main() -> None:
    store = HistoryStore()
    all_touches = []

    print("═══ Estudo de níveis · repique real × placebo ═══")
    print("Repique = preço afasta 25% do range de ontem antes de romper 25% além\n")
    print(f"{'dataset':<14} {'toques reais':>12} {'repique':>8} {'placebos':>9} "
          f"{'repique':>8} {'diferença':>10} {'z':>6} {'p':>7}")
    print("-" * 82)

    for symbol, timeframe, round_step in DATASETS:
        candles = store.load(symbol, timeframe)
        if candles is None:
            print(f"{symbol} {timeframe}: sem dados")
            continue
        touches = study_levels(candles, round_step=round_step)
        touches["symbol"], touches["timeframe"] = symbol, timeframe
        all_touches.append(touches)

        real = touches[~touches["placebo"]]
        fake = touches[touches["placebo"]]
        n_r, bounce_r = bounce_stats(real)
        n_f, bounce_f = bounce_stats(fake)
        z, p = two_proportion_z(
            int((~real["broke"]).sum()), n_r, int((~fake["broke"]).sum()), n_f
        )
        print(f"{symbol + ' ' + timeframe:<14} {n_r:>12,} {bounce_r:>7.1%} "
              f"{n_f:>9,} {bounce_f:>7.1%} {bounce_r - bounce_f:>+9.1%} "
              f"{z:>6.2f} {p:>7.4f}")

    combined = pd.concat(all_touches, ignore_index=True)

    # ── Por tipo de nível (agregado nos 4 datasets) ──
    print("\n── Por tipo de nível (todos os datasets) ──")
    fake_all = combined[combined["placebo"]]
    n_f, bounce_f = bounce_stats(fake_all)
    print(f"{'nível':<8} {'toques':>8} {'repique':>8} {'vs placebo':>11} {'z':>6} {'p':>7}")
    per_level = []
    for name, group in combined[~combined["placebo"]].groupby("level_name"):
        n, bounce = bounce_stats(group)
        z, p = two_proportion_z(int((~group["broke"]).sum()), n,
                                int((~fake_all["broke"]).sum()), n_f)
        per_level.append({"nivel": name, "toques": n, "repique": round(bounce, 4),
                          "diff": round(bounce - bounce_f, 4), "z": round(z, 2),
                          "p": round(p, 4)})
        print(f"{name:<8} {n:>8,} {bounce:>7.1%} {bounce - bounce_f:>+10.1%} "
              f"{z:>6.2f} {p:>7.4f}")
    print(f"{'PLACEBO':<8} {n_f:>8,} {bounce_f:>7.1%} {'—':>11}")

    # ── Madang: P(rompimento | k-ésimo toque) ──
    print("\n── P(rompimento | k-ésimo toque do dia) ──")
    print(f"{'toque':<7} {'reais':>8} {'P(rompe)':>9} {'placebos':>9} {'P(rompe)':>9}")
    madang = []
    for k in (1, 2, 3, 4, 5):
        if k < 5:
            real_k = combined[(~combined["placebo"]) & (combined["index"] == k)]
            fake_k = combined[(combined["placebo"]) & (combined["index"] == k)]
            label = f"{k}º"
        else:
            real_k = combined[(~combined["placebo"]) & (combined["index"] >= k)]
            fake_k = combined[(combined["placebo"]) & (combined["index"] >= k)]
            label = "5º+"
        n_r = len(real_k)
        n_f2 = len(fake_k)
        break_r = float(real_k["broke"].mean()) if n_r else float("nan")
        break_f = float(fake_k["broke"].mean()) if n_f2 else float("nan")
        madang.append({"toque": label, "reais": n_r, "p_rompe_real": round(break_r, 4),
                       "placebos": n_f2, "p_rompe_placebo": round(break_f, 4)})
        print(f"{label:<7} {n_r:>8,} {break_r:>8.1%} {n_f2:>9,} {break_f:>8.1%}")

    out = ROOT / "web" / "level_study.json"
    out.write_text(
        json.dumps({"por_nivel": per_level, "madang": madang}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
