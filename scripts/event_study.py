"""Estudo de eventos (PEAD) — executa docs/pre_registro_eventos.md.

Para cada evento CVM (fato relevante ou comunicado de resultado) com
data de entrega D e papel com preço no acervo:
  r_reacao = ret(D → D+1) em ATR(14)     (convenção conservadora)
  r_deriva = ret(D+1 → D+15) em ATR
Filtro: |r_reacao| >= 1 ATR. Desfecho: deriva NA DIREÇÃO da reação.
Placebo: mesma quantidade de dias aleatórios por papel/ano com o
mesmo filtro (|r| >= 1 ATR) — separa "informação do evento" de
"momentum de qualquer dia forte".

Uso: python scripts/event_study.py
"""

import json
import sys
from math import erf, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.data.history import HistoryStore
from src.bot.strategies.swing_reversion import atr

ROOT = Path(__file__).resolve().parents[1]
REACTION_MIN = 1.0
DRIFT_DAYS = 14
RNG = np.random.default_rng(20260817)


def p_norm(z: float) -> float:
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def cell(values: np.ndarray) -> dict:
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 30:
        return {"n": n, "sem_amostra": True}
    mean = float(values.mean())
    se = float(values.std(ddof=1) / sqrt(n)) if n > 1 else float("nan")
    favor = float((values > 0).mean())
    return {
        "n": n, "favor": round(favor, 3),
        "p_favor": round(p_norm((favor - 0.5) * sqrt(n) / 0.5), 4),
        "media_atr": round(mean, 3),
        "p_media": round(p_norm(mean / se), 4) if se and se > 0 else 1.0,
    }


def load_events() -> pd.DataFrame:
    events = pd.read_parquet(ROOT / "data" / "cvm" / "eventos_liquid.parquet")
    cat = events["categoria"].fillna("")
    events["tipo"] = np.where(cat.str.contains("Fato Relevante", case=False), "fato",
                     np.where(cat.str.contains("Econômico-Financeiros", case=False), "resultado", ""))
    events = events[events["tipo"] != ""].copy()
    events["data"] = pd.to_datetime(events["data_entrega"]).dt.normalize()
    # um papel pode entregar vários documentos no mesmo dia: conta uma vez
    return events.drop_duplicates(subset=["ticker", "data", "tipo"])[["ticker", "data", "tipo", "assunto"]]


def measure(daily: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Reação D→D+1 e deriva D+1→D+15 (em ATR) para uma lista de datas."""
    closes = daily["close"]
    vol = atr(daily, 14).shift(1)
    idx = daily.index
    pos = idx.get_indexer(dates)
    rows = []
    for d, i in zip(dates, pos):
        if i < 20 or i + 1 + DRIFT_DAYS >= len(idx):
            continue
        a = float(vol.iloc[i + 1])
        if not a or np.isnan(a):
            continue
        reaction = (closes.iloc[i + 1] - closes.iloc[i]) / a
        drift = (closes.iloc[i + 1 + DRIFT_DAYS] - closes.iloc[i + 1]) / a
        rows.append({"data": d, "reacao": reaction, "deriva": drift})
    return pd.DataFrame(rows)


def main() -> None:
    store = HistoryStore(ROOT / "data")
    events = load_events()
    tickers = sorted(events["ticker"].unique())

    real_rows, placebo_rows = [], []
    used = 0
    for ticker in tickers:
        daily = store.load(ticker, "1d")
        if daily is None or len(daily) < 400:
            continue
        used += 1
        ev = events[events["ticker"] == ticker]
        # datas de evento que existem no calendário do papel (ou o pregão seguinte)
        dates = pd.DatetimeIndex(sorted(set(ev["data"])))
        aligned = daily.index[daily.index.get_indexer(dates, method="bfill").clip(0)]
        measured = measure(daily, pd.DatetimeIndex(aligned))
        if measured.empty:
            continue
        measured["ticker"] = ticker
        # tipo por data (se o mesmo dia tem fato E resultado, marca 'resultado')
        tipo_map = ev.groupby("data")["tipo"].agg(lambda s: "resultado" if "resultado" in set(s) else "fato")
        aligned_map = dict(zip(pd.DatetimeIndex(aligned), [tipo_map.get(d, "fato") for d in dates]))
        measured["tipo"] = measured["data"].map(aligned_map).fillna("fato")
        real_rows.append(measured)

        # placebo: mesma quantidade de dias aleatórios (fora dos dias de evento)
        candidates = daily.index[25:-(DRIFT_DAYS + 2)]
        candidates = candidates[~candidates.isin(pd.DatetimeIndex(aligned))]
        k = min(len(measured), len(candidates))
        sample = pd.DatetimeIndex(RNG.choice(candidates, size=k, replace=False))
        pl = measure(daily, sample)
        pl["ticker"] = ticker
        placebo_rows.append(pl)

    real = pd.concat(real_rows, ignore_index=True)
    placebo = pd.concat(placebo_rows, ignore_index=True)
    for frame in (real, placebo):
        frame["forte"] = frame["reacao"].abs() >= REACTION_MIN
        frame["a_favor"] = frame["deriva"] * np.sign(frame["reacao"])

    print(f"═══ Estudo de eventos · {used} papéis · {len(real):,} eventos medidos "
          f"({int(real['forte'].sum()):,} com reação forte) · placebo {int(placebo['forte'].sum()):,} ═══\n")
    print(f"{'célula':<40} {'n':>6} {'P(a favor)':>11} {'p':>7} {'deriva ATR':>11} {'p':>7} {'replica':>9}")
    print("-" * 98)

    def replication(frame: pd.DataFrame) -> str:
        by = frame.groupby("ticker")["a_favor"].agg(["mean", "size"])
        by = by[by["size"] >= 5]
        if len(by) < 5:
            return "—"
        return f"{int((by['mean'] > 0).sum())}/{len(by)}"

    report = {}
    strong = real[real["forte"]]
    strong_pl = placebo[placebo["forte"]]
    cells = [
        ("H1 eventos, reação forte (todos)", strong),
        ("H2 · só RESULTADOS", strong[strong["tipo"] == "resultado"]),
        ("H2 · só FATOS RELEVANTES", strong[strong["tipo"] == "fato"]),
        ("H3 · reação POSITIVA", strong[strong["reacao"] > 0]),
        ("H3 · reação NEGATIVA", strong[strong["reacao"] < 0]),
        ("H4 PLACEBO dias fortes sem evento", strong_pl),
        ("H4 · placebo positivo", strong_pl[strong_pl["reacao"] > 0]),
        ("H4 · placebo negativo", strong_pl[strong_pl["reacao"] < 0]),
        ("(ref) eventos com reação FRACA", real[~real["forte"]]),
    ]
    for label, frame in cells:
        s = cell(frame["a_favor"].to_numpy(float))
        rep = replication(frame) if not s.get("sem_amostra") else "—"
        report[label] = {**s, "replica": rep}
        if s.get("sem_amostra"):
            print(f"{label:<40} {s['n']:>6} {'— sem amostra —':>40}")
            continue
        flag = " ◀" if (s["p_media"] < 0.01 and abs(s["media_atr"]) >= 0.10) else ""
        print(f"{label:<40} {s['n']:>6} {s['favor']:>10.1%} {s['p_favor']:>7.3f} "
              f"{s['media_atr']:>+11.3f} {s['p_media']:>7.3f} {rep:>9}{flag}")

    # Diferença evento − placebo (o valor da informação)
    ev_mean = strong["a_favor"].mean(); pl_mean = strong_pl["a_favor"].mean()
    se = sqrt(strong["a_favor"].var(ddof=1) / len(strong) + strong_pl["a_favor"].var(ddof=1) / len(strong_pl))
    z = (ev_mean - pl_mean) / se
    print(f"\nEvento − placebo: {ev_mean - pl_mean:+.3f} ATR · z = {z:.2f} · p = {p_norm(z):.4f}")
    report["evento_menos_placebo"] = {"diff_atr": round(ev_mean - pl_mean, 3), "z": round(z, 2), "p": round(p_norm(z), 4)}

    real.to_parquet(ROOT / "data" / "event_study.parquet")
    out = ROOT / "web" / "event_study.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nRégua: p < 0,01 E |deriva| ≥ 0,10 ATR (◀) · replicação ≥ 60% dos papéis")
    print(f"Exportado para {out}")


if __name__ == "__main__":
    main()
