"""Executa a bateria pré-registrada de price action diário (H1-H10).

As definições, palpites e réguas estão seladas em
docs/pre_registro_price_action.md — este script só as executa.
Convenção: cada evento tem uma DIREÇÃO ESPERADA pela hipótese; o
desfecho é o retorno do dia seguinte (em ATRs) NA direção esperada.
P(favor) > 50% e média > 0 = a hipótese acerta o lado.

Uso: python scripts/pa_battery.py
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
MIN_N = 30


def p_norm(z: float) -> float:
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def cell_stats(values: np.ndarray) -> dict | None:
    """Estatística de uma célula: média em ATRs a favor + P(favor)."""
    values = values[~np.isnan(values)]
    n = len(values)
    if n < MIN_N:
        return {"n": n, "sem_amostra": True}
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    t = mean / (std / sqrt(n)) if std > 0 else 0.0
    favor = float((values > 0).mean())
    z_bin = (favor - 0.5) * sqrt(n) / 0.5
    return {"n": n, "favor": round(favor, 3), "p_favor": round(p_norm(z_bin), 4),
            "media_atr": round(mean, 3), "p_media": round(p_norm(t), 4)}


def collect_cells(df: pd.DataFrame) -> dict[str, list[float]]:
    """Coleta os eventos brutos das hipóteses H1-H10 num DataFrame diário."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = atr(df, 14).shift(1).to_numpy(float)
    n = len(c)

    r = np.full(n, np.nan)
    r[1:] = (c[1:] - c[:-1]) / v[1:]
    d_move = (c - o) / v
    rng = (h - l) / v
    span = np.where(h - l > 0, h - l, np.nan)
    pos = (c - l) / span
    high20 = pd.Series(h).rolling(20).max().shift(1).to_numpy()
    low20 = pd.Series(l).rolling(20).min().shift(1).to_numpy()
    ch_span = np.where(high20 - low20 > 0, high20 - low20, np.nan)
    ch_pos = (c - low20) / ch_span
    min10 = pd.Series(l).rolling(10).min().shift(1).to_numpy()
    argmin10 = pd.Series(l).rolling(10).apply(np.argmin, raw=True).shift(1).to_numpy()

    def next_r(t, dir_, horizon=1):
        if t + horizon >= n:
            return np.nan
        total = (c[t + horizon] - c[t]) / v[t + 1] if not np.isnan(v[t + 1]) else np.nan
        return dir_ * total

    cells: dict[str, list[float]] = {}

    def add(name, value):
        cells.setdefault(name, []).append(value)

    for t in range(21, n - 4):
        if np.isnan(v[t]) or np.isnan(r[t]) or np.isnan(pos[t]):
            continue

        # H1: martelo de verdade vs capitulação (ambos esperam reversão de alta)
        if l[t] <= c[t - 1] - v[t]:
            if pos[t] >= 0.67:
                add("H1 martelo (cai 1 ATR, fecha em cima)", next_r(t, +1))
            elif pos[t] <= 0.33:
                add("H1 capitulação (cai e fecha embaixo)", next_r(t, +1))

        # H2: momentum de fechamento no decil extremo
        if pos[t] >= 0.9:
            add("H2 fecha no decil superior", next_r(t, +1))
        elif pos[t] <= 0.1:
            add("H2 fecha no decil inferior", next_r(t, -1))

        # H3: range gigante (direção) e range anão (expansão)
        if rng[t] >= 2 and d_move[t] != 0:
            add("H3 range gigante continua", next_r(t, np.sign(d_move[t])))
        if rng[t] <= 0.5 and t + 1 < n:
            add("H3 range anão: expansão amanhã", rng[t + 1] - np.nanmean(rng[21:t + 1]))

        # H4: gap and go (gap >= 0,5 ATR que não fechou no dia)
        if o[t] >= c[t - 1] + 0.5 * v[t] and l[t] > c[t - 1]:
            add("H4 gap and go (3 dias)", next_r(t, +1, horizon=3))
        elif o[t] <= c[t - 1] - 0.5 * v[t] and h[t] < c[t - 1]:
            add("H4 gap and go (3 dias)", next_r(t, -1, horizon=3))

        # H5: bandeira de dois candles (impulso forte + inside day)
        if abs(d_move[t - 1]) >= 1 and h[t] < h[t - 1] and l[t] > l[t - 1]:
            add("H5 bandeira de 2 candles (D+1)", next_r(t, np.sign(d_move[t - 1])))
            add("H5 bandeira de 2 candles (3 dias)", next_r(t, np.sign(d_move[t - 1]), horizon=3))

        # H6: três fechamentos fortes do mesmo lado
        if min(pos[t], pos[t - 1], pos[t - 2]) >= 0.6:
            add("H6 três fechamentos compradores", next_r(t, +1))
        elif max(pos[t], pos[t - 1], pos[t - 2]) <= 0.4:
            add("H6 três fechamentos vendedores", next_r(t, -1))

        # H7: falso rompimento do canal de 20 (espera REVERSÃO)
        if not np.isnan(high20[t]) and h[t] > high20[t] and c[t] <= high20[t]:
            add("H7 falso rompimento p/ cima → cai?", next_r(t, -1))
        if not np.isnan(low20[t]) and l[t] < low20[t] and c[t] >= low20[t]:
            add("H7 falso rompimento p/ baixo → sobe?", next_r(t, +1))

        # H8: queda forte no fundo do canal vs no meio (espera reversão)
        if r[t] <= -1 and not np.isnan(ch_pos[t]):
            if ch_pos[t] <= 0.25:
                add("H8 queda forte no FUNDO do canal", next_r(t, +1))
            elif 0.25 < ch_pos[t] < 0.75:
                add("H8 queda forte no MEIO do canal", next_r(t, +1))

        # H9: o "V" de um dia (queda forte ontem, recuperação total hoje)
        if r[t - 1] <= -1 and c[t] > c[t - 2]:
            add("H9 'V' de um dia → continua?", next_r(t, +1))

        # H10: fundo duplo objetivo (reteste da mínima de 10, formada há 3-7 dias)
        if (not np.isnan(min10[t]) and not np.isnan(argmin10[t])):
            age = 9 - int(argmin10[t])          # barras desde a mínima na janela
            if 3 <= age <= 7 and abs(l[t] - min10[t]) <= 0.15 * v[t] and c[t] > min10[t]:
                add("H10 fundo duplo → repica?", next_r(t, +1))

    return cells


def run_symbol(symbol: str, store: HistoryStore) -> dict:
    cells = collect_cells(store.load(symbol, "1d"))
    return {name: cell_stats(np.array(values)) for name, values in cells.items()}


ORDER = [
    "H1 martelo (cai 1 ATR, fecha em cima)", "H1 capitulação (cai e fecha embaixo)",
    "H2 fecha no decil superior", "H2 fecha no decil inferior",
    "H3 range gigante continua", "H3 range anão: expansão amanhã",
    "H4 gap and go (3 dias)",
    "H5 bandeira de 2 candles (D+1)", "H5 bandeira de 2 candles (3 dias)",
    "H6 três fechamentos compradores", "H6 três fechamentos vendedores",
    "H7 falso rompimento p/ cima → cai?", "H7 falso rompimento p/ baixo → sobe?",
    "H8 queda forte no FUNDO do canal", "H8 queda forte no MEIO do canal",
    "H9 'V' de um dia → continua?", "H10 fundo duplo → repica?",
]


def main() -> None:
    store = HistoryStore()
    payload = {}
    for symbol in ("WIN$N", "WDO$N"):
        results = run_symbol(symbol, store)
        payload[symbol] = results
        print(f"\n═══ {symbol} · bateria pré-registrada (a favor = lado que a hipótese prevê) ═══")
        print(f"{'célula':<38} {'n':>5} {'P(favor)':>9} {'p':>7} {'média ATR':>10} {'p':>7}")
        print("-" * 84)
        for name in ORDER:
            s = results.get(name)
            if s is None:
                continue
            if s.get("sem_amostra"):
                print(f"{name:<38} {s['n']:>5} {'— sem amostra (mín. 30) —':>44}")
                continue
            passed = (s["p_media"] < 0.005 and abs(s["media_atr"]) >= 0.03)
            flag = " ✅ PASSA A RÉGUA" if passed else ""
            print(f"{name:<38} {s['n']:>5} {s['favor']:>8.1%} {s['p_favor']:>7.3f} "
                  f"{s['media_atr']:>+10.3f} {s['p_media']:>7.3f}{flag}")

    out = ROOT / "web" / "pa_battery.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nRégua selada no pré-registro: p < 0,005 E |média| ≥ 0,03 ATR; sobrevivente")
    print(f"vai à replicação nos 28 instrumentos. Exportado para {out}")


if __name__ == "__main__":
    main()
