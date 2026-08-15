"""Micro-tendências intraday: a janela de 10-20 candles prevê algo?

A definição veio do usuário e virou fórmula sem tradução livre:

  MICRO-ALTA    a mínima da janela ocorreu há >= 2/3 da janela
                ("não renova fundo há X períodos") E o saldo é positivo
  MICRO-BAIXA   espelho
  SERROTE       as sequências de fechamentos na mesma direção duram
                <= 1,5 candle em média ("fecha um-dois e reverte")
                E a eficiência de Kaufman é < 0,2
  INDEFINIDO    o resto

Para cada barra classificada, três medições:

  1. PERSISTÊNCIA: o estado em t continua em t+janela?
  2. DIREÇÃO: corrida simétrica +-1 ATR a partir do fechamento — o
     lado do estado vem primeiro? (random walk simétrico = 50%)
  3. EXPECTATIVA: pontos por trade a favor do estado, menos fricção
     (WIN ~12,5 pts limitada / ~30 a mercado).

No serrote a aposta é outra: o próximo fechamento INVERTE o lado do
anterior? (alternância = a tese do "fecha um-dois e reverte").

Amostragem a cada 3 barras para reduzir sobreposição; corrida
resolvida dentro da própria sessão (sem atravessar a noite).

Uso: python scripts/micro_trends.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.data.history import HistoryStore
from src.bot.strategies.swing_reversion import atr

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (15, 10, 20)          # a primeira é a principal
STEP = 3
FRICTION = {"WIN$N": 12.5, "WDO$N": 1.25}


def mean_run_length(signs: np.ndarray) -> float:
    signs = signs[signs != 0]
    if len(signs) == 0:
        return float("nan")
    runs, current = [], 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i - 1]:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    return float(np.mean(runs))


def classify(closes: np.ndarray, lows: np.ndarray, highs: np.ndarray, W: int) -> str:
    net = closes[-1] - closes[0]
    path = np.abs(np.diff(closes)).sum()
    eff = abs(net) / path if path > 0 else 0.0
    age_low = W - 1 - int(np.argmin(lows))    # barras desde a mínima da janela
    age_high = W - 1 - int(np.argmax(highs))
    runs = mean_run_length(np.sign(np.diff(closes)))

    if age_low >= (2 * W) // 3 and net > 0:
        return "micro-alta"
    if age_high >= (2 * W) // 3 and net < 0:
        return "micro-baixa"
    if not np.isnan(runs) and runs <= 1.5 and eff < 0.2:
        return "serrote"
    return "indefinido"


def race(highs, lows, closes, start, up, down):
    """A partir de start: qual barreira vem primeiro? (+1 alta, -1 baixa)."""
    for i in range(start, len(highs)):
        hit_up, hit_down = highs[i] >= up, lows[i] <= down
        if hit_up and hit_down:
            return 1 if closes[i] >= (up + down) / 2 else -1
        if hit_up:
            return 1
        if hit_down:
            return -1
    return 0                                   # sessão acabou sem resolução


def study(symbol: str, store: HistoryStore, W: int) -> dict:
    candles = store.load(symbol, "5m")
    volatility = atr(candles, 14)
    rows = []

    for _, bars in candles.groupby(candles.index.normalize()):
        if len(bars) < W + 20:
            continue
        closes = bars["close"].to_numpy(float)
        highs = bars["high"].to_numpy(float)
        lows = bars["low"].to_numpy(float)
        vol = volatility.loc[bars.index].to_numpy(float)
        n = len(closes)

        states = {}
        for t in range(W, n - 3, STEP):
            state = classify(closes[t - W + 1 : t + 1], lows[t - W + 1 : t + 1],
                             highs[t - W + 1 : t + 1], W)
            states[t] = state
            if state == "indefinido" or vol[t] <= 0:
                continue

            radius = vol[t]
            outcome = race(highs, lows, closes, t + 1,
                           closes[t] + radius, closes[t] - radius)
            last_sign = np.sign(closes[t] - closes[t - 1])
            rows.append({
                "t": t, "estado": state, "radius": radius, "outcome": outcome,
                "last_sign": last_sign,
                "next_sign": np.sign(closes[t + 1] - closes[t]),
            })

        # persistência: o estado em t se repete ~W barras à frente?
        keys = sorted(states)
        for k in keys:
            nearest = min((x for x in keys if x >= k + W), default=None)
            if nearest is not None and nearest - (k + W) <= STEP:
                for row in rows[::-1]:
                    if row["t"] == k:
                        row["estado_futuro"] = states[nearest]
                        break

    frame = pd.DataFrame(rows)
    result = {"janela": W, "estados": {}}
    for state, group in frame.groupby("estado"):
        resolved = group[group["outcome"] != 0]
        direction = 1 if state == "micro-alta" else -1
        favor = float((resolved["outcome"] == direction).mean()) if len(resolved) else float("nan")
        mean_radius = float(group["radius"].mean())
        # EV por trade a favor do estado: acerta +R, erra -R
        ev = (2 * favor - 1) * mean_radius if not np.isnan(favor) else float("nan")

        with_future = group.dropna(subset=["estado_futuro"]) if "estado_futuro" in group else group.iloc[0:0]
        persistence = float((with_future["estado_futuro"] == state).mean()) if len(with_future) else float("nan")

        alt = float((group["next_sign"] == -group["last_sign"]).mean())
        result["estados"][state] = {
            "n": len(group),
            "persistencia": round(persistence, 3) if not np.isnan(persistence) else None,
            "p_favor": round(favor, 3) if not np.isnan(favor) else None,
            "raio_medio": round(mean_radius, 1),
            "ev_bruto_pts": round(ev, 1) if not np.isnan(ev) else None,
            "alternancia": round(alt, 3),
        }
    return result


def main() -> None:
    store = HistoryStore()
    report = {}
    for symbol in ("WIN$N", "WDO$N"):
        report[symbol] = []
        for W in WINDOWS:
            outcome = study(symbol, store, W)
            report[symbol].append(outcome)
            friction = FRICTION.get(symbol, 0)
            print(f"\n── {symbol} · janela de {W} candles de 5m · corrida ±1 ATR ──")
            print(f"{'estado':<12} {'amostras':>9} {'persiste':>9} {'P(favor)':>9} "
                  f"{'raio':>7} {'EV bruto':>9} {'EV−fricção':>11} {'alternância':>12}")
            for state, s in outcome["estados"].items():
                ev_net = s["ev_bruto_pts"] - friction if s["ev_bruto_pts"] is not None else None
                print(f"{state:<12} {s['n']:>9,} "
                      f"{s['persistencia'] if s['persistencia'] is not None else '—':>9} "
                      f"{s['p_favor'] if s['p_favor'] is not None else '—':>9} "
                      f"{s['raio_medio']:>7} "
                      f"{s['ev_bruto_pts'] if s['ev_bruto_pts'] is not None else '—':>9} "
                      f"{round(ev_net, 1) if ev_net is not None else '—':>11} "
                      f"{s['alternancia']:>12}")

    out = ROOT / "web" / "micro_trends.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nReferências: P(favor) do random walk = 50% · alternância da moeda = 50%")
    print(f"Fricção: WIN 12,5 pts (limitada) a ~30 (mercado) · WDO ~1,25 pts")
    print(f"Exportado para {out}")


if __name__ == "__main__":
    main()
