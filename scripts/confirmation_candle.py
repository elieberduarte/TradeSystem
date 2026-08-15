"""O setup do usuário: candle de confirmação com entrada no rompimento.

Especificação pré-registrada, traduzida da descrição em conversa:

  CONTEXTO   os 2 candles anteriores têm saldo na direção do sinal
  CANDLE     corpo >= 60% do range e fechamento no quartil extremo
             (perto da máxima para compra, da mínima para venda)
  ENTRADA    ordem stop 1 tick além do extremo do candle de
             confirmação, válida por 3 candles (senão cancela);
             abertura além do gatilho executa na abertura (gap)
  STOP       1 tick além do extremo oposto do candle de confirmação
  ALVO       R:R em {1, 2, 3} — grade pré-declarada
  VARIANTE   filtro de "perna forte": eficiência de Kaufman dos
             últimos 12 candles >= 0,5 (o contexto dos prints)

Réguas: random walk com barreiras assimétricas
P_rw(alvo) = dist_stop / (dist_stop + dist_alvo); fricção honesta
(entrada a mercado paga slippage; stop também; alvo é limitada).
Empate intrabar (alvo E stop no mesmo candle) resolve como STOP.

Uso: python scripts/confirmation_candle.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.data.history import HistoryStore

ROOT = Path(__file__).resolve().parents[1]

SPEC = {
    "WIN$N": {"tick": 5.0, "slip": 10.0, "emol": 7.5},
    "WDO$N": {"tick": 0.5, "slip": 0.5, "emol": 0.2},
}
BODY_MIN = 0.60
CLOSE_QUARTILE = 0.75
VALID_BARS = 3
RRS = (1.0, 2.0, 3.0)
EFF_WINDOW = 12
EFF_MIN = 0.50


def efficiency(closes: np.ndarray) -> float:
    path = np.abs(np.diff(closes)).sum()
    return abs(closes[-1] - closes[0]) / path if path > 0 else 0.0


def signals_of_session(o, h, l, c) -> list[dict]:
    out = []
    n = len(c)
    for t in range(EFF_WINDOW + 3, n - 2):
        rng = h[t] - l[t]
        if rng <= 0:
            continue
        body = abs(c[t] - o[t])
        if body / rng < BODY_MIN:
            continue
        close_pos = (c[t] - l[t]) / rng
        eff = efficiency(c[t - EFF_WINDOW + 1 : t + 1])

        if c[t] > o[t] and close_pos >= CLOSE_QUARTILE and c[t - 1] > c[t - 3]:
            out.append({"t": t, "dir": 1, "eff": eff})
        elif c[t] < o[t] and close_pos <= 1 - CLOSE_QUARTILE and c[t - 1] < c[t - 3]:
            out.append({"t": t, "dir": -1, "eff": eff})
    return out


def simulate(o, h, l, c, sig, rr, spec) -> dict | None:
    """Executa um sinal: ordem stop, depois corrida alvo × stop."""
    t, d = sig["t"], sig["dir"]
    tick, slip = spec["tick"], spec["slip"]
    trigger = h[t] + tick if d > 0 else l[t] - tick
    protect = l[t] - tick if d > 0 else h[t] + tick

    fill = fill_bar = None
    for i in range(t + 1, min(t + 1 + VALID_BARS, len(c))):
        if d > 0 and o[i] >= trigger:
            fill, fill_bar = o[i], i
        elif d < 0 and o[i] <= trigger:
            fill, fill_bar = o[i], i
        elif d > 0 and h[i] >= trigger:
            fill, fill_bar = trigger, i
        elif d < 0 and l[i] <= trigger:
            fill, fill_bar = trigger, i
        if fill is not None:
            break
    if fill is None:
        return None

    entry = fill + d * slip                       # entrada a mercado paga slippage
    stop_dist = abs(entry - protect)
    if stop_dist <= 0:
        return None
    target = entry + d * rr * stop_dist

    # No candle da execução, só o stop conta (pior caso — sem look-ahead)
    if (d > 0 and l[fill_bar] <= protect) or (d < 0 and h[fill_bar] >= protect):
        exit_price = protect - d * slip
        return {"win": False, "pnl": d * (exit_price - entry), "hora": fill_bar}

    for i in range(fill_bar + 1, len(c)):
        hit_stop = l[i] <= protect if d > 0 else h[i] >= protect
        hit_target = h[i] >= target if d > 0 else l[i] <= target
        if hit_stop:                              # empate resolve como stop
            exit_price = protect - d * slip
            return {"win": False, "pnl": d * (exit_price - entry), "hora": fill_bar}
        if hit_target:                            # alvo é ordem limitada
            return {"win": True, "pnl": d * (target - entry), "hora": fill_bar}
    exit_price = c[-1] - d * slip                 # fim da sessão: zera a mercado
    return {"win": None, "pnl": d * (exit_price - entry), "hora": fill_bar}


def study(symbol: str, store: HistoryStore) -> list[dict]:
    spec = SPEC[symbol]
    candles = store.load(symbol, "5m")
    rows = []
    for _, bars in candles.groupby(candles.index.normalize()):
        if len(bars) < 40:
            continue
        o = bars["open"].to_numpy(float)
        h = bars["high"].to_numpy(float)
        l = bars["low"].to_numpy(float)
        c = bars["close"].to_numpy(float)
        hours = bars.index.hour
        for sig in signals_of_session(o, h, l, c):
            for rr in RRS:
                result = simulate(o, h, l, c, sig, rr, spec)
                if result is None or result["win"] is None:
                    continue
                rows.append({
                    "symbol": symbol, "rr": rr, "win": result["win"],
                    "pnl": result["pnl"] - spec["emol"],
                    "eff": sig["eff"], "manha": int(hours[result["hora"]]) < 13,
                })
    return rows


def report(frame: pd.DataFrame, label: str) -> list[dict]:
    out = []
    print(f"\n{'variante':<26} {'RR':>4} {'trades':>8} {'acerto':>8} {'RW prevê':>9} "
          f"{'edge':>7} {'EV líq/trade':>13}")
    print("-" * 82)
    for (rr,), group in frame.groupby(["rr"]):
        p_rw = 1 / (1 + rr)                    # barreiras: stop 1R, alvo RR
        for name, part in (("todos os sinais", group),
                           ("perna forte (eff≥0,5)", group[group["eff"] >= EFF_MIN]),
                           ("· manhã", group[group["eff"].ge(EFF_MIN) & group["manha"]]),
                           ("· tarde", group[group["eff"].ge(EFF_MIN) & ~group["manha"]])):
            if len(part) < 50:
                continue
            win = float(part["win"].mean())
            ev = float(part["pnl"].mean())
            print(f"{name:<26} {rr:>4.0f} {len(part):>8,} {win:>7.1%} {p_rw:>8.1%} "
                  f"{win - p_rw:>+6.1%} {ev:>+13.1f}")
            out.append({"variante": name, "rr": rr, "n": len(part),
                        "acerto": round(win, 4), "rw": round(p_rw, 4),
                        "ev_liq": round(ev, 2)})
    return out


def main() -> None:
    store = HistoryStore()
    payload = {}
    for symbol in ("WIN$N", "WDO$N"):
        rows = pd.DataFrame(study(symbol, store))
        print(f"\n═══ {symbol} · candle de confirmação · 5m · "
              f"{rows.groupby('rr').size().iloc[0]:,} sinais executados ═══")
        payload[symbol] = report(rows, symbol)

    out = ROOT / "web" / "confirmation_candle.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nEmpate intrabar resolve como STOP (conservador) · fricção: slippage na "
          f"entrada e no stop, alvo limitada, emolumentos por giro")
    print(f"Exportado para {out}")


if __name__ == "__main__":
    main()
