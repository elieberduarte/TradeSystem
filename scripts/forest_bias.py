"""A floresta antes das folhas: o viés DIÁRIO condiciona o gatilho de 5m?

A hipótese do usuário, dita com todas as letras: "não adianta operar
um gatilho de compra em um mercado em queda". O viés é calculado só
com dados até o PREGÃO ANTERIOR (zero look-ahead):

  posição no canal   onde o fechamento de ontem está dentro do canal
                     de 20 dias (0 = na mínima, 1 = na máxima)
  saldo de 5 dias    fechamento de ontem vs 5 pregões atrás, em ATRs

  VIÉS DE BAIXA  posição <= 0,30 e saldo negativo
  VIÉS DE ALTA   posição >= 0,70 e saldo positivo
  NEUTRO         o resto

Por que essa escala: o regime diário tem persistência medida (94,7%
em 1 dia, 68,8% em 10); a micro-tendência de 15 candles tem 37%.
A floresta existe no diário — a pergunta é se ela desce até as
folhas: os sinais do candle de confirmação (estudo anterior) ficam
melhores quando ALINHADOS ao viés e piores quando CONTRA?

Uso: python scripts/forest_bias.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.confirmation_candle import SPEC, signals_of_session, simulate
from src.bot.data.history import HistoryStore
from src.bot.strategies.swing_reversion import atr

ROOT = Path(__file__).resolve().parents[1]
RRS = (1.0, 2.0, 3.0)


def daily_bias(daily: pd.DataFrame) -> pd.Series:
    """Viés de cada DATA usando apenas dados até o dia anterior."""
    close = daily["close"]
    high20 = daily["high"].rolling(20).max()
    low20 = daily["low"].rolling(20).min()
    span = (high20 - low20).replace(0, float("nan"))
    position = (close - low20) / span
    net5 = (close - close.shift(5)) / atr(daily, 14)

    bias = pd.Series("neutro", index=daily.index)
    bias[(position <= 0.30) & (net5 < 0)] = "baixa"
    bias[(position >= 0.70) & (net5 > 0)] = "alta"
    return bias.shift(1).fillna("neutro")     # o viés de HOJE vem de ONTEM


def main() -> None:
    store = HistoryStore()
    payload = {}

    for symbol in ("WIN$N", "WDO$N"):
        spec = SPEC[symbol]
        daily = store.load(symbol, "1d")
        bias_by_day = daily_bias(daily)
        bias_by_day.index = bias_by_day.index.normalize()

        # Sanidade da floresta: o viés de ontem prevê o dia de hoje?
        next_ret = daily["close"].pct_change().shift(-1)
        sane = pd.DataFrame({"bias": daily_bias(daily), "ret": daily["close"].pct_change()}).dropna()
        print(f"\n═══ {symbol} · a floresta (viés diário) ═══")
        for state, group in sane.groupby("bias"):
            up = float((group["ret"] > 0).mean())
            print(f"  viés {state:<7} {len(group):>5} dias · dia seguinte sobe {up:.1%}")

        candles = store.load(symbol, "5m")
        rows = []
        for day, bars in candles.groupby(candles.index.normalize()):
            if len(bars) < 40:
                continue
            bias = bias_by_day.get(day, "neutro")
            o = bars["open"].to_numpy(float)
            h = bars["high"].to_numpy(float)
            l = bars["low"].to_numpy(float)
            c = bars["close"].to_numpy(float)
            for sig in signals_of_session(o, h, l, c):
                if bias == "neutro":
                    align = "neutro"
                elif (bias == "alta") == (sig["dir"] > 0):
                    align = "a favor"
                else:
                    align = "contra"
                for rr in RRS:
                    result = simulate(o, h, l, c, sig, rr, spec)
                    if result is None or result["win"] is None:
                        continue
                    rows.append({"rr": rr, "align": align, "win": result["win"],
                                 "pnl": result["pnl"] - spec["emol"],
                                 "eff": sig["eff"]})

        frame = pd.DataFrame(rows)
        print(f"\n── {symbol} · gatilho de confirmação recortado pela floresta ──")
        print(f"{'alinhamento':<22} {'RR':>4} {'trades':>8} {'acerto':>8} "
              f"{'RW':>7} {'edge':>7} {'EV líq':>8}")
        print("-" * 70)
        report = []
        for rr in RRS:
            p_rw = 1 / (1 + rr)
            part_rr = frame[frame["rr"] == rr]
            for align in ("a favor", "contra", "neutro"):
                part = part_rr[part_rr["align"] == align]
                strong = part[part["eff"] >= 0.5]
                for label, subset in ((align, part), (f"{align} + perna forte", strong)):
                    if len(subset) < 50:
                        continue
                    win = float(subset["win"].mean())
                    ev = float(subset["pnl"].mean())
                    print(f"{label:<22} {rr:>4.0f} {len(subset):>8,} {win:>7.1%} "
                          f"{p_rw:>6.1%} {win - p_rw:>+6.1%} {ev:>+8.1f}")
                    report.append({"align": label, "rr": rr, "n": len(subset),
                                   "acerto": round(win, 4), "rw": round(p_rw, 4),
                                   "ev_liq": round(ev, 2)})
        payload[symbol] = report

    out = ROOT / "web" / "forest_bias.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
