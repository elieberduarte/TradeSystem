"""O último teste do capítulo intraday: sinal aprovado + montaria calma.

A pilha que produziu o primeiro edge positivo do funil (floresta a
favor + perna forte + candle de confirmação) morreu na execução: a
ordem stop acima do estouro compra o topo do burst. Aqui a montaria
troca de natureza, com regras pré-declaradas:

  ENTRADA  no FECHAMENTO do candle de confirmação (a mercado, sem
           perseguir — o preço que existe, não o topo do estouro)
  ALVO     nenhum (a micro-reversão não tem o que alcançar)
  SAÍDA    no fechamento da sessão — onde os +111 pts do estudo das
           14h moram
  STOP     variante A: sem stop (o valor puro do sinal)
           variante B: desastre, a 2× o range do candle além do
           extremo oposto (fora do alcance da respiração de 5m)

Fricção integral: slippage na entrada e na saída a mercado +
emolumentos. Recortes: alinhamento com a floresta × perna forte ×
manhã/tarde.

Uso: python scripts/calm_mount.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.confirmation_candle import SPEC, signals_of_session
from scripts.forest_bias import daily_bias
from src.bot.data.history import HistoryStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    store = HistoryStore()
    payload = {}

    for symbol in ("WIN$N", "WDO$N"):
        spec = SPEC[symbol]
        slip, emol = spec["slip"], spec["emol"]
        bias_by_day = daily_bias(store.load(symbol, "1d"))
        bias_by_day.index = bias_by_day.index.normalize()

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
            hours = bars.index.hour
            n = len(c)

            for sig in signals_of_session(o, h, l, c):
                t, d = sig["t"], sig["dir"]
                if n - t < 6:
                    continue
                if bias == "neutro":
                    align = "neutro"
                elif (bias == "alta") == (d > 0):
                    align = "a favor"
                else:
                    align = "contra"

                entry = c[t] + d * slip
                candle_range = h[t] - l[t]
                stop_level = (l[t] - 2 * candle_range) if d > 0 else (h[t] + 2 * candle_range)

                # Variante A: sem stop, carrega até o fim da sessão
                pnl_a = d * ((c[-1] - d * slip) - entry) - emol

                # Variante B: stop de desastre a caminho do fechamento
                pnl_b = None
                for i in range(t + 1, n):
                    hit = l[i] <= stop_level if d > 0 else h[i] >= stop_level
                    if hit:
                        pnl_b = d * ((stop_level - d * slip) - entry) - emol
                        break
                if pnl_b is None:
                    pnl_b = pnl_a

                rows.append({"align": align, "eff": sig["eff"],
                             "manha": int(hours[t]) < 13,
                             "sem_stop": pnl_a, "com_stop": pnl_b})

        frame = pd.DataFrame(rows)
        print(f"\n═══ {symbol} · montaria calma: entrada no fechamento, saída no fim do dia ═══")
        print(f"{'recorte':<26} {'variante':<10} {'trades':>7} {'% positiva':>10} "
              f"{'EV líq/trade':>13} {'total (pts)':>12}")
        print("-" * 84)
        report = []
        cuts = [
            ("a favor", frame[frame["align"] == "a favor"]),
            ("a favor + perna forte", frame[(frame["align"] == "a favor") & (frame["eff"] >= 0.5)]),
            ("· manhã", frame[(frame["align"] == "a favor") & (frame["eff"] >= 0.5) & frame["manha"]]),
            ("· tarde", frame[(frame["align"] == "a favor") & (frame["eff"] >= 0.5) & ~frame["manha"]]),
            ("contra + perna forte", frame[(frame["align"] == "contra") & (frame["eff"] >= 0.5)]),
            ("neutro + perna forte", frame[(frame["align"] == "neutro") & (frame["eff"] >= 0.5)]),
        ]
        for label, part in cuts:
            if len(part) < 50:
                continue
            for variant in ("sem_stop", "com_stop"):
                values = part[variant]
                ev = float(values.mean())
                print(f"{label:<26} {variant:<10} {len(part):>7,} "
                      f"{float((values > 0).mean()):>9.1%} {ev:>+13.1f} {values.sum():>+12,.0f}")
                report.append({"recorte": label, "variante": variant, "n": len(part),
                               "positiva": round(float((values > 0).mean()), 4),
                               "ev_liq": round(ev, 2), "total": round(float(values.sum()), 0)})
        payload[symbol] = report

    out = ROOT / "web" / "calm_mount.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nFricção integral: slippage na entrada e na saída a mercado + emolumentos")
    print(f"Exportado para {out}")


if __name__ == "__main__":
    main()
