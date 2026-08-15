"""Defesa de zona: o fundo/topo da própria sessão é defendido de verdade?

O trade clássico de price action: o preço fez um fundo, subiu,
voltou à zona do fundo — se defender, entra ali com stop curto. O
estudo dos níveis estáticos (YH, pivô...) mostrou efeito de +1,4
p.p. sobre placebo; aqui a pergunta é sobre os níveis DINÂMICOS que
a própria sessão cria, que são os que o trader da tela realmente
usa — e que têm a tese mais forte (stops e memória recentes).

Desenho:
  · Topos e fundos pelo zigzag CAUSAL (a zona só existe a partir da
    barra em que o pivô fica conhecível).
  · A zona é testada apenas pelo lado de onde nasceu, e a corrida
    repique×rompimento usa os mesmos parâmetros do estudo de níveis
    (banda de toque 5% do range típico; corrida = o próprio limiar
    do zigzag, 15% — "defender" significa lançar uma perna nova).
  · PLACEBO: a mesma zona deslocada ±{13,21,34}% do range típico,
    nascendo no mesmo instante, testada pelo mesmo lado.
  · RANDOM WALK: o benchmark decisivo. O toque acontece em
    nível+banda, o repique exige (corrida−banda) e o rompimento
    (corrida+banda) — random walk prevê
    P(repique) = (corrida+banda) / (2·corrida), sem informação alguma.

VEREDITO (registrado após a primeira rodada): a taxa de repique das
zonas REAIS bate com o random walk em ~1 p.p. nas três geometrias e
nos dois símbolos (68,2% vs 66,7% · 74,2% vs 75,0% · 60,8% vs
60,0%). A "defesa" é geometria, não memória. O placebo, aqui, é um
controle CONTAMINADO: ele só é tocado em estados adversos (na maior
parte, no meio do rompimento de uma zona real), por isso repica
menos que o random walk — os +20-27 p.p. "sobre o placebo" eram
artefato do controle, não vantagem da zona. Fica o aviso de método:
sempre confrontar com o random walk antes de celebrar.

Uso: python scripts/zone_defense.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.bot.analysis.levels import two_proportion_z, walk_zone
from src.bot.analysis.swings import swing_pivots
from src.bot.data.history import HistoryStore

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.15, 0.10, 0.25)          # limiar do zigzag; o 1º é o principal
BAND_FRAC = 0.05
SHIFTS = (0.13, 0.21, 0.34)


def collect(symbol: str, store: HistoryStore, threshold_frac: float) -> pd.DataFrame:
    candles = store.load(symbol, "5m")
    daily_range = (
        candles["high"].groupby(candles.index.normalize()).max()
        - candles["low"].groupby(candles.index.normalize()).min()
    )
    typical = daily_range.rolling(20).mean().shift(1)

    rows = []
    for day, bars in candles.groupby(candles.index.normalize()):
        base = typical.get(day)
        if base is None or pd.isna(base) or len(bars) < 60:
            continue
        threshold = threshold_frac * float(base)
        band = BAND_FRAC * float(base)
        bars = bars.reset_index(drop=True)
        pivots = swing_pivots(bars["close"], threshold)
        if not pivots:
            continue
        real_prices = [p.price for p in pivots]

        candidates = []
        placed: list[float] = []
        for pivot in pivots:
            side = "above" if pivot.kind == "fundo" else "below"
            start = pivot.confirm_index + 1
            candidates.append((pivot.kind, pivot.price, side, start, False))
            for frac in SHIFTS:
                for shifted in (pivot.price - frac * base, pivot.price + frac * base):
                    if min(abs(shifted - rp) for rp in real_prices) < 2 * band:
                        continue
                    if placed and min(abs(shifted - pp) for pp in placed) < 2 * band:
                        continue
                    placed.append(shifted)
                    candidates.append((pivot.kind, shifted, side, start, True))

        for kind, level, side, start, is_placebo in candidates:
            events = walk_zone(bars, level, band, threshold, start, side)
            for k, (bar, broke) in enumerate(events, start=1):
                rows.append({
                    "day": day, "kind": kind, "k": k,
                    "broke": broke, "placebo": is_placebo,
                })
    return pd.DataFrame(rows)


def line(label: str, real: pd.DataFrame, fake: pd.DataFrame) -> dict:
    n_r, n_f = len(real), len(fake)
    if n_r == 0 or n_f == 0:
        return {}
    bounce_r = float((~real["broke"]).mean())
    bounce_f = float((~fake["broke"]).mean())
    z, p = two_proportion_z(
        int((~real["broke"]).sum()), n_r, int((~fake["broke"]).sum()), n_f
    )
    print(f"{label:<18} {n_r:>7,} {bounce_r:>8.1%} {n_f:>9,} {bounce_f:>8.1%} "
          f"{bounce_r - bounce_f:>+9.1%} {z:>6.2f} {p:>8.4f}")
    return {"label": label, "reais": n_r, "repique_real": round(bounce_r, 3),
            "placebos": n_f, "repique_placebo": round(bounce_f, 3),
            "z": round(z, 2), "p": round(p, 4)}


def main() -> None:
    store = HistoryStore()
    report = {}

    print("═══ Defesa de zona · fundos/topos da própria sessão × placebo ═══")
    print("repique = a zona lança uma perna do tamanho do limiar antes de romper\n")

    for symbol in ("WIN$N", "WDO$N"):
        report[symbol] = {}
        for threshold in THRESHOLDS:
            frame = collect(symbol, store, threshold)
            if frame.empty:
                continue
            real = frame[~frame["placebo"]]
            fake = frame[frame["placebo"]]

            random_walk = (threshold + BAND_FRAC) / (2 * threshold)
            print(f"── {symbol} · limiar {threshold:.0%} do range típico · "
                  f"random walk prevê repique de {random_walk:.1%} ──")
            print(f"{'grupo':<18} {'toques':>7} {'repique':>8} {'placebos':>9} "
                  f"{'repique':>8} {'diferença':>9} {'z':>6} {'p':>8}")
            entry = {
                "random_walk": round(random_walk, 3),
                "geral": line("todas as zonas", real, fake),
                "fundos": line("fundos (suporte)", real[real["kind"] == "fundo"],
                               fake[fake["kind"] == "fundo"]),
                "topos": line("topos (resist.)", real[real["kind"] == "topo"],
                              fake[fake["kind"] == "topo"]),
                "primeiro_reteste": line("1º reteste", real[real["k"] == 1],
                                         fake[fake["k"] == 1]),
            }
            report[symbol][f"{threshold:.2f}"] = entry
            print()

    out = ROOT / "web" / "zone_defense.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Exportado para {out}")


if __name__ == "__main__":
    main()
