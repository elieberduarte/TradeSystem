"""Topos e fundos: a estrutura HH/HL confirmada prevê o resto do dia?

A crítica que motivou o estudo: analisar horários fixos mede o tempo,
não o preço. Aqui o gatilho é o EVENTO estrutural — o momento em que
um zigzag causal (pivô só existe após o limiar de confirmação)
completa topos e fundos ascendentes (ou descendentes) dentro da
sessão.

Para saber se a estrutura AGREGA informação, cada evento é comparado
com o controle ingênuo pareado por horário: entrar naquele mesmo
bloco de 30 min, na direção da sessão até ali, em todos os dias. Se
"topos e fundos" forem só um jeito bonito de dizer "o dia está de
alta", o excesso sobre o controle será zero.

Limiar de reversão pré-declarado: 15% do range médio dos últimos 20
pregões (robustez em 10% e 25%). Eventos nos últimos 30 min do dia
são descartados (não sobra dia para medir).

Uso: python scripts/swing_structure.py
"""

import json
import sys
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.analysis.swings import structure_events, swing_pivots
from src.bot.data.history import HistoryStore

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.15, 0.10, 0.25)     # o primeiro é o principal
MIN_BARS_LEFT = 6


def bucket_of(ts: pd.Timestamp) -> str:
    minute = 0 if ts.minute < 30 else 30
    return f"{ts.hour:02d}:{minute:02d}"


def control_table(sessions: dict) -> pd.DataFrame:
    """Entrada ingênua: a cada 30 min, na direção da sessão, todo dia."""
    rows = []
    for day, bars in sessions.items():
        closes = bars["close"]
        open_session = float(bars["open"].iloc[0])
        close_end = float(closes.iloc[-1])
        marks = bars.groupby([bucket_of(t) for t in bars.index]).tail(1)
        for ts, row in marks.iterrows():
            remaining = len(bars) - bars.index.get_loc(ts) - 1
            if remaining < MIN_BARS_LEFT:
                continue
            direction = np.sign(float(row["close"]) - open_session)
            if direction == 0:
                continue
            rows.append({
                "bucket": bucket_of(ts),
                "move": (close_end - float(row["close"])) * direction,
                "continuou": np.sign(close_end - float(row["close"])) == direction,
            })
    frame = pd.DataFrame(rows)
    return frame.groupby("bucket").agg(
        n=("move", "size"), move=("move", "mean"), cont=("continuou", "mean")
    )


def run_symbol(symbol: str, store: HistoryStore, threshold_frac: float) -> dict:
    candles = store.load(symbol, "5m")
    daily_range = (
        candles["high"].groupby(candles.index.normalize()).max()
        - candles["low"].groupby(candles.index.normalize()).min()
    )
    typical_range = daily_range.rolling(20).mean().shift(1)

    sessions = {
        day: bars for day, bars in candles.groupby(candles.index.normalize())
        if len(bars) >= 60
    }
    controls = control_table(sessions)

    events = []
    for day, bars in sessions.items():
        base_range = typical_range.get(day)
        if base_range is None or pd.isna(base_range):
            continue
        threshold = threshold_frac * float(base_range)
        closes = bars["close"].reset_index(drop=True)
        close_end = float(closes.iloc[-1])

        for event in structure_events(swing_pivots(closes, threshold)):
            i = event.confirm_index
            if len(closes) - i - 1 < MIN_BARS_LEFT:
                continue
            entry = float(closes.iloc[i])
            move = (close_end - entry) * event.direction
            bucket = bucket_of(bars.index[i])
            if bucket not in controls.index:
                continue
            events.append({
                "bucket": bucket, "leg": event.leg, "move": move,
                "continuou": np.sign(close_end - entry) == event.direction,
                "excesso": move - float(controls.loc[bucket, "move"]),
                "manha": bars.index[i].hour < 13,
            })

    frame = pd.DataFrame(events)
    if frame.empty:
        return {}

    def stats(group: pd.DataFrame) -> dict:
        n = len(group)
        excess = group["excesso"]
        se = float(excess.std() / sqrt(n)) if n > 1 else float("nan")
        return {
            "n": n,
            "cont": round(float(group["continuou"].mean()), 3),
            "move": round(float(group["move"].mean()), 1),
            "excesso": round(float(excess.mean()), 1),
            "t": round(float(excess.mean() / se), 2) if se and se > 0 else None,
        }

    result = {
        "geral": stats(frame),
        "manha": stats(frame[frame["manha"]]),
        "tarde": stats(frame[~frame["manha"]]),
        "por_perna": {
            str(k): stats(frame[frame["leg"] == k]) for k in (1, 2, 3)
        } | {"4+": stats(frame[frame["leg"] >= 4])},
        "controle_medio": round(float(controls["move"].mean()), 1),
    }
    return result


def show(symbol: str, threshold: float, result: dict) -> None:
    if not result:
        print(f"{symbol} · limiar {threshold:.0%}: sem eventos")
        return
    g = result["geral"]
    print(f"\n── {symbol} · limiar {threshold:.0%} do range típico ──")
    print(f"{'grupo':<10} {'eventos':>8} {'continua':>9} {'pts':>7} {'excesso':>8} {'t':>6}")
    for label, key in (("geral", "geral"), ("manhã", "manha"), ("tarde", "tarde")):
        s = result[key]
        if not s or not s.get("n"):
            continue
        print(f"{label:<10} {s['n']:>8} {s['cont']:>8.1%} {s['move']:>+7.0f} "
              f"{s['excesso']:>+8.0f} {str(s['t']):>6}")
    print("por perna da estrutura:")
    for leg, s in result["por_perna"].items():
        if not s or not s.get("n"):
            continue
        print(f"  {leg:<8} {s['n']:>8} {s['cont']:>8.1%} {s['move']:>+7.0f} "
              f"{s['excesso']:>+8.0f} {str(s['t']):>6}")


def main() -> None:
    store = HistoryStore()
    report = {}
    print("═══ Estrutura de topos e fundos · evento × controle de mesmo horário ═══")
    print("excesso = pontos do evento MENOS o controle ingênuo (direção da sessão)")

    for symbol in ("WIN$N", "WDO$N"):
        report[symbol] = {}
        for threshold in THRESHOLDS:
            result = run_symbol(symbol, store, threshold)
            report[symbol][f"{threshold:.2f}"] = result
            show(symbol, threshold, result)

    out = ROOT / "web" / "swing_structure.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
