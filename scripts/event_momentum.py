"""Momentum pós-evento (Eykyn E-4b): a direção do choque persiste?

A tese: em dia de payroll, depois da janela de choque (15-30 min), a
direção estabelecida persiste até o fechamento — o evento resolve a
indecisão do dia.

Payroll = primeira sexta-feira do mês, 8h30 de NY = 9h30 ou 10h30 de
Brasília conforme o horário de verão americano. A janela 9h00→10h45
cobre os dois casos. Medição em WIN e WDO 15m (5 anos):

  direção da manhã  = fechamento das 10h45 − abertura das 9h00
  persistência      = fechamento do dia na MESMA direção da manhã?

Controles: as demais sextas e os demais dias. Se a persistência do
payroll não superar a dos controles, não há evento a operar.

Uso: python scripts/event_momentum.py
"""

import sys
from datetime import time as dtime
from math import erf, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.data.history import HistoryStore

CUTOFF = dtime(10, 45)


def sign_test(hits: int, n: int) -> float:
    if n == 0:
        return 1.0
    z = (hits - n / 2) / (sqrt(n) / 2)
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def first_friday(day: pd.Timestamp) -> bool:
    return day.weekday() == 4 and day.day <= 7


def day_records(intraday: pd.DataFrame) -> pd.DataFrame:
    records = []
    for day, bars in intraday.groupby(intraday.index.normalize()):
        if len(bars) < 20:
            continue
        morning = bars[bars.index.time <= CUTOFF]
        if len(morning) < 4:
            continue
        open_ = float(morning["open"].iloc[0])
        anchor = float(morning["close"].iloc[-1])   # pós-janela de choque
        close = float(bars["close"].iloc[-1])
        am, pm = anchor - open_, close - anchor
        if am == 0 or pm == 0:
            continue
        records.append({
            "day": day, "payroll": first_friday(day),
            "friday": day.weekday() == 4,
            "am_abs": abs(am),
            "persistiu": np.sign(am) == np.sign(pm),
        })
    return pd.DataFrame(records)


def report(label: str, group: pd.DataFrame) -> None:
    n = len(group)
    if n == 0:
        return
    hits = int(group["persistiu"].sum())
    print(f"{label:<22} {n:>6} {group['am_abs'].mean():>10,.0f} "
          f"{hits / n:>12.1%} {sign_test(hits, n):>8.4f}")


def main() -> None:
    store = HistoryStore()
    for symbol in ("WIN$N", "WDO$N"):
        candles = store.load(symbol, "15m")
        if candles is None:
            continue
        frame = day_records(candles)

        print(f"\n═══ {symbol} · manhã (9h00→10h45) → tarde (10h45→fechamento) ═══")
        print(f"{'grupo':<22} {'dias':>6} {'|manhã| méd':>10} {'persistência':>12} {'p':>8}")
        print("-" * 64)
        report("payroll (1ª sexta)", frame[frame["payroll"]])
        report("outras sextas", frame[frame["friday"] & ~frame["payroll"]])
        report("demais dias", frame[~frame["friday"]])

        # O choque é maior no payroll? (pré-condição da tese)
        payroll_move = frame[frame["payroll"]]["am_abs"].mean()
        normal_move = frame[~frame["payroll"]]["am_abs"].mean()
        print(f"\nChoque da manhã no payroll = {payroll_move / normal_move:.2f}x o dia comum")

    print("\nReferência: persistência de 50% = a tarde ignora a manhã.")


if __name__ == "__main__":
    main()
