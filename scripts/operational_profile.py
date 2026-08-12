"""Como seria operar isso na prática, em números.

Responde as perguntas que decidem o desenho do bot ao vivo:
quantos sinais por ano, quantas posições simultâneas, quanto capital
seria preciso, e com que frequência dois ou mais ativos disparam no
mesmo dia (que é quando o bot precisa ESCOLHER).
"""

import sys
from collections import defaultdict
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.backtest.engine import BacktestEngine
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.universe import EXPANDED, POINT_VALUE, block_of

CAPITAL = 150_000.0
BASE = {"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}


def risk() -> RiskManager:
    return RiskManager(
        RiskConfig(
            capital=CAPITAL, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=100.0, max_weekly_loss_pct=6.0,
            max_open_positions=1, mode="swing_trade",
            trading_start=time(0, 0), trading_end=time(23, 59),
            max_consecutive_losses=0,
        )
    )


def friction(symbol: str) -> tuple[float, float]:
    if symbol == "WIN$N":
        return 10.0, 1.0
    if symbol == "WDO$N":
        return 0.5, 2.0
    if symbol.startswith("DI1"):
        return 0.005, 0.01
    if symbol.endswith("$N"):
        return 0.5, 1.0
    return 0.01, 0.01


def main() -> None:
    store = HistoryStore()
    trades = []
    for symbol in EXPANDED:
        candles = store.load(symbol, "1d")
        if candles is None or len(candles) < 700:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            DonchianStrategy(BASE), risk(), point_value=POINT_VALUE.get(symbol, 1.0),
            warmup=210, slippage_points=slippage, cost_per_contract=cost,
        )
        for trade in engine.run(symbol, candles).trades:
            trades.append(
                {
                    "symbol": symbol,
                    "bloco": block_of(symbol),
                    "entrada": pd.Timestamp(trade.entry_time).normalize(),
                    "saida": pd.Timestamp(trade.exit_time).normalize(),
                    "dias": (trade.exit_time - trade.entry_time).days,
                }
            )

    frame = pd.DataFrame(trades)
    if frame.empty:
        raise SystemExit("Sem trades")

    span_years = (frame["saida"].max() - frame["entrada"].min()).days / 365.25
    print("═══ Perfil operacional · donchian somente-compra · 28 instrumentos ═══")
    print(f"Período: {frame['entrada'].min().date()} → {frame['saida'].max().date()} "
          f"({span_years:.1f} anos)\n")

    print("── Ritmo ──")
    print(f"Operações no período: {len(frame)}")
    print(f"Sinais por ano: {len(frame) / span_years:.0f}  (~1 a cada {252 * span_years / len(frame):.1f} pregões)")
    print(f"Dias na posição: mediana {frame['dias'].median():.0f} · "
          f"média {frame['dias'].mean():.0f} · máximo {frame['dias'].max():.0f}")

    # ── Posições simultâneas, dia a dia ──
    days = pd.date_range(frame["entrada"].min(), frame["saida"].max(), freq="B")
    concurrent = pd.Series(0, index=days)
    for _, row in frame.iterrows():
        concurrent.loc[row["entrada"] : row["saida"]] += 1

    print("\n── Posições abertas ao mesmo tempo ──")
    print(f"Média: {concurrent.mean():.1f} · mediana {concurrent.median():.0f} · "
          f"máximo {concurrent.max()}")
    for pct in (50, 75, 90, 95, 99):
        print(f"  percentil {pct}: {np.percentile(concurrent, pct):.0f} posições")
    print(f"Dias sem posição alguma: {(concurrent == 0).sum()} de {len(concurrent)} "
          f"({(concurrent == 0).mean():.0%})")

    # ── Dias com mais de um sinal: quando o bot precisa escolher ──
    by_day = frame.groupby("entrada").size()
    print("\n── Quando o bot precisa ESCOLHER (sinais no mesmo dia) ──")
    print(f"Dias com sinal: {len(by_day)}")
    for n in (1, 2, 3, 4):
        count = int((by_day == n).sum()) if n < 4 else int((by_day >= n).sum())
        label = f"{n} sinal" if n == 1 else (f"{n} sinais" if n < 4 else f"{n} ou mais")
        print(f"  {label:<12} {count:>4} dias ({count / len(by_day):>5.1%})")
    print(f"Maior número de sinais num único dia: {by_day.max()}")

    # ── Capital: quanto seria preciso para caber tudo ──
    print("\n── Implicação de capital ──")
    typical = np.percentile(concurrent, 90)
    print(f"No percentil 90 há {typical:.0f} posições abertas simultaneamente.")
    print(f"Com risco de 1% por posição, isso expõe {typical:.0f}% do capital de uma vez —")
    print(f"por isso o risco precisa ser dividido: 1% ÷ {typical:.0f} ≈ "
          f"{1 / typical:.2f}% por operação, ou aceitar exposição maior.")

    print("\n── Ritmo por bloco ──")
    per_block = frame.groupby("bloco").agg(
        operacoes=("symbol", "size"),
        dias_medianos=("dias", "median"),
    )
    per_block["por_ano"] = (per_block["operacoes"] / span_years).round(1)
    print(per_block.to_string())


if __name__ == "__main__":
    main()
