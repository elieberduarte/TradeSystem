"""Backtest com capital de verdade: risco E caixa respeitados.

Agora que o dimensionamento honra as duas restrições, os valores em
reais voltam a significar reais. Roda a carteira em vários níveis de
capital para mostrar como o retorno percentual se comporta.

Uso: python scripts/run_realistic.py [capital] [vagas]
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.backtest.engine import BacktestEngine
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.universe import EXPANDED, POINT_VALUE, block_of, unit_cost_of

ROOT = Path(__file__).resolve().parents[1]
BASE = {"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}


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


def run(store: HistoryStore, capital: float, slots: int) -> dict:
    def risk() -> RiskManager:
        return RiskManager(
            RiskConfig(
                capital=capital, max_risk_per_trade_pct=1.0,
                max_daily_loss_pct=100.0, max_weekly_loss_pct=6.0,
                max_open_positions=1, mode="swing_trade",
                trading_start=time(0, 0), trading_end=time(23, 59),
                max_consecutive_losses=0, risk_slots=slots,
                enforce_cash=True,
            )
        )

    rows, all_trades = [], []
    for symbol in EXPANDED:
        candles = store.load(symbol, "1d")
        if candles is None or len(candles) < 700:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            DonchianStrategy(BASE), risk(), point_value=POINT_VALUE.get(symbol, 1.0),
            warmup=210, slippage_points=slippage, cost_per_contract=cost,
            unit_cost=unit_cost_of(symbol),
        )
        result = engine.run(symbol, candles)
        if not result.trades:
            continue
        rows.append(
            {
                "symbol": symbol, "bloco": block_of(symbol),
                "pnl": result.total_pnl, "trades": len(result.trades),
                "positivo": result.total_pnl > 0,
            }
        )
        for trade in result.trades:
            all_trades.append(
                {
                    "saida": pd.Timestamp(trade.exit_time).normalize(),
                    "pnl": trade.pnl,
                }
            )

    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    trades_frame = pd.DataFrame(all_trades).sort_values("saida")
    equity = trades_frame["pnl"].cumsum()
    peak, max_dd = float("-inf"), 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)

    total = float(frame["pnl"].sum())
    years = (trades_frame["saida"].max() - trades_frame["saida"].min()).days / 365.25
    return {
        "capital": capital, "slots": slots,
        "instruments": len(frame),
        "positive": int(frame["positivo"].sum()),
        "trades": int(frame["trades"].sum()),
        "total_pnl": round(total, 0),
        "return_pct": round(total / capital * 100, 1),
        "annual_pct": round(((1 + total / capital) ** (1 / years) - 1) * 100, 1) if years > 0 else 0,
        "max_drawdown": round(max_dd, 0),
        "drawdown_pct": round(max_dd / capital * 100, 1),
        "years": round(years, 1),
    }


def main() -> None:
    store = HistoryStore()
    if len(sys.argv) > 1:
        configs = [(float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 10)]
    else:
        configs = [
            (20_000.0, 5), (20_000.0, 10),
            (50_000.0, 10), (100_000.0, 10),
            (200_000.0, 15), (500_000.0, 20),
        ]

    print("═══ Backtest com capital real (risco E caixa) ═══")
    print("donchian somente-compra · canal 20 · 28 instrumentos · diário\n")
    print(f"{'capital':>10} {'vagas':>6} {'ativos':>7} {'positivos':>10} {'trades':>7} "
          f"{'PnL':>11} {'retorno':>9} {'a.a.':>7} {'drawdown':>10}")
    print("-" * 88)

    report = []
    for capital, slots in configs:
        row = run(store, capital, slots)
        if not row:
            continue
        report.append(row)
        print(
            f"{row['capital']:>10,.0f} {row['slots']:>6} {row['instruments']:>7} "
            f"{row['positive']:>4}/{row['instruments']:<5} {row['trades']:>7} "
            f"{row['total_pnl']:>11,.0f} {row['return_pct']:>8.1f}% "
            f"{row['annual_pct']:>6.1f}% {row['drawdown_pct']:>9.1f}%"
        )

    print("-" * 88)
    if report:
        print(f"\nPeríodo: {report[0]['years']} anos")
        print("Retorno e drawdown em % do capital — agora comparáveis ao CDI (~14% a.a.).")

    out = ROOT / "web" / "realistic.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
