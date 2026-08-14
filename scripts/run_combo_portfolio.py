"""Donchian + Inside Day na carteira de futuros: a soma melhora?

O Inside Day replica em 79% dos instrumentos mas rende pouco (Calmar
0,07 isolado no universo amplo). A tese aqui é a mesma que levou a
carteira de 3 estratégias de Calmar 3,37 para 4,42: uma fonte de
sinal REAL e DESCORRELACIONADA agrega mais que uma boa e redundante.

Donchian = continuação de tendência (canal de 20 dias, só compra).
Inside Day = compressão seguida de expansão (os dois lados).
Estruturas diferentes → correlação esperada baixa. Se a correlação
for baixa E o Calmar combinado superar o do Donchian isolado, o
Inside Day ganha vaga na carteira; senão, morre também.

Uso: python scripts/run_combo_portfolio.py
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.bot.backtest.engine import BacktestEngine, Trade
from src.bot.backtest.portfolio import combine, correlation_matrix, equal_risk_weights
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.book_setups import InsideDayStrategy
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.universe import FUT_MARGIN, FUT_POINT_VALUE, FUTUROS

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 100_000.0
SLOTS = 10

STRATEGIES = {
    "donchian": lambda: DonchianStrategy(
        {"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}
    ),
    "inside_day": lambda: InsideDayStrategy({"rr": 2.0}),
}


def friction(symbol: str) -> tuple[float, float]:
    if symbol in ("WIN$N", "IND$N"):
        return 10.0, 1.5
    if symbol in ("WDO$N", "DOL$N"):
        return 0.5, 2.0
    if symbol.startswith("DI1"):
        return 0.005, 1.0
    if symbol in ("WSP$N", "T10$N"):
        return 1.0, 2.0
    return 0.5, 2.0


def risk() -> RiskManager:
    return RiskManager(
        RiskConfig(
            capital=CAPITAL, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=100.0, max_weekly_loss_pct=6.0,
            max_open_positions=1, mode="swing_trade",
            trading_start=time(0, 0), trading_end=time(23, 59),
            max_consecutive_losses=0, risk_slots=1, cash_slots=SLOTS,
            enforce_cash=True,
        )
    )


def run_strategy(name: str, store: HistoryStore) -> list[Trade]:
    trades: list[Trade] = []
    for symbol in FUTUROS:
        candles = store.load(symbol, "1d")
        if candles is None or len(candles) < 700:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            STRATEGIES[name](), risk(),
            point_value=FUT_POINT_VALUE.get(symbol, 1.0), warmup=210,
            slippage_points=slippage, cost_per_contract=cost,
            unit_cost=FUT_MARGIN.get(symbol),
        )
        trades.extend(engine.run(symbol, candles).trades)
    return trades


def describe(label: str, result, capital: float) -> dict:
    years = (result.equity.index.max() - result.equity.index.min()).days / 365.25
    annual = ((1 + result.total_pnl / capital) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    print(f"{label:<28} {result.total_pnl:>10,.0f} {result.total_pnl / capital:>8.1%} "
          f"{annual:>6.1f}% {result.max_drawdown / capital:>8.1%} "
          f"{result.calmar:>7.2f} {result.daily_std:>8,.0f}")
    return {
        "label": label, "pnl": round(result.total_pnl, 0),
        "return_pct": round(result.total_pnl / capital * 100, 1),
        "annual_pct": round(annual, 1),
        "drawdown_pct": round(result.max_drawdown / capital * 100, 1),
        "calmar": round(result.calmar, 2),
        "daily_std": round(result.daily_std, 0),
    }


def main() -> None:
    store = HistoryStore()
    results = {name: run_strategy(name, store) for name in STRATEGIES}

    for name, trades in results.items():
        wins = sum(1 for t in trades if t.pnl > 0)
        print(f"{name}: {len(trades)} trades, acerto {wins / len(trades):.1%}")

    corr = correlation_matrix(results)
    print(f"\nCorrelação diária donchian × inside_day: "
          f"{corr.loc['donchian', 'inside_day']:.3f}\n")

    print(f"{'carteira':<28} {'PnL':>10} {'retorno':>8} {'a.a.':>7} "
          f"{'drawdown':>8} {'Calmar':>7} {'±dia':>8}")
    print("-" * 82)

    report = {"correlacao": float(corr.loc["donchian", "inside_day"])}
    rows = []

    # Isoladas: cada uma usando o capital inteiro
    for name in STRATEGIES:
        rows.append(describe(f"{name} isolada", combine({name: results[name]}), CAPITAL))

    # Combinada meio a meio: mesmo capital, metade do risco para cada.
    # É a comparação honesta com as isoladas — ninguém dobra a conta.
    rows.append(describe(
        "combinada 50/50", combine(results, {"donchian": 0.5, "inside_day": 0.5}), CAPITAL
    ))

    # Combinada com risco equalizado pela volatilidade de cada uma
    weights = equal_risk_weights(results)
    half = {name: w / 2 for name, w in weights.items()}
    rows.append(describe(
        f"risco equalizado ({weights['donchian']:.2f}/{weights['inside_day']:.2f})",
        combine(results, half), CAPITAL,
    ))

    report["carteiras"] = rows
    out = ROOT / "web" / "combo_portfolio.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
