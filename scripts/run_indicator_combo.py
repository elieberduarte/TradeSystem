"""Donchian + Tenkan×Kijun + Squeeze na carteira de futuros.

A bateria de indicadores encontrou dois sobreviventes com replicação
alta e correlação moderada com a base (TK 0,41; squeeze 0,53 no
universo de 28). Aqui a pergunta operacional: nos 13 FUTUROS, com
margens auditadas, a combinação melhora o Calmar da carteira que o
bot já opera (Donchian solo)?

Lição do Inside Day presente: correlação baixa só agrega se a
expectativa da fonte for POSITIVA no universo operado — e os dois
candidatos usam stop 2×ATR (largo, como o Donchian), então não devem
morrer de fricção como o Inside Day morreu.

Uso: python scripts/run_indicator_combo.py
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.bot.backtest.engine import BacktestEngine, Trade
from src.bot.backtest.portfolio import combine, correlation_matrix
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.strategies.ichimoku import IchimokuCrossStrategy
from src.bot.strategies.squeeze import SqueezeBreakoutStrategy
from src.bot.universe import FUT_MARGIN, FUT_POINT_VALUE, FUTUROS

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 100_000.0
SLOTS = 10

STRATEGIES = {
    "donchian": lambda: DonchianStrategy(
        {"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}
    ),
    "tk_cross": IchimokuCrossStrategy,
    "squeeze": SqueezeBreakoutStrategy,
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


def describe(label: str, result) -> dict:
    print(f"{label:<30} {result.total_pnl:>10,.0f} "
          f"{result.total_pnl / CAPITAL:>8.1%} "
          f"{result.max_drawdown / CAPITAL:>8.1%} {result.calmar:>7.2f} "
          f"{result.daily_std:>8,.0f}")
    return {"label": label, "pnl": round(result.total_pnl, 0),
            "return_pct": round(result.total_pnl / CAPITAL * 100, 1),
            "drawdown_pct": round(result.max_drawdown / CAPITAL * 100, 1),
            "calmar": round(result.calmar, 2)}


def main() -> None:
    store = HistoryStore()
    results = {name: run_strategy(name, store) for name in STRATEGIES}

    for name, trades in results.items():
        wins = sum(1 for t in trades if t.pnl > 0)
        total = sum(t.pnl for t in trades)
        print(f"{name}: {len(trades)} trades · acerto {wins / len(trades):.1%} "
              f"· PnL {total:,.0f}")

    print("\nCorrelação do PnL diário:")
    print(correlation_matrix(results).to_string())

    print(f"\n{'carteira':<30} {'PnL':>10} {'retorno':>8} {'drawdown':>8} "
          f"{'Calmar':>7} {'±dia':>8}")
    print("-" * 78)

    rows = []
    for name in STRATEGIES:
        rows.append(describe(f"{name} isolada", combine({name: results[name]})))

    pairs = [
        ("donchian + tk_cross", {"donchian": 0.5, "tk_cross": 0.5}),
        ("donchian + squeeze", {"donchian": 0.5, "squeeze": 0.5}),
        ("as três (1/3 cada)", {"donchian": 1 / 3, "tk_cross": 1 / 3, "squeeze": 1 / 3}),
    ]
    for label, weights in pairs:
        subset = {k: results[k] for k in weights}
        rows.append(describe(label, combine(subset, weights)))

    out = ROOT / "web" / "indicator_combo.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
