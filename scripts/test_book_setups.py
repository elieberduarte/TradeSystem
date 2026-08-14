"""Testa todos os setups extraídos dos livros, no universo completo.

Critério: replicação entre instrumentos, o mesmo que aprovou o
Donchian e reprovou o band fade. Um setup que lucra em três ativos e
perde em vinte não passa, por melhor que seja o número agregado.

Uso: python scripts/test_book_setups.py
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
from src.bot.strategies.book_setups import (
    GapFadeStrategy,
    InsideDayStrategy,
    OopsStrategy,
)
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.universe import EXPANDED, POINT_VALUE, unit_cost_of

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 100_000.0
SLOTS = 10


def friction(symbol: str) -> tuple[float, float]:
    if symbol in ("WIN$N", "IND$N"):
        return 10.0, 1.5
    if symbol in ("WDO$N", "DOL$N"):
        return 0.5, 2.0
    if symbol.startswith("DI1"):
        return 0.005, 1.0
    if symbol.endswith("$N"):
        return 0.5, 1.0
    return 0.01, 0.01


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


def evaluate(label: str, factory, store: HistoryStore, max_bars: int = 0) -> dict:
    rows, trades = [], []
    for symbol in EXPANDED:
        candles = store.load(symbol, "1d")
        if candles is None or len(candles) < 700:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            factory(), risk(), point_value=POINT_VALUE.get(symbol, 1.0),
            warmup=210, slippage_points=slippage, cost_per_contract=cost,
            unit_cost=unit_cost_of(symbol), max_holding_bars=max_bars,
        )
        result = engine.run(symbol, candles)
        if len(result.trades) < 5:
            continue
        pnls = [t.pnl for t in result.trades]
        rows.append(
            {
                "symbol": symbol, "pnl": result.total_pnl,
                "trades": len(result.trades),
                "positivo": result.total_pnl > 0,
                "sharpe": float(np.mean(pnls) / np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0,
            }
        )
        for trade in result.trades:
            trades.append({"saida": pd.Timestamp(trade.exit_time).normalize(), "pnl": trade.pnl})

    if not rows:
        return {"label": label, "tested": 0}
    frame = pd.DataFrame(rows)
    tf = pd.DataFrame(trades).sort_values("saida")
    equity = tf["pnl"].cumsum()
    peak, max_dd = float("-inf"), 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)

    total = float(frame["pnl"].sum())
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return {
        "label": label,
        "tested": len(frame),
        "positive": int(frame["positivo"].sum()),
        "rate": round(frame["positivo"].mean(), 3),
        "trades": len(trades),
        "win_rate": round(wins / len(trades), 3),
        "total_pnl": round(total, 0),
        "median_pnl": round(float(frame["pnl"].median()), 0),
        "return_pct": round(total / CAPITAL * 100, 1),
        "drawdown_pct": round(max_dd / CAPITAL * 100, 1),
        "calmar": round(total / max_dd, 2) if max_dd > 0 else None,
        "sharpe_medio": round(float(frame["sharpe"].mean()), 3),
    }


def main() -> None:
    store = HistoryStore()

    variants = [
        ("Donchian (referência aprovada)",
         lambda: DonchianStrategy({"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}), 0),

        # ── Oops Trade (Williams / Eykyn) ──
        ("Oops · só compra", lambda: OopsStrategy({"side": "long"}), 1),
        ("Oops · só venda", lambda: OopsStrategy({"side": "short"}), 1),
        ("Oops · gap mínimo 0,3 ATR", lambda: OopsStrategy({"side": "long", "min_gap_atr": 0.3}), 1),

        # ── Inside Day (Eykyn) ──
        ("Inside Day · ambos os lados", lambda: InsideDayStrategy({"rr": 2.0}), 0),
        ("Inside Day · só compra", lambda: InsideDayStrategy({"side": "long", "rr": 2.0}), 0),
        ("Inside Day + range estreito", lambda: InsideDayStrategy({"rr": 2.0, "narrow_pct": 0.3}), 0),

        # ── Gap fade (metade confirmada da tese de Eykyn) ──
        ("Gap fade · 0,10–0,50 ATR", lambda: GapFadeStrategy({"min_gap_atr": 0.10, "max_gap_atr": 0.50}), 1),
        ("Gap fade · 0,25–1,00 ATR", lambda: GapFadeStrategy({"min_gap_atr": 0.25, "max_gap_atr": 1.00}), 1),
        ("Gap fade · stop 2× o gap", lambda: GapFadeStrategy({"stop_mult": 2.0}), 1),
    ]

    print("═══ Setups dos livros · universo de 28 instrumentos · diário ═══")
    print("Critério: REPLICAÇÃO entre instrumentos, não lucro agregado\n")
    print(f"{'setup':<32} {'ativos':>7} {'replica':>9} {'trades':>7} {'acerto':>7} "
          f"{'mediana':>9} {'retorno':>9} {'Calmar':>7}")
    print("-" * 92)

    report = []
    for label, factory, max_bars in variants:
        row = evaluate(label, factory, store, max_bars)
        report.append(row)
        if not row.get("tested"):
            print(f"{label:<32} {'sem trades suficientes':>7}")
            continue
        print(
            f"{label:<32} {row['tested']:>7} {row['positive']:>3}/{row['tested']:<5} "
            f"{row['trades']:>7} {row['win_rate']:>6.1%} {row['median_pnl']:>9,.0f} "
            f"{row['return_pct']:>8.1f}% {str(row['calmar']):>7}"
        )

    print("-" * 92)
    print("\n── Veredito por replicação ──")
    ranked = sorted(
        (r for r in report if r.get("tested")),
        key=lambda r: (r["rate"], r["median_pnl"]), reverse=True,
    )
    for row in ranked:
        verdict = "replica" if row["rate"] >= 0.70 else (
            "inconclusivo" if row["rate"] >= 0.50 else "NÃO replica"
        )
        print(f"  {row['rate']:>5.0%}  {row['label']:<32} {verdict}")

    out = ROOT / "web" / "book_setups.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
