"""Bateria de indicadores: Ichimoku e squeeze de Bollinger.

Sete variantes pré-declaradas, universo de 28 instrumentos, diário,
critério de replicação — a mesma régua que aprovou o Donchian e
reprovou os setups dos livros. Saídas idênticas à base (stop 2×ATR,
alvo 3R) para isolar o efeito do GATILHO/FILTRO.

Para qualquer variante que replique (≥70%), mede-se a correlação do
PnL diário com o Donchian: sinal correlacionado não agrega carteira,
por melhor que pareça sozinho.

Uso: python scripts/test_indicators.py
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.backtest.engine import BacktestEngine
from src.bot.backtest.portfolio import daily_pnl
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.strategies.ichimoku import (
    CloudCrossStrategy,
    CloudFilterOverlay,
    IchimokuCrossStrategy,
)
from src.bot.strategies.squeeze import SqueezeBreakoutStrategy, SqueezeFilterOverlay
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


def donchian() -> DonchianStrategy:
    return DonchianStrategy({"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True})


def evaluate(label: str, factory, store: HistoryStore) -> tuple[dict, list]:
    rows, trades = [], []
    for symbol in EXPANDED:
        candles = store.load(symbol, "1d")
        if candles is None or len(candles) < 700:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            factory(), risk(), point_value=POINT_VALUE.get(symbol, 1.0),
            warmup=210, slippage_points=slippage, cost_per_contract=cost,
            unit_cost=unit_cost_of(symbol),
        )
        result = engine.run(symbol, candles)
        if len(result.trades) < 5:
            continue
        rows.append({"symbol": symbol, "pnl": result.total_pnl,
                     "positivo": result.total_pnl > 0})
        trades.extend(result.trades)

    if not rows:
        return {"label": label, "tested": 0}, trades
    frame = pd.DataFrame(rows)
    series = pd.DataFrame(
        [{"saida": pd.Timestamp(t.exit_time).normalize(), "pnl": t.pnl} for t in trades]
    ).sort_values("saida")
    equity = series["pnl"].cumsum()
    peak, max_dd = float("-inf"), 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)

    total = float(frame["pnl"].sum())
    wins = sum(1 for t in trades if t.pnl > 0)
    return {
        "label": label,
        "tested": len(frame),
        "positive": int(frame["positivo"].sum()),
        "rate": round(float(frame["positivo"].mean()), 3),
        "trades": len(trades),
        "win_rate": round(wins / len(trades), 3),
        "median_pnl": round(float(frame["pnl"].median()), 0),
        "return_pct": round(total / CAPITAL * 100, 1),
        "calmar": round(total / max_dd, 2) if max_dd > 0 else None,
    }, trades


def main() -> None:
    store = HistoryStore()

    variants = [
        ("Donchian (base aprovada)", donchian),
        ("Donchian + filtro de nuvem", lambda: CloudFilterOverlay(donchian())),
        ("Tenkan × Kijun (só compra)", IchimokuCrossStrategy),
        ("Cruzamento da nuvem (só compra)", CloudCrossStrategy),
        ("Squeeze → rompimento (só compra)", SqueezeBreakoutStrategy),
        ("Squeeze → rompimento (ambos)",
         lambda: SqueezeBreakoutStrategy({"long_only": False})),
        ("Donchian + filtro de squeeze", lambda: SqueezeFilterOverlay(donchian())),
    ]

    print("═══ Indicadores · Ichimoku e Bollinger · 28 instrumentos · diário ═══")
    print("Critério: replicação ≥ 70%; depois, correlação com a base\n")
    print(f"{'variante':<34} {'replica':>9} {'trades':>7} {'acerto':>7} "
          f"{'mediana':>9} {'retorno':>9} {'Calmar':>7}")
    print("-" * 88)

    report, trade_sets = [], {}
    for label, factory in variants:
        row, trades = evaluate(label, factory, store)
        report.append(row)
        trade_sets[label] = trades
        if not row.get("tested"):
            print(f"{label:<34} sem trades suficientes")
            continue
        print(f"{label:<34} {row['positive']:>3}/{row['tested']:<5} {row['trades']:>7} "
              f"{row['win_rate']:>6.1%} {row['median_pnl']:>9,.0f} "
              f"{row['return_pct']:>8.1f}% {str(row['calmar']):>7}")

    # Correlação com a base para quem replicou
    base_label = variants[0][0]
    base_series = daily_pnl(trade_sets[base_label])
    print("\n── Correlação do PnL diário com o Donchian (sobreviventes) ──")
    for row in report[1:]:
        if not row.get("tested") or row["rate"] < 0.70:
            continue
        series = daily_pnl(trade_sets[row["label"]])
        joined = pd.DataFrame({"base": base_series, "alt": series}).fillna(0.0)
        corr = float(joined["base"].corr(joined["alt"]))
        row["corr_donchian"] = round(corr, 3)
        print(f"  {row['label']:<34} correlação {corr:+.3f}")

    out = ROOT / "web" / "indicators.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
