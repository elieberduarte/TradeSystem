"""Saída por falha de Fibonacci (Eykyn E-5) sobre a base Donchian.

A única regra de saída inequívoca dos livros em inglês: fechamento
que devolve mais de 61,8% do swing a favor encerra a posição. Aqui
ela substitui o alvo fixo do Donchian (entradas idênticas, aprovadas;
só a saída muda), com 0,50 e 0,786 como vizinhança de robustez —
três valores fixados antes de rodar, sem otimização.

Controles: o Donchian original (alvo 3R) e o Donchian sem alvo
nenhum (só stop), para separar "a regra Fib ajuda" de "qualquer
coisa que deixe correr ajuda".

Uso: python scripts/test_fib_exit.py
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
from src.bot.strategies.overlays import NoTargetOverlay
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
    return DonchianStrategy(
        {"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}
    )


def evaluate(label: str, factory, store: HistoryStore, fib: float = 0.0) -> dict:
    rows, trades = [], []
    for symbol in EXPANDED:
        candles = store.load(symbol, "1d")
        if candles is None or len(candles) < 700:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            factory(), risk(), point_value=POINT_VALUE.get(symbol, 1.0),
            warmup=210, slippage_points=slippage, cost_per_contract=cost,
            unit_cost=unit_cost_of(symbol), fib_exit=fib,
        )
        result = engine.run(symbol, candles)
        if len(result.trades) < 5:
            continue
        rows.append({"symbol": symbol, "pnl": result.total_pnl,
                     "trades": len(result.trades),
                     "positivo": result.total_pnl > 0})
        for trade in result.trades:
            trades.append({"saida": pd.Timestamp(trade.exit_time).normalize(),
                           "pnl": trade.pnl, "reason": trade.exit_reason,
                           "bars": trade.bars_held})

    frame = pd.DataFrame(rows)
    tf = pd.DataFrame(trades).sort_values("saida")
    equity = tf["pnl"].cumsum()
    peak, max_dd = float("-inf"), 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)

    total = float(frame["pnl"].sum())
    return {
        "label": label,
        "tested": len(frame),
        "positive": int(frame["positivo"].sum()),
        "rate": round(frame["positivo"].mean(), 3),
        "trades": len(trades),
        "win_rate": round(float((tf["pnl"] > 0).mean()), 3),
        "median_pnl": round(float(frame["pnl"].median()), 0),
        "return_pct": round(total / CAPITAL * 100, 1),
        "calmar": round(total / max_dd, 2) if max_dd > 0 else None,
        "barras_medias": round(float(tf["bars"].mean()), 1),
        "saidas_fib_pct": round(float((tf["reason"] == "fib").mean()), 3),
    }


def main() -> None:
    store = HistoryStore()

    variants = [
        ("Donchian · alvo 3R (base)", donchian, 0.0),
        ("Donchian · sem alvo, só stop", lambda: NoTargetOverlay(donchian()), 0.0),
        ("Donchian · saída Fib 0,618", lambda: NoTargetOverlay(donchian()), 0.618),
        ("Donchian · saída Fib 0,500", lambda: NoTargetOverlay(donchian()), 0.500),
        ("Donchian · saída Fib 0,786", lambda: NoTargetOverlay(donchian()), 0.786),
    ]

    print("═══ Saída por falha de Fibonacci (Eykyn) · entradas Donchian · 28 instrumentos ═══\n")
    print(f"{'variante':<30} {'replica':>9} {'trades':>7} {'acerto':>7} {'mediana':>9} "
          f"{'retorno':>9} {'Calmar':>7} {'barras':>7} {'%fib':>6}")
    print("-" * 100)

    report = []
    for label, factory, fib in variants:
        row = evaluate(label, factory, store, fib)
        report.append(row)
        print(f"{label:<30} {row['positive']:>3}/{row['tested']:<5} {row['trades']:>7} "
              f"{row['win_rate']:>6.1%} {row['median_pnl']:>9,.0f} {row['return_pct']:>8.1f}% "
              f"{str(row['calmar']):>7} {row['barras_medias']:>7} {row['saidas_fib_pct']:>6.1%}")

    out = ROOT / "web" / "fib_exit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
