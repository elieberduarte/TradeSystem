"""A saída por ADR melhora o que já funciona? E é ela ou o sinal?

Desenho do experimento, com controle:

  A) Donchian + saída original (2×ATR / 3× o risco)  ← nossa base
  B) Donchian + saída ADR (ADR/3 stop, ADR/2 alvo)   ← a hipótese
  C) Entrada ALEATÓRIA + saída ADR                    ← o controle

Se B > A, a saída ADR agrega. Mas se C também for positiva, a saída
está carregando o resultado sozinha e o padrão de entrada é
decorativo — conclusão que só aparece com o controle no lugar.

Uso: python scripts/test_adr_exit.py
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
from src.bot.strategies.adr_exit import AdrExitOverlay, RandomEntryStrategy
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.universe import FUT_MARGIN, FUT_POINT_VALUE, FUTUROS, fut_block_of

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 100_000.0
SLOTS = 10
BASE = {"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}


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


def evaluate(label: str, factory, store: HistoryStore) -> dict:
    rows, trades = [], []
    for symbol in FUTUROS:
        candles = store.load(symbol, "1d")
        if candles is None or len(candles) < 700:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            factory(), risk(), point_value=FUT_POINT_VALUE.get(symbol, 1.0),
            warmup=210, slippage_points=slippage, cost_per_contract=cost,
            unit_cost=FUT_MARGIN.get(symbol),
        )
        result = engine.run(symbol, candles)
        if not result.trades:
            continue
        rows.append(
            {
                "symbol": symbol, "bloco": fut_block_of(symbol),
                "pnl": result.total_pnl, "trades": len(result.trades),
                "positivo": result.total_pnl > 0,
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
    years = (tf["saida"].max() - tf["saida"].min()).days / 365.25
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return {
        "label": label,
        "tested": len(frame),
        "positive": int(frame["positivo"].sum()),
        "trades": len(trades),
        "win_rate": round(wins / len(trades), 3),
        "total_pnl": round(total, 0),
        "return_pct": round(total / CAPITAL * 100, 1),
        "annual_pct": round(((1 + total / CAPITAL) ** (1 / years) - 1) * 100, 1)
        if years > 0 and total > -CAPITAL else None,
        "drawdown_pct": round(max_dd / CAPITAL * 100, 1),
        "calmar": round(total / max_dd, 2) if max_dd > 0 else None,
    }


def main() -> None:
    store = HistoryStore()

    variants = [
        ("A · donchian + saída ATR (base)", lambda: DonchianStrategy(BASE)),
        ("B · donchian + saída ADR", lambda: AdrExitOverlay(DonchianStrategy(BASE), "lento")),
        ("C · aleatório + saída ADR", lambda: AdrExitOverlay(RandomEntryStrategy(0.02), "lento")),
        ("D · aleatório + saída ATR", lambda: RandomEntryStrategy(0.02)),
    ]

    print("═══ A saída por ADR agrega? ═══")
    print("donchian somente-compra · 13 futuros · diário · com controle aleatório\n")
    print(f"{'variante':<34} {'ativos':>7} {'positivos':>10} {'trades':>7} "
          f"{'acerto':>7} {'retorno':>9} {'a.a.':>7} {'drawdown':>9} {'Calmar':>7}")
    print("-" * 100)

    report = []
    for label, factory in variants:
        row = evaluate(label, factory, store)
        report.append(row)
        if not row.get("tested"):
            print(f"{label:<34} {'sem trades':>7}")
            continue
        annual = f"{row['annual_pct']:.1f}%" if row["annual_pct"] is not None else "—"
        print(
            f"{label:<34} {row['tested']:>7} {row['positive']:>4}/{row['tested']:<5} "
            f"{row['trades']:>7} {row['win_rate']:>6.1%} {row['return_pct']:>8.1f}% "
            f"{annual:>7} {row['drawdown_pct']:>8.1f}% {str(row['calmar']):>7}"
        )

    print("-" * 100)
    print("\n── Leitura ──")
    base, adr_exit, control, control_atr = report[0], report[1], report[2], report[3]
    if adr_exit.get("tested") and base.get("tested"):
        delta = adr_exit["return_pct"] - base["return_pct"]
        print(f"B vs A (a saída ADR agrega ao donchian?): {delta:+.1f} pontos percentuais")
    if control.get("tested"):
        print(f"C (entrada aleatória + saída ADR): {control['return_pct']:+.1f}%")
        if control["return_pct"] > 0:
            print("  ⚠️ O controle aleatório também lucra — a saída está carregando")
            print("     o resultado, e o padrão de entrada pode ser decorativo.")
        else:
            print("  ✓ O controle aleatório perde — o sinal de entrada é que sustenta.")
    if control_atr.get("tested"):
        print(f"D (entrada aleatória + saída ATR): {control_atr['return_pct']:+.1f}%")

    out = ROOT / "web" / "adr_exit_test.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
