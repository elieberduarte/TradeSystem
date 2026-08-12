"""Carteira de trend following em vários instrumentos + robustez.

Duas validações finais do único candidato que replicou:

1. CARTEIRA — a mesma estratégia rodando em TODOS os instrumentos ao
   mesmo tempo. Sem escolher os vencedores: na hora de operar não se
   sabe quais vão funcionar, então incluir só ABEV3 e IVVB11 seria
   repetir exatamente o viés que o teste de replicação expôs.

2. ROBUSTEZ — o resultado sobrevive a mudar os parâmetros? Edge real
   funciona numa vizinhança; sorte funciona num ponto só.
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.backtest.portfolio import combine, correlation_matrix, equal_risk_weights
from src.bot.backtest.walkforward import WalkForward
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 150_000.0
POINT_VALUE = {"WIN$N": 0.20, "IND$N": 1.00, "WDO$N": 10.00, "DOL$N": 50.00}
UNIVERSE = [
    "WIN$N", "WDO$N", "BOVA11", "SMAL11", "IVVB11",
    "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "B3SA3", "WEGE3", "PRIO3",
]
GRID = {"channel": [20, 40], "stop_atr": [2.0], "rr": [2.0, 3.0]}


def swing_risk() -> RiskManager:
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
    if symbol in POINT_VALUE:
        return (10.0, 1.0) if symbol in ("WIN$N", "IND$N") else (0.5, 2.0)
    return (0.01, 0.01)


def run_walkforward(symbol: str, candles: pd.DataFrame, grid: dict):
    slippage, cost = friction(symbol)
    wf = WalkForward(
        strategy_factory=lambda p: DonchianStrategy(p),
        risk_factory=swing_risk,
        point_value=POINT_VALUE.get(symbol, 1.0),
        warmup=210, slippage_points=slippage, cost_per_contract=cost,
    )
    return wf.run(symbol, candles, grid, train_bars=500, test_bars=125)


def portfolio_section(store: HistoryStore) -> dict:
    print("═══ CARTEIRA DE TREND FOLLOWING ═══")
    print("Mesma estratégia em todos os instrumentos, sem escolher vencedores\n")

    results = {}
    for symbol in UNIVERSE:
        candles = store.load(symbol, "1d")
        if candles is None:
            continue
        report = run_walkforward(symbol, candles, GRID)
        if report.oos_trades:
            results[symbol] = report.oos_trades
            print(f"  {symbol:<9} {report.oos_pnl:>10,.0f}")

    if len(results) < 2:
        print("Instrumentos insuficientes")
        return {}

    correlations = correlation_matrix(results)
    # Só os pares (fora da diagonal, que é sempre 1)
    mask = ~np.eye(len(correlations), dtype=bool)
    values = correlations.where(mask).stack()
    print(f"\nCorrelação média entre instrumentos: {values.mean():.3f}")
    print(f"  (faixa: {values.min():.2f} a {values.max():.2f})")

    weights = equal_risk_weights(results)
    balanced = combine(results, weights)
    equal = combine(results)
    print("\nCarteira completa:")
    print(f"  Pesos iguais:      {equal.summary()}")
    print(f"  Risco equilibrado: {balanced.summary()}")

    best = max(results, key=lambda s: sum(t.pnl for t in results[s]))
    single = combine({best: results[best]})
    print(f"\n  Melhor isolado ({best}): {single.summary()}")
    print(f"\n  Carteira vs melhor isolado: Calmar {balanced.calmar:.2f} vs {single.calmar:.2f}")

    return {
        "instruments": len(results),
        "mean_correlation": round(float(values.mean()), 3),
        "weights": weights,
        "equal": {
            "pnl": round(equal.total_pnl, 2), "drawdown": round(equal.max_drawdown, 2),
            "calmar": round(equal.calmar, 2), "daily_std": round(equal.daily_std, 2),
        },
        "balanced": {
            "pnl": round(balanced.total_pnl, 2), "drawdown": round(balanced.max_drawdown, 2),
            "calmar": round(balanced.calmar, 2), "daily_std": round(balanced.daily_std, 2),
        },
        "equity": [
            {"date": str(d.date()), "value": round(float(v), 2)}
            for d, v in balanced.equity.items()
        ],
    }


def robustness_section(store: HistoryStore) -> dict:
    """Fixa cada parâmetro e mede o universo inteiro — sem otimização."""
    print("\n\n═══ ROBUSTEZ DE PARÂMETROS ═══")
    print("Cada canal testado sozinho em todos os instrumentos, sem otimizar\n")
    print(f"{'canal':>6} {'positivos':>11} {'PnL mediano':>13} {'PnL total':>12}")
    print("-" * 45)

    rows = []
    for channel in (10, 15, 20, 30, 40, 60):
        grid = {"channel": [channel], "stop_atr": [2.0], "rr": [3.0]}
        pnls = []
        for symbol in UNIVERSE:
            candles = store.load(symbol, "1d")
            if candles is None:
                continue
            report = run_walkforward(symbol, candles, grid)
            if report.oos_trades:
                pnls.append(report.oos_pnl)
        if not pnls:
            continue
        positive = sum(1 for p in pnls if p > 0)
        median = sorted(pnls)[len(pnls) // 2]
        rows.append(
            {
                "channel": channel, "tested": len(pnls), "positive": positive,
                "median_pnl": round(median, 2), "total_pnl": round(sum(pnls), 2),
            }
        )
        print(
            f"{channel:>6} {positive:>4}/{len(pnls):<6} {median:>13,.0f} {sum(pnls):>12,.0f}"
        )

    stable = sum(1 for r in rows if r["total_pnl"] > 0)
    print("-" * 45)
    print(f"Canais com resultado positivo: {stable}/{len(rows)}")
    print("  (edge real sobrevive à vizinhança; sorte funciona num ponto só)")
    return {"channels": rows, "positive_channels": stable, "tested_channels": len(rows)}


def main() -> None:
    store = HistoryStore()
    payload = {"portfolio": portfolio_section(store), "robustness": robustness_section(store)}
    out = ROOT / "web" / "donchian_validation.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
