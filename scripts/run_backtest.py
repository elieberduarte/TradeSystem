"""Primeiro backtest real: walk-forward das estratégias no WIN 5min.

Lê o acervo local (data/), roda a otimização walk-forward e imprime o
relatório. Resultado honesto = janelas out-of-sample.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.backtest.walkforward import WalkForward
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.ema_cross import EmaCrossStrategy
from src.bot.strategies.opening_range import OpeningRangeStrategy

WIN_POINT_VALUE = 0.20
CAPITAL = 20_000.0


def risk_factory() -> RiskManager:
    return RiskManager(
        RiskConfig(
            capital=CAPITAL,
            max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=3.0,
            max_open_positions=1,
            max_trades_per_day=5,
            max_consecutive_losses=3,
        )
    )


def main() -> None:
    store = HistoryStore()
    candles = store.load("WINQ26", "5m")
    if candles is None:
        raise SystemExit("Sem acervo: rode scripts/download_history.py primeiro")
    print(f"WINQ26 5m: {len(candles)} candles ({candles.index.min()} → {candles.index.max()})\n")

    runs = [
        (
            "opening_range",
            lambda p: OpeningRangeStrategy(p),
            {"range_bars": [3, 6], "rr": [1.5, 2.0]},
        ),
        (
            "ema_cross",
            lambda p: EmaCrossStrategy(p),
            {"fast": [9], "slow": [21], "trend": [0, 80], "rr": [1.5, 2.5]},
        ),
    ]

    for name, factory, grid in runs:
        print(f"═══ {name} · WIN 5min · capital R$ {CAPITAL:,.0f} ═══")
        wf = WalkForward(
            strategy_factory=factory,
            risk_factory=risk_factory,
            point_value=WIN_POINT_VALUE,
            warmup=110,
        )
        report = wf.run("WIN", candles, grid, train_bars=2500, test_bars=1250)
        print(report.summary())
        trades = report.oos_trades
        if trades:
            wins = sum(1 for t in trades if t.pnl > 0)
            print(
                f"  Win rate OOS: {wins / len(trades):.1%} | "
                f"melhor: {max(t.pnl for t in trades):+.2f} | "
                f"pior: {min(t.pnl for t in trades):+.2f}"
            )
        print()


if __name__ == "__main__":
    main()
