"""Teste de replicação: a estratégia funciona em quantos instrumentos?

O teste mais duro que temos. Uma estratégia com vantagem estrutural
deve funcionar — com números diferentes, mas sinal igual — em vários
ativos. Funcionar em 1 de 15 é sorte; em 12 de 15 é edge.

Uso: python scripts/replication_study.py [estrategia]
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.backtest.walkforward import WalkForward
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.band_fade import BandFadeStrategy
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.strategies.ibs import IbsStrategy
from src.bot.strategies.swing_reversion import SwingReversionStrategy

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 150_000.0

# Valor financeiro de 1 ponto/real de variação, por contrato ou ação
POINT_VALUE = {
    "WIN$N": 0.20, "IND$N": 1.00,
    "WDO$N": 10.00, "DOL$N": 50.00,
}
UNIVERSE = [
    "WIN$N", "IND$N", "WDO$N", "DOL$N",
    "BOVA11", "SMAL11", "IVVB11",
    "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "B3SA3", "WEGE3", "PRIO3",
]

STRATEGIES = {
    "band_fade": (
        lambda p: BandFadeStrategy(p),
        {"period": [20], "mult": [2.0, 2.5], "band": ["keltner"], "target": ["mid", 1.5]},
    ),
    "ibs": (
        lambda p: IbsStrategy(p),
        {"entry_low": [0.1, 0.2], "target_atr": [0.5, 1.0], "stop_atr": [2.0]},
    ),
    "donchian": (
        lambda p: DonchianStrategy(p),
        {"channel": [20, 40], "stop_atr": [2.0], "rr": [2.0, 3.0]},
    ),
    "swing_reversion": (
        lambda p: SwingReversionStrategy(p),
        {"lookback": [40, 60], "threshold_std": [1.0, 1.5], "rr": [1.5, 2.0]},
    ),
}


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


def slippage_for(symbol: str) -> tuple[float, float]:
    """Fricção por instrumento: futuros em pontos, ações em centavos."""
    if symbol in POINT_VALUE:
        return (10.0, 1.0) if "WIN" in symbol or "IND" in symbol else (0.5, 2.0)
    # Ações e ETFs: 1 centavo de spread + corretagem simbólica por ação
    return (0.01, 0.01)


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    store = HistoryStore()
    strategies = {k: v for k, v in STRATEGIES.items() if not only or k == only}

    report: dict[str, dict] = {}
    for name, (factory, grid) in strategies.items():
        print(f"\n═══ {name} — replicação em {len(UNIVERSE)} instrumentos ═══")
        print(f"{'ativo':<9} {'PnL':>11} {'trades':>7} {'acerto':>7} {'Calmar':>7} {'janelas+':>9}")
        print("-" * 55)

        rows = []
        for symbol in UNIVERSE:
            candles = store.load(symbol, "1d")
            if candles is None or len(candles) < 700:
                continue
            slippage, cost = slippage_for(symbol)
            wf = WalkForward(
                strategy_factory=factory, risk_factory=swing_risk,
                point_value=POINT_VALUE.get(symbol, 1.0), warmup=210,
                slippage_points=slippage, cost_per_contract=cost,
            )
            result = wf.run(symbol, candles, grid, train_bars=500, test_bars=125)
            trades = result.oos_trades
            if not trades:
                print(f"{symbol:<9} {'sem trades':>11}")
                continue
            metrics = result.risk_metrics()
            wins = sum(1 for t in trades if t.pnl > 0)
            rows.append(
                {
                    "symbol": symbol,
                    "pnl": round(result.oos_pnl, 2),
                    "trades": len(trades),
                    "win_rate": round(wins / len(trades), 3),
                    "calmar": metrics.get("calmar"),
                    "windows_positive": metrics.get("windows_positive"),
                    "windows": len(result.windows),
                }
            )
            print(
                f"{symbol:<9} {result.oos_pnl:>11,.0f} {len(trades):>7} "
                f"{wins / len(trades):>6.1%} {str(metrics.get('calmar')):>7} "
                f"{metrics.get('windows_positive')}/{len(result.windows):>7}"
            )

        positive = [r for r in rows if r["pnl"] > 0]
        print("-" * 55)
        print(
            f"POSITIVA EM {len(positive)}/{len(rows)} instrumentos "
            f"({len(positive) / len(rows):.0%})" if rows else "sem resultados"
        )
        if rows:
            median = sorted(r["pnl"] for r in rows)[len(rows) // 2]
            print(f"PnL mediano: {median:,.0f} | PnL total: {sum(r['pnl'] for r in rows):,.0f}")
        report[name] = {"rows": rows, "positive": len(positive), "tested": len(rows)}

    # Mescla: rodar uma estratégia isolada não apaga as anteriores
    out = ROOT / "web" / "replication.json"
    merged = {}
    if out.exists():
        try:
            merged = json.loads(out.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    merged.update(report)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
