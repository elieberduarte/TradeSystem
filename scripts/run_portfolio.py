"""Testa a ideia da carteira: várias estratégias operando juntas.

Roda o walk-forward de cada estratégia, mede a correlação entre elas e
compara a carteira combinada contra a melhor estratégia isolada.

Uso: python scripts/run_portfolio.py [SIMBOLO]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.bot.backtest.portfolio import (
    combine,
    correlation_matrix,
    equal_risk_weights,
)
from src.bot.backtest.walkforward import WalkForward
from src.bot.data.history import HistoryStore
from src.bot.strategies.band_fade import BandFadeStrategy
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.strategies.ibs import IbsStrategy

from run_swing_backtest import (  # noqa: E402
    CAPITAL,
    COST_PER_CONTRACT,
    POINT_VALUE,
    SLIPPAGE_POINTS,
    swing_risk,
)

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = [
    ("donchian", lambda p: DonchianStrategy(p),
     {"channel": [20, 40], "stop_atr": [2.0], "rr": [2.0, 3.0]}),
    ("band_fade", lambda p: BandFadeStrategy(p),
     {"period": [20], "mult": [2.0, 2.5], "band": ["keltner"], "target": ["mid", 1.5]}),
    ("ibs", lambda p: IbsStrategy(p),
     {"entry_low": [0.1, 0.2], "target_atr": [0.5, 1.0], "stop_atr": [2.0]}),
]


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "WIN$N"
    candles = HistoryStore().load(symbol, "1d")
    if candles is None:
        raise SystemExit(f"Sem acervo diário para {symbol}")

    point_value = POINT_VALUE.get(symbol, 1.0)
    print(f"═══ CARTEIRA · {symbol} diário · capital R$ {CAPITAL:,.0f} ═══")
    print(f"{len(candles)} pregões ({candles.index.min().date()} → {candles.index.max().date()})\n")

    results = {}
    for name, factory, grid in CANDIDATES:
        wf = WalkForward(
            strategy_factory=factory, risk_factory=swing_risk, point_value=point_value,
            warmup=210, slippage_points=SLIPPAGE_POINTS, cost_per_contract=COST_PER_CONTRACT,
        )
        report = wf.run(symbol, candles, grid, train_bars=500, test_bars=125)
        results[name] = report.oos_trades
        metrics = report.risk_metrics()
        print(f"── {name} isolada ──")
        print(f"  PnL {report.oos_pnl:,.0f} | Drawdown {metrics.get('max_drawdown', 0):,.0f} | "
              f"Calmar {metrics.get('calmar')} | {len(report.oos_trades)} trades")

    print("\n── As estratégias ganham nos mesmos dias? ──")
    correlations = correlation_matrix(results)
    if correlations.empty:
        print("  Dados insuficientes")
        return
    print(correlations.to_string())
    print("  (perto de 0 ou negativo = se complementam; perto de 1 = redundantes)")

    print("\n── Carteira combinada ──")
    equal = combine(results)
    print(f"  Pesos iguais:      {equal.summary()}")

    weights = equal_risk_weights(results)
    balanced = combine(results, weights)
    print(f"  Risco equilibrado: {balanced.summary()}")
    print(f"  Pesos usados: {weights}")

    best_name = max(results, key=lambda n: sum(t.pnl for t in results[n]))
    best = combine({best_name: results[best_name]})
    print(f"\n  Melhor isolada ({best_name}): {best.summary()}")

    print("\n── Vale a pena combinar? ──")
    for label, portfolio in (("pesos iguais", equal), ("risco equilibrado", balanced)):
        delta_calmar = portfolio.calmar - best.calmar
        verdict = "SIM" if delta_calmar > 0 else "não"
        print(f"  {label}: Calmar {portfolio.calmar:.2f} vs {best.calmar:.2f} isolada "
              f"→ {verdict} ({delta_calmar:+.2f})")

    out = ROOT / "web" / "portfolio.json"
    out.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "correlations": correlations.to_dict(),
                "weights": weights,
                "portfolio": {
                    "total_pnl": round(balanced.total_pnl, 2),
                    "max_drawdown": round(balanced.max_drawdown, 2),
                    "calmar": round(balanced.calmar, 2),
                    "days": balanced.days,
                },
                "equity": [
                    {"date": str(d.date()), "value": round(float(v), 2)}
                    for d, v in balanced.equity.items()
                ],
            },
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
