"""Backtest de scalping com execução por ordem limitada.

Uso: python scripts/run_scalping.py [SIMBOLO]

Diferença crucial para os outros scripts: mede também a TAXA DE
EXECUÇÃO das ordens. Um scalp que só funciona quando 100% das
limitadas são preenchidas não existe na prática.
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.backtest.limit_engine import LimitOrderEngine
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.scalp import MicroFadeStrategy, RangeScalpStrategy

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 20_000.0
POINT_VALUE = 0.20
# Emolumentos + liquidação (round trip). O spread NÃO entra: a entrada
# e o alvo são limitados; só o stop paga, via stop_slippage.
COST_PER_CONTRACT = 1.5
STOP_SLIPPAGE = 5.0  # 1 tick do WIN


def risk_factory() -> RiskManager:
    return RiskManager(
        RiskConfig(
            capital=CAPITAL, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=3.0, max_open_positions=1,
            max_trades_per_day=0, max_consecutive_losses=0,
            trading_start=time(9, 10), trading_end=time(17, 30),
            flat_time=time(17, 45),
        )
    )


# Alvo e stop SIMÉTRICOS: é a única configuração que a análise de
# viabilidade mostrou ter chance (alvo 20 com stop 20 precisa de 72,2%
# de acerto com ordem limitada, e o mercado entrega 73,4%). Alvo menor
# que o stop exigiria taxas de acerto que ninguém alcança.
CONFIGS = [
    ("micro_fade  20/20 · offset 10", MicroFadeStrategy,
     {"lookback": 5, "min_move": 60.0, "target": 20.0, "stop": 20.0}, 10.0),
    ("micro_fade  20/20 · offset 20", MicroFadeStrategy,
     {"lookback": 5, "min_move": 60.0, "target": 20.0, "stop": 20.0}, 20.0),
    ("micro_fade  30/30 · offset 15", MicroFadeStrategy,
     {"lookback": 5, "min_move": 80.0, "target": 30.0, "stop": 30.0}, 15.0),
    ("range_scalp 20/20 · offset 10", RangeScalpStrategy,
     {"window": 30, "max_range": 250.0, "target": 20.0, "stop": 20.0}, 10.0),
    ("range_scalp 20/20 · offset 20", RangeScalpStrategy,
     {"window": 30, "max_range": 250.0, "target": 20.0, "stop": 20.0}, 20.0),
    ("range_scalp 30/30 · offset 15", RangeScalpStrategy,
     {"window": 30, "max_range": 250.0, "target": 30.0, "stop": 30.0}, 15.0),
]


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "WIN$N"
    candles = HistoryStore().load(symbol, "1m")
    if candles is None:
        raise SystemExit(f"Sem acervo de 1 minuto para {symbol}")

    print(f"═══ SCALPING · {symbol} 1min · capital R$ {CAPITAL:,.0f} ═══")
    print(f"{len(candles):,} candles ({candles.index.min()} → {candles.index.max()})")
    print(f"Execução: entrada e alvo LIMITADOS; stop a mercado ({STOP_SLIPPAGE} pts)")
    print(f"Custo: R$ {COST_PER_CONTRACT}/contrato (emolumentos, sem spread)\n")

    print(f"{'configuração':<34} {'execuções':>10} {'trades':>7} {'acerto':>7} {'PnL':>11} {'R$/trade':>9}")
    print("-" * 84)

    report = []
    for label, factory, params, offset in CONFIGS:
        engine = LimitOrderEngine(
            factory(params), risk_factory(), point_value=POINT_VALUE,
            warmup=60, lookback=120, limit_offset=offset, limit_timeout_bars=3,
            stop_slippage=STOP_SLIPPAGE, cost_per_contract=COST_PER_CONTRACT,
            max_holding_bars=15,
        )
        result = engine.run(symbol, candles)
        trades = result.trades
        if not trades:
            print(f"{label:<34} {'sem trades':>10}")
            continue
        wins = sum(1 for t in trades if t.pnl > 0)
        print(
            f"{label:<34} {engine.fill_rate:>9.1%} {len(trades):>7} "
            f"{wins / len(trades):>6.1%} {result.total_pnl:>11,.0f} "
            f"{result.expectancy:>9.2f}"
        )
        report.append(
            {
                "config": label, "offset": offset,
                "orders_placed": engine.orders_placed,
                "fill_rate": round(engine.fill_rate, 4),
                "trades": len(trades),
                "win_rate": round(wins / len(trades), 4),
                "pnl": round(result.total_pnl, 2),
                "expectancy": round(result.expectancy, 2),
                "max_drawdown": round(result.max_drawdown, 2),
                "longest_losing_streak": result.longest_losing_streak,
            }
        )

    print("-" * 84)
    if report:
        best = max(report, key=lambda r: r["pnl"])
        print(f"Melhor: {best['config']} → R$ {best['pnl']:,.0f} "
              f"({best['trades']} trades, R$ {best['expectancy']:.2f}/trade, "
              f"execução {best['fill_rate']:.0%})")
        print("\nLembrete: expectativa por trade abaixo de ~R$ 1 é margem de navalha —")
        print("qualquer degradação de execução na conta real apaga o resultado.")

    out = ROOT / "web" / "scalping.json"
    out.write_text(
        json.dumps({"symbol": symbol, "results": report}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
