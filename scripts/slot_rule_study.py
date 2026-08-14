"""Qual regra de seleção usar quando há mais sinais que vagas?

Compara alfabética, aleatória (20 sementes), bloco e margem, em
grades de 4 a 10 vagas com o capital de R$ 100k. Também mede o
"sem teto" (todo sinal entra — o número que o backtest por
instrumento reporta) para saber quanto o teto custa.

A escolha certa de vagas é um dilema: menos vagas = mais caixa por
vaga (contratos caros entram, mais contratos por posição), porém
mais disputa e sinais descartados. Mais vagas = mais diversificação,
porém vagas de R$ 10k não compram T10 (margem R$ 31,6k) nem ICF.

Uso: python scripts/slot_rule_study.py
"""

import json
import statistics
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.backtest.engine import BacktestEngine
from src.bot.backtest.slots import SlotTrade, simulate_slots
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.universe import FUT_MARGIN, FUT_POINT_VALUE, FUTUROS, fut_block_of

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 100_000.0


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


def collect_trades(store: HistoryStore, slots: int) -> list[SlotTrade]:
    """Roda o Donchian nos 13 futuros com o sizing daquela contagem de vagas."""
    def risk() -> RiskManager:
        return RiskManager(
            RiskConfig(
                capital=CAPITAL, max_risk_per_trade_pct=1.0,
                max_daily_loss_pct=100.0, max_weekly_loss_pct=6.0,
                max_open_positions=1, mode="swing_trade",
                trading_start=time(0, 0), trading_end=time(23, 59),
                max_consecutive_losses=0, risk_slots=1, cash_slots=slots,
                enforce_cash=True,
            )
        )

    trades: list[SlotTrade] = []
    for symbol in FUTUROS:
        candles = store.load(symbol, "1d")
        if candles is None or len(candles) < 700:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            DonchianStrategy({"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}),
            risk(), point_value=FUT_POINT_VALUE.get(symbol, 1.0), warmup=210,
            slippage_points=slippage, cost_per_contract=cost,
            unit_cost=FUT_MARGIN.get(symbol),
        )
        for trade in engine.run(symbol, candles).trades:
            trades.append(SlotTrade(
                symbol=symbol, block=fut_block_of(symbol),
                entry=trade.entry_time.normalize(), exit=trade.exit_time.normalize(),
                pnl=trade.pnl, margin=FUT_MARGIN.get(symbol, 0.0),
            ))
    return trades


def main() -> None:
    store = HistoryStore()

    print("═══ Regra de seleção de sinais · donchian só compra · R$ 100k ═══\n")
    print(f"{'vagas':>5} {'regra':<12} {'PnL':>10} {'retorno':>8} {'DD':>7} {'Calmar':>7} "
          f"{'pulados':>8} {'disputas':>9}")
    print("-" * 74)

    report = []
    for slots in (4, 6, 8, 10):
        trades = collect_trades(store, slots)

        rows = []
        for rule in ("alfabetica", "bloco", "margem"):
            result = simulate_slots(trades, slots, rule)
            rows.append((rule, result.total_pnl, result.max_drawdown,
                         result.calmar, len(result.skipped), result.contention_days))

        randoms = [simulate_slots(trades, slots, "aleatoria", seed=s) for s in range(20)]
        pnls = [r.total_pnl for r in randoms]
        calmars = [r.calmar for r in randoms if r.calmar != float("inf")]
        rows.append(("aleatória·med", statistics.median(pnls),
                     statistics.median(r.max_drawdown for r in randoms),
                     statistics.median(calmars) if calmars else float("inf"),
                     round(statistics.mean(len(r.skipped) for r in randoms)),
                     randoms[0].contention_days))

        no_cap = simulate_slots(trades, slots=len(FUTUROS) + 1, rule="alfabetica")
        rows.append(("sem teto", no_cap.total_pnl, no_cap.max_drawdown,
                     no_cap.calmar, 0, 0))

        for rule, pnl, dd, calmar, skipped, contention in rows:
            calmar_txt = f"{calmar:.2f}" if calmar != float("inf") else "∞"
            print(f"{slots:>5} {rule:<12} {pnl:>10,.0f} {pnl / CAPITAL:>7.1%} "
                  f"{dd / CAPITAL:>6.1%} {calmar_txt:>7} {skipped:>8} {contention:>9}")
            report.append({"vagas": slots, "regra": rule, "pnl": round(pnl, 0),
                           "dd": round(dd, 0),
                           "calmar": round(calmar, 2) if calmar != float("inf") else None,
                           "pulados": skipped, "disputas": contention})
        aleatoria_range = f"{min(pnls):,.0f} a {max(pnls):,.0f}"
        print(f"{'':>5} {'(aleatória: faixa de PnL em 20 sementes: ' + aleatoria_range + ')'}")
        print()

    out = ROOT / "web" / "slot_rules.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Exportado para {out}")


if __name__ == "__main__":
    main()
