"""Carteira de futuros: a versão eficiente em capital.

Futuros exigem só margem, não o valor cheio. Como o trend following
precisa ficar posicionado em muitos mercados ao mesmo tempo, isso muda
a economia por completo — e explica por que fundos de managed futures
operam futuros e não ações à vista.

Uso: python scripts/run_futures_portfolio.py [capital]
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
from src.bot.execution.mt5_broker import MT5Broker
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.universe import FUT_MARGIN, FUT_POINT_VALUE, FUTUROS, fut_block_of

ROOT = Path(__file__).resolve().parents[1]
BASE = {"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": False}


def friction(symbol: str) -> tuple[float, float]:
    """Slippage em unidades do preço cotado + custo por contrato."""
    if symbol in ("WIN$N", "IND$N"):
        return 10.0, 1.5
    if symbol in ("WDO$N", "DOL$N"):
        return 0.5, 2.0
    if symbol.startswith("DI1"):
        return 0.005, 1.0
    if symbol in ("WSP$N", "T10$N"):
        return 1.0, 2.0
    return 0.5, 2.0  # commodities


def ensure_data(store: HistoryStore) -> None:
    missing = [s for s in FUTUROS if store.load(s, "1d") is None]
    if not missing:
        return
    print(f"Baixando {len(missing)} instrumentos ausentes: {missing}")
    broker = MT5Broker()
    broker.connect()
    for symbol in missing:
        try:
            store.update_from_broker(broker, symbol, "1d", limit=99_999)
        except Exception as exc:  # noqa: BLE001
            print(f"  {symbol}: falhou — {exc}")
    broker.disconnect()


def run(store: HistoryStore, capital: float, slots: int, long_only: bool) -> dict:
    def risk() -> RiskManager:
        return RiskManager(
            RiskConfig(
                capital=capital, max_risk_per_trade_pct=1.0,
                max_daily_loss_pct=100.0, max_weekly_loss_pct=6.0,
                max_open_positions=1, mode="swing_trade",
                trading_start=time(0, 0), trading_end=time(23, 59),
                max_consecutive_losses=0,
                # Risco cheio por trade; o caixa é que se divide entre
                # as vagas. Dividir o risco também deixaria cada posição
                # pequena demais para caber num contrato.
                risk_slots=1, cash_slots=slots,
                enforce_cash=True,
            )
        )

    rows, trades = [], []
    params = {**BASE, "long_only": long_only}
    for symbol in FUTUROS:
        candles = store.load(symbol, "1d")
        if candles is None or len(candles) < 700:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            DonchianStrategy(params), risk(),
            point_value=FUT_POINT_VALUE.get(symbol, 1.0), warmup=210,
            slippage_points=slippage, cost_per_contract=cost,
            unit_cost=FUT_MARGIN.get(symbol),
        )
        result = engine.run(symbol, candles)
        if not result.trades:
            continue
        rows.append(
            {
                "symbol": symbol, "bloco": fut_block_of(symbol),
                "pnl": result.total_pnl, "trades": len(result.trades),
                "win_rate": sum(1 for t in result.trades if t.pnl > 0) / len(result.trades),
                "positivo": result.total_pnl > 0,
            }
        )
        for trade in result.trades:
            trades.append(
                {"saida": pd.Timestamp(trade.exit_time).normalize(), "pnl": trade.pnl}
            )

    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    tf = pd.DataFrame(trades).sort_values("saida")
    equity = tf["pnl"].cumsum()
    peak, max_dd = float("-inf"), 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)

    total = float(frame["pnl"].sum())
    years = (tf["saida"].max() - tf["saida"].min()).days / 365.25
    return {
        "capital": capital, "slots": slots, "long_only": long_only,
        "instruments": len(frame), "positive": int(frame["positivo"].sum()),
        "trades": int(frame["trades"].sum()),
        "total_pnl": round(total, 0),
        "return_pct": round(total / capital * 100, 1),
        "annual_pct": round(((1 + total / capital) ** (1 / years) - 1) * 100, 1) if years > 0 and total > -capital else None,
        "drawdown_pct": round(max_dd / capital * 100, 1),
        "calmar": round(total / max_dd, 2) if max_dd > 0 else None,
        "years": round(years, 1),
        "per_symbol": frame.to_dict("records"),
    }


def main() -> None:
    store = HistoryStore()
    ensure_data(store)

    print("═══ Carteira de FUTUROS · donchian canal 20 · diário ═══")
    print(f"{len(FUTUROS)} contratos em {len(set(fut_block_of(s) for s in FUTUROS))} blocos")
    print("⚠️ Valores de ponto e margem são estimativas — confirmar antes de operar\n")
    print(f"{'capital':>10} {'vagas':>6} {'lados':>10} {'ativos':>7} {'positivos':>10} "
          f"{'trades':>7} {'retorno':>9} {'a.a.':>7} {'drawdown':>9} {'Calmar':>7}")
    print("-" * 92)

    report = []
    for long_only in (True, False):
        for capital, slots in ((20_000.0, 6), (50_000.0, 8), (100_000.0, 10), (200_000.0, 13)):
            row = run(store, capital, slots, long_only)
            if not row:
                continue
            report.append(row)
            annual = f"{row['annual_pct']:.1f}%" if row["annual_pct"] is not None else "—"
            print(
                f"{row['capital']:>10,.0f} {row['slots']:>6} "
                f"{'só compra' if long_only else 'compra+venda':>10} "
                f"{row['instruments']:>7} {row['positive']:>4}/{row['instruments']:<5} "
                f"{row['trades']:>7} {row['return_pct']:>8.1f}% {annual:>7} "
                f"{row['drawdown_pct']:>8.1f}% {str(row['calmar']):>7}"
            )

    print("-" * 92)
    if report:
        best = max(report, key=lambda r: (r["calmar"] or -99))
        print(f"\nMelhor por Calmar: capital R$ {best['capital']:,.0f}, "
              f"{best['slots']} vagas, {'só compra' if best['long_only'] else 'compra+venda'}")
        print(f"  retorno {best['return_pct']:.1f}% em {best['years']} anos "
              f"· drawdown {best['drawdown_pct']:.1f}% · Calmar {best['calmar']}")
        print("\nPor bloco (melhor configuração):")
        frame = pd.DataFrame(best["per_symbol"])
        by_block = frame.groupby("bloco").agg(
            ativos=("symbol", "size"), positivos=("positivo", "sum"),
            pnl=("pnl", "sum"), acerto=("win_rate", "mean"),
        ).round(2).sort_values("pnl", ascending=False)
        print(by_block.to_string())

    out = ROOT / "web" / "futures_portfolio.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
