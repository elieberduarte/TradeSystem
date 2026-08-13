"""O retorno de 10,9% está certo? E comparado com o quê?

Duas verificações:

1. A ARITMÉTICA — o número bate com os trades?

2. A COMPARAÇÃO — e esta é a parte que muda tudo. Em futuros a margem
   é depositada como GARANTIA, e a B3 aceita Tesouro Selic e CDB como
   garantia. O dinheiro continua rendendo enquanto está lá. Então o
   resultado da estratégia não substitui o CDI: ele se SOMA ao CDI.

   Comparar "10,9% da estratégia" com "14% do CDI" só faria sentido se
   operar exigisse deixar o dinheiro parado sem render — o que não é o
   caso.

Este script mede quanto capital fica de fato imobilizado em margem ao
longo do tempo, e refaz a conta do jeito certo.
"""

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
from src.bot.universe import FUT_MARGIN, FUT_POINT_VALUE, FUTUROS, fut_block_of

CAPITAL = 100_000.0
SLOTS = 10
CDI_ANUAL = 0.1415
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


def main() -> None:
    store = HistoryStore()
    trades = []
    for symbol in FUTUROS:
        candles = store.load(symbol, "1d")
        if candles is None or len(candles) < 700:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            DonchianStrategy(BASE), risk(),
            point_value=FUT_POINT_VALUE.get(symbol, 1.0), warmup=210,
            slippage_points=slippage, cost_per_contract=cost,
            unit_cost=FUT_MARGIN.get(symbol),
        )
        for trade in engine.run(symbol, candles).trades:
            trades.append(
                {
                    "symbol": symbol, "bloco": fut_block_of(symbol),
                    "entrada": pd.Timestamp(trade.entry_time).normalize(),
                    "saida": pd.Timestamp(trade.exit_time).normalize(),
                    "quantidade": trade.quantity,
                    "margem": FUT_MARGIN.get(symbol, 0.0) * trade.quantity,
                    "pnl": trade.pnl,
                }
            )

    frame = pd.DataFrame(trades)
    if frame.empty:
        raise SystemExit("Sem trades")

    total_pnl = float(frame["pnl"].sum())
    start, end = frame["entrada"].min(), frame["saida"].max()
    years = (end - start).days / 365.25

    print("═══ Auditoria do retorno ═══\n")
    print("── 1. A aritmética confere? ──")
    print(f"Operações: {len(frame)} em {frame['symbol'].nunique()} instrumentos")
    print(f"Período: {start.date()} → {end.date()} ({years:.2f} anos)")
    print(f"Soma dos resultados: R$ {total_pnl:,.2f}")
    print(f"Sobre capital de R$ {CAPITAL:,.0f}: {total_pnl / CAPITAL:.1%} no período")
    annual = (1 + total_pnl / CAPITAL) ** (1 / years) - 1
    print(f"Anualizado: {annual:.2%} ao ano  ← o número que estava no relatório")
    print(f"Resultado médio por operação: R$ {frame['pnl'].mean():,.2f}")

    # ── Margem realmente utilizada ──
    days = pd.date_range(start, end, freq="B")
    margin_used = pd.Series(0.0, index=days)
    for _, row in frame.iterrows():
        margin_used.loc[row["entrada"] : row["saida"]] += row["margem"]

    print("\n── 2. Quanto capital fica de fato imobilizado? ──")
    print(f"Margem média em uso: R$ {margin_used.mean():,.0f} "
          f"({margin_used.mean() / CAPITAL:.1%} do capital)")
    for pct in (50, 90, 99, 100):
        value = np.percentile(margin_used, pct)
        print(f"  percentil {pct:>3}: R$ {value:>10,.0f} ({value / CAPITAL:>5.1%})")
    idle = 1 - margin_used.mean() / CAPITAL
    print(f"Capital ocioso em média: {idle:.1%}")

    print("\n── 3. A comparação correta ──")
    print("Em futuros a margem é depositada como GARANTIA, e a B3 aceita")
    print("Tesouro Selic e CDB como garantia — o dinheiro continua rendendo.")
    print("Logo o resultado da estratégia SOMA ao CDI, não substitui.\n")

    cdi_total = (1 + CDI_ANUAL) ** years - 1
    print(f"{'cenário':<44} {'no período':>12} {'ao ano':>9}")
    print("-" * 68)
    print(f"{'Tudo em CDI, sem operar':<44} {cdi_total:>11.1%} {CDI_ANUAL:>8.1%}")
    print(f"{'Só a estratégia (capital sem render)':<44} "
          f"{total_pnl / CAPITAL:>11.1%} {annual:>8.1%}")
    combined_total = cdi_total + total_pnl / CAPITAL
    combined_annual = (1 + combined_total) ** (1 / years) - 1
    print(f"{'CDI na garantia + estratégia':<44} "
          f"{combined_total:>11.1%} {combined_annual:>8.1%}")
    print("-" * 68)
    print(f"\nGanho da estratégia sobre deixar tudo no CDI: "
          f"{combined_annual - CDI_ANUAL:+.1%} ao ano")

    print("\n── 4. Retorno sobre o capital REALMENTE empregado ──")
    avg_margin = float(margin_used.mean())
    if avg_margin > 0:
        on_margin = (1 + total_pnl / avg_margin) ** (1 / years) - 1
        print(f"Sobre a margem média de R$ {avg_margin:,.0f}: {on_margin:.1%} ao ano")
        print("Este número mostra a eficiência do sinal, mas NÃO é operável:")
        print("o capital de folga precisa existir para aguentar drawdown e")
        print("chamada de margem — não dá para operar só com a média.")


if __name__ == "__main__":
    main()
