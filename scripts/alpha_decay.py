"""A vantagem está decaindo?

Um edge de arbitragem some conforme mais gente o descobre. Um edge de
prêmio de risco ou de viés comportamental persiste, porque não depende
de ninguém ignorá-lo.

Este script separa os dois casos empiricamente: roda a estratégia em
todo o universo e quebra o resultado por ano. Se o desempenho cai
monotonicamente, é arbitragem sendo consumida. Se oscila sem
tendência, é prêmio de risco — que tem anos bons e ruins, mas não
"acaba".

Uso: python scripts/alpha_decay.py [estrategia]
"""

import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.bot.backtest.engine import BacktestEngine
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy

CAPITAL = 150_000.0
POINT_VALUE = {"WIN$N": 0.20, "IND$N": 1.00, "WDO$N": 10.00, "DOL$N": 50.00}
UNIVERSE = [
    "WIN$N", "WDO$N", "BOVA11", "SMAL11", "IVVB11",
    "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "B3SA3", "WEGE3", "PRIO3",
]


def risk() -> RiskManager:
    return RiskManager(
        RiskConfig(
            capital=CAPITAL, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=100.0, max_weekly_loss_pct=6.0,
            max_open_positions=1, mode="swing_trade",
            trading_start=time(0, 0), trading_end=time(23, 59),
            max_consecutive_losses=0, risk_slots=len(UNIVERSE),
        )
    )


def friction(symbol: str) -> tuple[float, float]:
    if symbol in POINT_VALUE:
        return (10.0, 1.0) if symbol in ("WIN$N", "IND$N") else (0.5, 2.0)
    return (0.01, 0.01)


def main() -> None:
    long_only = "--short" not in sys.argv
    store = HistoryStore()
    params = {"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": long_only}

    print("═══ A vantagem está decaindo? ═══")
    print(f"donchian canal 20 · {'somente compra' if long_only else 'compra e venda'} · "
          f"{len(UNIVERSE)} instrumentos · parâmetros FIXOS (sem otimização)\n")

    rows = []
    for symbol in UNIVERSE:
        candles = store.load(symbol, "1d")
        if candles is None:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            DonchianStrategy(params), risk(), point_value=POINT_VALUE.get(symbol, 1.0),
            warmup=210, slippage_points=slippage, cost_per_contract=cost,
        )
        result = engine.run(symbol, candles)
        for trade in result.trades:
            rows.append(
                {
                    "symbol": symbol,
                    "ano": trade.exit_time.year,
                    "pnl": trade.pnl,
                    "acertou": 1 if trade.pnl > 0 else 0,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("Nenhum trade gerado")

    grouped = frame.groupby("ano")
    yearly = pd.DataFrame(
        {
            "trades": grouped.size(),
            "acerto": (grouped["acertou"].mean() * 100).round(1),
            "pnl_total": grouped["pnl"].sum().round(0),
            "pnl_por_trade": grouped["pnl"].mean().round(2),
            "ativos_positivos": grouped.apply(
                lambda g: (g.groupby("symbol")["pnl"].sum() > 0).sum(), include_groups=False
            ),
            "ativos": grouped["symbol"].nunique(),
        }
    )
    print("── Resultado por ano ──")
    print(yearly.to_string())
    print()

    # Tendência: correlação entre o ano e o resultado por trade
    anos = yearly.index.to_numpy(dtype=float)
    valores = yearly["pnl_por_trade"].to_numpy(dtype=float)
    if len(anos) > 2:
        correlacao = float(pd.Series(anos).corr(pd.Series(valores)))
        print(f"Correlação ano × resultado por trade: {correlacao:+.3f}")
        if correlacao < -0.5:
            leitura = "queda consistente — sinal de arbitragem sendo consumida"
        elif correlacao > 0.5:
            leitura = "melhora consistente — provavelmente sorte de período, não decaimento"
        else:
            leitura = "sem tendência — comportamento de prêmio de risco (anos bons e ruins)"
        print(f"Leitura: {leitura}")

    print()
    metade = len(yearly) // 2
    primeira = yearly.iloc[:metade]["pnl_por_trade"].mean()
    segunda = yearly.iloc[metade:]["pnl_por_trade"].mean()
    print(f"Primeira metade do período: R$ {primeira:,.2f} por trade")
    print(f"Segunda metade do período:  R$ {segunda:,.2f} por trade")
    if primeira != 0:
        print(f"Variação: {(segunda - primeira) / abs(primeira) * 100:+.0f}%")


if __name__ == "__main__":
    main()
