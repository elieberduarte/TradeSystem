"""Testes da combinação de estratégias em carteira."""

import pandas as pd

from src.bot.backtest.engine import Trade
from src.bot.backtest.portfolio import (
    combine,
    correlation_matrix,
    daily_pnl,
    equal_risk_weights,
)


def trades(pnls_by_day: dict[str, float]) -> list[Trade]:
    return [
        Trade(
            symbol="WIN", side="buy", entry_time=pd.Timestamp(day),
            entry_price=0, quantity=1, stop_loss=0, take_profit=0,
            exit_time=pd.Timestamp(day), pnl=pnl,
        )
        for day, pnl in pnls_by_day.items()
    ]


def test_daily_pnl_agrupa_por_dia_de_saida():
    t = trades({"2024-01-02": 100.0, "2024-01-03": -50.0})
    t += trades({"2024-01-02": 20.0})  # segundo trade no mesmo dia
    series = daily_pnl(t)
    assert series.loc[pd.Timestamp("2024-01-02")] == 120.0
    assert series.loc[pd.Timestamp("2024-01-03")] == -50.0


def test_correlacao_detecta_estrategias_opostas():
    dias = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    a = trades(dict(zip(dias, [100.0, -100.0, 100.0, -100.0])))
    b = trades(dict(zip(dias, [-100.0, 100.0, -100.0, 100.0])))
    matrix = correlation_matrix({"a": a, "b": b})
    assert matrix.loc["a", "b"] == -1.0


def test_correlacao_detecta_estrategias_redundantes():
    dias = ["2024-01-02", "2024-01-03", "2024-01-04"]
    a = trades(dict(zip(dias, [100.0, -50.0, 30.0])))
    b = trades(dict(zip(dias, [200.0, -100.0, 60.0])))
    matrix = correlation_matrix({"a": a, "b": b})
    assert matrix.loc["a", "b"] == 1.0


def test_carteira_de_opostas_tem_drawdown_menor():
    dias = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    # Duas estratégias lucrativas que oscilam em momentos opostos
    a = trades(dict(zip(dias, [300.0, -200.0, 300.0, -200.0])))
    b = trades(dict(zip(dias, [-200.0, 300.0, -200.0, 300.0])))

    isolada = combine({"a": a})
    carteira = combine({"a": a, "b": b})

    assert carteira.total_pnl == isolada.total_pnl * 2
    # A soma das duas nunca cai: o drawdown desaparece
    assert carteira.max_drawdown < isolada.max_drawdown
    assert carteira.calmar > isolada.calmar


def test_pesos_por_risco_reduzem_a_estrategia_mais_erratica():
    dias = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    calma = trades(dict(zip(dias, [50.0, 40.0, 60.0, 50.0])))
    erratica = trades(dict(zip(dias, [500.0, -400.0, 600.0, -300.0])))

    weights = equal_risk_weights({"calma": calma, "erratica": erratica})
    assert weights["calma"] > weights["erratica"]


def test_combine_sem_trades():
    assert combine({}).total_pnl == 0.0
    assert correlation_matrix({"a": []}).empty
