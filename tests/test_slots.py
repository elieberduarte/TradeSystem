"""Testes da simulação de vagas e das regras de seleção."""

import pandas as pd

from src.bot.backtest.slots import SlotTrade, simulate_slots


def trade(symbol, block, entry, exit_, pnl, margin=1000.0):
    return SlotTrade(
        symbol=symbol, block=block,
        entry=pd.Timestamp(entry), exit=pd.Timestamp(exit_),
        pnl=pnl, margin=margin,
    )


def test_sem_disputa_todos_entram():
    trades = [
        trade("A", "x", "2024-01-02", "2024-01-05", 100),
        trade("B", "y", "2024-01-03", "2024-01-06", 200),
    ]
    result = simulate_slots(trades, slots=2, rule="alfabetica")
    assert len(result.taken) == 2
    assert result.skipped == []
    assert result.contention_days == 0
    assert result.total_pnl == 300


def test_teto_de_vagas_e_respeitado():
    # Três sinais no mesmo dia, duas vagas
    trades = [
        trade("A", "x", "2024-01-02", "2024-01-10", 100),
        trade("B", "y", "2024-01-02", "2024-01-10", 200),
        trade("C", "z", "2024-01-02", "2024-01-10", 300),
    ]
    result = simulate_slots(trades, slots=2, rule="alfabetica")
    assert len(result.taken) == 2
    assert [t.symbol for t in result.taken] == ["A", "B"]
    assert [t.symbol for t in result.skipped] == ["C"]
    assert result.contention_days == 1


def test_vaga_liberada_no_dia_da_saida():
    # A sai dia 5; B entra dia 5 e deve caber na vaga única
    trades = [
        trade("A", "x", "2024-01-02", "2024-01-05", 100),
        trade("B", "y", "2024-01-05", "2024-01-08", 200),
    ]
    result = simulate_slots(trades, slots=1, rule="alfabetica")
    assert len(result.taken) == 2


def test_regra_de_bloco_prioriza_o_menos_representado():
    # Carteira já tem posição no bloco "x"; sinal de "x" e de "y"
    # disputam uma vaga — "y" diversifica, deve vencer mesmo sendo
    # alfabeticamente posterior
    trades = [
        trade("A1", "x", "2024-01-02", "2024-01-10", 0),
        trade("A2", "x", "2024-01-03", "2024-01-10", 0),
        trade("B1", "y", "2024-01-03", "2024-01-10", 0),
    ]
    result = simulate_slots(trades, slots=2, rule="bloco")
    taken = {t.symbol for t in result.taken}
    assert taken == {"A1", "B1"}


def test_regra_de_margem_prioriza_a_mais_barata():
    trades = [
        trade("CARO", "x", "2024-01-02", "2024-01-10", 0, margin=30_000),
        trade("BARATO", "y", "2024-01-02", "2024-01-10", 0, margin=1_000),
    ]
    result = simulate_slots(trades, slots=1, rule="margem")
    assert [t.symbol for t in result.taken] == ["BARATO"]


def test_aleatoria_e_deterministica_por_semente():
    trades = [
        trade(s, "x", "2024-01-02", "2024-01-10", 0) for s in "ABCDEF"
    ]
    first = simulate_slots(trades, slots=3, rule="aleatoria", seed=7)
    second = simulate_slots(trades, slots=3, rule="aleatoria", seed=7)
    assert [t.symbol for t in first.taken] == [t.symbol for t in second.taken]


def test_drawdown_e_calmar():
    trades = [
        trade("A", "x", "2024-01-02", "2024-01-03", 100),
        trade("B", "x", "2024-01-04", "2024-01-05", -50),
        trade("C", "x", "2024-01-08", "2024-01-09", 150),
    ]
    result = simulate_slots(trades, slots=1, rule="alfabetica")
    assert result.total_pnl == 200
    assert result.max_drawdown == 50
    assert result.calmar == 4.0
