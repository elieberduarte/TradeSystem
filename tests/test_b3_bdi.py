"""Testes da derivação do fluxo diário a partir do acumulado mensal da B3."""

import pandas as pd

from src.bot.data.b3_bdi import daily_flow


def frame(rows):
    return pd.DataFrame(rows, columns=["data", "categoria", "compras_mil", "vendas_mil"])


def test_saldo_diario_e_a_diferenca_dos_acumulados():
    part = frame([
        ("2026-08-03", "Estrangeiro", 100.0, 40.0),   # acumulado dia 1: saldo +60
        ("2026-08-04", "Estrangeiro", 250.0, 90.0),   # acumulado: +160 → dia = +100
        ("2026-08-05", "Estrangeiro", 300.0, 200.0),  # acumulado: +100 → dia = −60
    ])
    flow = daily_flow(part).set_index("data")["saldo_dia_mil"]
    assert flow["2026-08-03"] == 60.0
    assert flow["2026-08-04"] == 100.0
    assert flow["2026-08-05"] == -60.0


def test_virada_de_mes_reinicia_o_acumulado():
    part = frame([
        ("2026-07-30", "Estrangeiro", 900.0, 100.0),  # julho acumulado +800
        ("2026-07-31", "Estrangeiro", 950.0, 100.0),  # +850 → dia +50
        ("2026-08-03", "Estrangeiro", 30.0, 10.0),    # agosto reinicia: dia = +20
    ])
    flow = daily_flow(part).set_index("data")["saldo_dia_mil"]
    assert flow["2026-07-31"] == 50.0
    assert flow["2026-08-03"] == 20.0     # não é 20 − 850


def test_buraco_no_acervo_vira_nan_em_vez_de_salto_falso():
    part = frame([
        ("2026-08-03", "Estrangeiro", 100.0, 0.0),
        ("2026-08-12", "Estrangeiro", 900.0, 0.0),    # 9 dias depois: diferença acumula vários dias
    ])
    flow = daily_flow(part).set_index("data")["saldo_dia_mil"]
    assert pd.isna(flow["2026-08-12"])


def test_categorias_sao_independentes():
    part = frame([
        ("2026-08-03", "Estrangeiro", 100.0, 0.0),
        ("2026-08-03", "Pessoa Física", 10.0, 50.0),
        ("2026-08-04", "Estrangeiro", 150.0, 0.0),
        ("2026-08-04", "Pessoa Física", 20.0, 55.0),
    ])
    flow = daily_flow(part).set_index(["data", "categoria"])["saldo_dia_mil"]
    assert flow[("2026-08-04", "Estrangeiro")] == 50.0
    assert flow[("2026-08-04", "Pessoa Física")] == 5.0
