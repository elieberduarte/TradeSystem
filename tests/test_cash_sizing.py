"""Testes do dimensionamento com restrição de caixa."""

from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.universe import unit_cost_of


def manager(enforce: bool, slots: int = 1, capital: float = 100_000.0) -> RiskManager:
    return RiskManager(
        RiskConfig(
            capital=capital, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=50.0, max_open_positions=1,
            risk_slots=slots, enforce_cash=enforce,
        )
    )


def test_sem_restricao_de_caixa_so_o_risco_manda():
    # Ação de R$ 20 com stop a R$ 19: risco R$ 1/ação, risco total R$ 1.000
    # → 1.000 ações, o que custaria R$ 20.000 (mais que o risco permite ver)
    assert manager(enforce=False).position_size(20.0, 19.0) == 1000.0


def test_com_restricao_o_caixa_limita():
    # Mesmo trade, mas agora o caixa de R$ 100.000 comporta 5.000 ações;
    # o risco continua sendo o limite mais apertado
    assert manager(enforce=True).position_size(20.0, 19.0) == 1000.0


def test_caixa_vira_o_limite_quando_o_stop_e_curto():
    # Stop a R$ 0,05 do preço: pelo risco caberiam 20.000 ações
    # (R$ 400.000), mas o caixa só permite 5.000
    quantity = manager(enforce=True).position_size(20.0, 19.95)
    assert quantity == 100_000.0 / 20.0


def test_vagas_dividem_o_caixa():
    # Com 10 vagas, cada uma dispõe de R$ 10.000 → 500 ações de R$ 20
    quantity = manager(enforce=True, slots=10).position_size(20.0, 19.95)
    assert quantity == 500.0


def test_caixa_e_risco_podem_ter_divisores_diferentes():
    # Risco cheio (1% = R$ 1.000) mas caixa dividido em 10 vagas.
    # É o caso dos futuros: dividir o risco por 10 deixaria cada trade
    # pequeno demais para comprar um contrato.
    m = RiskManager(
        RiskConfig(
            capital=100_000.0, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=50.0, max_open_positions=1,
            risk_slots=1, cash_slots=10, enforce_cash=True,
        )
    )
    # Risco permite 1.000 unidades; o caixa da vaga (R$ 10.000) permite 500
    assert m.position_size(20.0, 19.0) == 500.0


def test_cash_slots_zero_usa_risk_slots():
    m = RiskManager(
        RiskConfig(
            capital=100_000.0, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=50.0, max_open_positions=1,
            risk_slots=4, cash_slots=0, enforce_cash=True,
        )
    )
    # Caixa por vaga: 100.000/4 = 25.000 → 1.250 ações de R$ 20
    assert m.position_size(20.0, 19.95) == 1250.0


def test_futuro_usa_margem_e_nao_preco():
    # WIN a 140.000 pontos: pagar o "preço" seria absurdo. Com margem de
    # R$ 2.000 por contrato e caixa de R$ 100.000, cabem 50 contratos.
    m = manager(enforce=True)
    quantity = m.position_size(
        entry_price=140_000.0, stop_loss=139_000.0,
        point_value=0.20, unit_cost=2_000.0,
    )
    # Pelo risco: 1.000 / (1.000 pts x 0,20) = 5 contratos — mais apertado
    assert quantity == 5.0


def test_margem_limita_quando_o_risco_permitiria_mais():
    m = manager(enforce=True, capital=10_000.0)
    quantity = m.position_size(
        entry_price=140_000.0, stop_loss=139_950.0,
        point_value=0.20, unit_cost=2_000.0,
    )
    # Caixa de R$ 10.000 / margem R$ 2.000 = 5 contratos
    assert quantity == 5.0


def test_universo_conhece_a_margem_dos_futuros():
    # Valores auditados na XP em 14/08/2026 (scripts/verify_contracts.py)
    assert unit_cost_of("WIN$N") == 7_400.0
    assert unit_cost_of("DI1F27") == 750.0
    # Ações não têm margem: paga-se o preço
    assert unit_cost_of("PETR4") is None
