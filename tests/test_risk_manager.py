"""Testes do gestor de risco."""

from datetime import datetime

from src.bot.risk.manager import RiskConfig, RiskManager


def make_manager(**overrides) -> RiskManager:
    config = RiskConfig(
        capital=10_000.0,
        max_risk_per_trade_pct=1.0,
        max_daily_loss_pct=3.0,
        max_open_positions=2,
    )
    return RiskManager(config=config, **overrides)


def test_permite_operar_em_condicoes_normais():
    manager = make_manager()
    allowed, _ = manager.can_open_position(now=datetime(2026, 8, 10, 10, 0))
    assert allowed


def test_bloqueia_apos_limite_de_perda_diaria():
    manager = make_manager()
    manager.register_trade_result(-300.0)  # 3% de 10.000
    allowed, reason = manager.can_open_position(now=datetime(2026, 8, 10, 10, 0))
    assert not allowed
    assert "perda" in reason.lower()


def test_bloqueia_fora_do_horario():
    manager = make_manager()
    allowed, _ = manager.can_open_position(now=datetime(2026, 8, 10, 8, 0))
    assert not allowed


def test_bloqueia_com_maximo_de_posicoes_abertas():
    manager = make_manager(open_positions_count=2)
    allowed, _ = manager.can_open_position(now=datetime(2026, 8, 10, 10, 0))
    assert not allowed


def test_position_size_respeita_risco_por_trade():
    manager = make_manager()
    # Risco por trade: 1% de 10.000 = 100. Stop a 5 pontos → 20 unidades.
    assert manager.position_size(entry_price=105.0, stop_loss=100.0) == 20.0


def test_position_size_zero_quando_stop_igual_entrada():
    manager = make_manager()
    assert manager.position_size(entry_price=100.0, stop_loss=100.0) == 0.0
