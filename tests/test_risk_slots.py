"""Testes da divisão do orçamento de risco entre instrumentos."""

from src.bot.risk.manager import RiskConfig, RiskManager


def manager(slots: int) -> RiskManager:
    return RiskManager(
        RiskConfig(
            capital=100_000.0, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=50.0, max_open_positions=1,
            risk_slots=slots,
        )
    )


def test_um_slot_arrisca_o_percentual_cheio():
    # 1% de 100k = 1.000; stop de 10 pontos → 100 unidades
    assert manager(1).position_size(entry_price=110.0, stop_loss=100.0) == 100.0


def test_varios_slots_dividem_o_risco():
    # Mesmo trade com 10 instrumentos: cada um leva 1/10 do risco
    assert manager(10).position_size(entry_price=110.0, stop_loss=100.0) == 10.0


def test_slots_invalidos_nao_quebram():
    assert manager(0).position_size(entry_price=110.0, stop_loss=100.0) == 100.0
