"""Testes das regras de risco da camada diária (overtrading, sequência
de derrotas e zeragem de fim de dia)."""

from datetime import datetime, time

import pandas as pd

from src.bot.backtest.engine import BacktestEngine
from src.bot.risk.manager import RiskConfig, RiskManager
from tests.conftest import make_candles
from tests.test_backtest_engine import BuyOnceStrategy

MIDDAY = datetime(2026, 8, 11, 10, 0)


def make_manager(**config_overrides) -> RiskManager:
    config = RiskConfig(
        capital=10_000.0,
        max_risk_per_trade_pct=1.0,
        max_daily_loss_pct=50.0,
        max_open_positions=5,
        **config_overrides,
    )
    return RiskManager(config=config)


def test_limite_de_trades_por_dia():
    manager = make_manager(max_trades_per_day=2)
    manager.register_trade_result(10.0)
    manager.register_trade_result(10.0)
    allowed, reason = manager.can_open_position(now=MIDDAY)
    assert not allowed
    assert "trades no dia" in reason


def test_sem_limite_de_trades_quando_zero():
    manager = make_manager(max_trades_per_day=0)
    for _ in range(10):
        manager.register_trade_result(10.0)
    allowed, _ = manager.can_open_position(now=MIDDAY)
    assert allowed


def test_derrotas_consecutivas_pausam_o_dia():
    manager = make_manager(max_consecutive_losses=3)
    for _ in range(3):
        manager.register_trade_result(-10.0)
    allowed, reason = manager.can_open_position(now=MIDDAY)
    assert not allowed
    assert "consecutivas" in reason


def test_vitoria_zera_a_sequencia_de_derrotas():
    manager = make_manager(max_consecutive_losses=3)
    manager.register_trade_result(-10.0)
    manager.register_trade_result(-10.0)
    manager.register_trade_result(50.0)  # vitória reseta
    manager.register_trade_result(-10.0)
    allowed, _ = manager.can_open_position(now=MIDDAY)
    assert allowed


def test_reset_day_limpa_contadores():
    manager = make_manager(max_trades_per_day=1, max_consecutive_losses=1)
    manager.register_trade_result(-10.0)
    assert not manager.can_open_position(now=MIDDAY)[0]
    manager.reset_day()
    assert manager.can_open_position(now=MIDDAY)[0]


def test_should_flatten_no_fim_do_dia():
    manager = make_manager(flat_time=time(17, 45))
    assert not manager.should_flatten(datetime(2026, 8, 11, 17, 40))
    assert manager.should_flatten(datetime(2026, 8, 11, 17, 45))
    assert manager.should_flatten(datetime(2026, 8, 11, 18, 0))


def test_backtest_zera_posicao_no_fim_do_dia():
    # Candles de 5min das 17:20 às 18:00; entrada às 17:40, sem stop/alvo
    # atingível: a saída deve ser a zeragem diária no candle das 17:45.
    closes = [1000.0] * 9
    candles = make_candles(closes, start="2026-08-11 17:20")
    risk = RiskManager(
        RiskConfig(
            capital=100_000.0,
            max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=50.0,
            max_open_positions=1,
            trading_end=time(17, 44),
            flat_time=time(17, 45),
        )
    )
    engine = BacktestEngine(
        BuyOnceStrategy(at=4, entry=1000.0, stop=900.0, target=1100.0),
        risk,
        warmup=2,
    )
    result = engine.run("WIN", candles)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "zeragem diária"
    assert trade.exit_time == pd.Timestamp("2026-08-11 17:45")
