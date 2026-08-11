"""Testes do modo swing trade: overnight, gaps e limites próprios."""

from datetime import date, datetime, time

import pandas as pd
import pytest

from src.bot.backtest.engine import BacktestEngine
from src.bot.config import validate_symbol_modes
from src.bot.data.contracts import days_to_expiry, wdo_expiry
from src.bot.risk.manager import RiskConfig, RiskManager
from tests.conftest import make_candles
from tests.test_backtest_engine import BuyOnceStrategy


def swing_risk(**overrides) -> RiskManager:
    defaults = dict(
        capital=100_000.0,
        max_risk_per_trade_pct=1.0,
        max_daily_loss_pct=50.0,
        max_open_positions=1,
        mode="swing_trade",
    )
    return RiskManager(RiskConfig(**{**defaults, **overrides}))


def two_day_candles(day1_closes, day2_closes, day2_open=None):
    c1 = make_candles(day1_closes, start="2026-08-11 10:00")
    c2 = make_candles(day2_closes, start="2026-08-12 10:00")
    if day2_open is not None:
        # Gap de abertura: o 1º candle do dia 2 abre longe do fechamento anterior
        first = c2.index[0]
        c2.loc[first, "open"] = day2_open
        c2.loc[first, "low"] = min(c2.loc[first, "low"], day2_open)
        c2.loc[first, "high"] = max(c2.loc[first, "high"], day2_open)
    return pd.concat([c1, c2])


def test_swing_nao_zera_por_horario():
    manager = swing_risk()
    assert not manager.should_flatten(datetime(2026, 8, 11, 18, 0))
    assert not manager.should_flatten(datetime(2026, 8, 11, 23, 59))


def test_swing_atravessa_a_noite_no_backtest():
    candles = two_day_candles([1000.0] * 8, [1001.0] * 5)
    engine = BacktestEngine(
        BuyOnceStrategy(at=5, entry=1000.0, stop=900.0, target=1100.0),
        swing_risk(),
        warmup=2,
    )
    result = engine.run("WDO", candles)

    assert len(result.trades) == 1
    trade = result.trades[0]
    # Sobreviveu à virada do dia: só fechou no fim do histórico, no dia 2
    assert trade.exit_reason == "fim do histórico"
    assert trade.exit_time.date() == date(2026, 8, 12)


def test_gap_contra_sai_pior_que_o_stop_nominal():
    # Compra a 1000 com stop 995; dia 2 abre em gap a 985 (pula o stop)
    candles = two_day_candles([1000.0] * 8, [985.0] * 3, day2_open=985.0)
    engine = BacktestEngine(
        BuyOnceStrategy(at=5, entry=1000.0, stop=995.0, target=1100.0),
        swing_risk(),
        warmup=2,
    )
    result = engine.run("WDO", candles)

    trade = result.trades[0]
    assert trade.exit_reason == "stop (gap)"
    assert trade.exit_price == 985.0
    # A perda real é o dobro da nominal: gap não respeita stop
    assert trade.pnl < (995.0 - 1000.0) * trade.quantity


def test_limite_de_perda_semanal_bloqueia_novas_entradas():
    manager = swing_risk(max_weekly_loss_pct=5.0)
    manager.register_trade_result(-5000.0)  # 5% de 100k
    allowed, reason = manager.can_open_position(now=datetime(2026, 8, 12, 10, 0))
    assert not allowed
    assert "semanal" in reason
    # Semana nova libera
    manager.reset_week()
    manager.reset_day()
    assert manager.can_open_position(now=datetime(2026, 8, 17, 10, 0))[0]


def test_config_rejeita_mesmo_simbolo_nos_dois_modos():
    with pytest.raises(ValueError, match="netting"):
        validate_symbol_modes([("WIN", "day_trade"), ("win", "swing_trade")])
    # Símbolos distintos: ok
    validate_symbol_modes([("WIN", "day_trade"), ("WDO", "swing_trade")])


def test_dias_ate_o_vencimento():
    # WIN: vencimento 12/08/2026 (quarta mais próxima do dia 15)
    assert days_to_expiry("WIN", date(2026, 8, 10)) == 2
    # WDO: contrato de setembro vence no 1º dia útil (01/09/2026, terça)
    assert wdo_expiry(2026, 9) == date(2026, 9, 1)
    assert days_to_expiry("WDO", date(2026, 8, 10)) == 22
    with pytest.raises(ValueError):
        days_to_expiry("PETR4", date(2026, 8, 10))
