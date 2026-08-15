"""O gatilho da entrada viaja do sinal até o trade — sem entrada muda."""

from datetime import time

import pandas as pd

from src.bot.backtest.engine import BacktestEngine
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.strategies.ibs import IbsStrategy


def frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1000.0] * len(rows),
        },
        index=pd.date_range("2024-01-01", periods=len(rows), freq="B"),
    )


def breakout() -> pd.DataFrame:
    rows = [(100.0, 101.0, 99.0, 100.0)] * 40
    rows.append((100.0, 106.0, 100.0, 105.0))   # rompe a máxima do canal
    return frame(rows)


def test_donchian_explica_o_rompimento():
    signal = DonchianStrategy({"channel": 20}).generate_signal("WIN", breakout())
    assert signal.type.name == "BUY"
    assert "20" in signal.reason
    assert "ACIMA da máxima" in signal.reason
    assert "sem volume" in signal.reason


def test_ibs_explica_a_reversao():
    rows = [(100.0, 102.0, 98.0, 100.0)] * 30
    rows.append((100.0, 102.0, 96.0, 96.2))     # fechou colado na mínima
    signal = IbsStrategy({"entry_low": 0.2}).generate_signal("WIN", frame(rows))
    assert signal.type.name == "BUY"
    assert "IBS" in signal.reason
    assert "MÍNIMA" in signal.reason


def test_motivo_chega_ao_trade_no_motor():
    risk = RiskManager(RiskConfig(
        capital=100_000.0, max_risk_per_trade_pct=1.0, max_daily_loss_pct=100.0,
        max_open_positions=1, mode="swing_trade",
        trading_start=time(0, 0), trading_end=time(23, 59),
        max_consecutive_losses=0,
    ))
    engine = BacktestEngine(DonchianStrategy({"channel": 20}), risk, warmup=30)
    result = engine.run("WIN", breakout())
    assert result.trades
    assert "rompimento de canal" in result.trades[0].entry_reason
