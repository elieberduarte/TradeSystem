"""Testes das estratégias de swing (candles diários)."""

import pandas as pd

from src.bot.analysis.profile import swing_regime_persistence
from src.bot.strategies.base import SignalType
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.strategies.swing_reversion import SwingReversionStrategy, atr


def daily(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    closes_s = pd.Series(closes, dtype=float)
    opens = closes_s.shift(1).fillna(closes_s.iloc[0])
    return pd.DataFrame(
        {
            "open": opens.values,
            "high": (pd.concat([opens, closes_s], axis=1).max(axis=1) + 50).values,
            "low": (pd.concat([opens, closes_s], axis=1).min(axis=1) - 50).values,
            "close": closes_s.values,
            "volume": [1000.0] * len(closes),
        },
        index=index,
    )


# ───────────────────────────── swing reversion ─────────────────────────────

def test_reversion_compra_apos_queda_longa():
    # Sobe, depois cai forte nos últimos 60 pregões
    closes = [100_000.0 + i * 60 for i in range(140)]
    closes += [closes[-1] - i * 250 for i in range(1, 61)]
    signal = SwingReversionStrategy({"lookback": 60, "threshold_std": 0.5}).generate_signal(
        "WIN", daily(closes)
    )
    assert signal.type == SignalType.BUY
    assert signal.stop_loss < signal.entry_price < signal.take_profit


def test_reversion_vende_apos_alta_longa():
    closes = [100_000.0 - i * 60 for i in range(140)]
    closes += [closes[-1] + i * 250 for i in range(1, 61)]
    signal = SwingReversionStrategy({"lookback": 60, "threshold_std": 0.5}).generate_signal(
        "WIN", daily(closes)
    )
    assert signal.type == SignalType.SELL


def test_reversion_hold_sem_movimento_relevante():
    closes = [100_000.0 + (i % 5) * 20 for i in range(200)]
    signal = SwingReversionStrategy({"lookback": 60, "threshold_std": 2.0}).generate_signal(
        "WIN", daily(closes)
    )
    assert signal.type == SignalType.HOLD


def test_reversion_hold_sem_historico():
    assert SwingReversionStrategy().generate_signal("WIN", daily([100.0] * 20)).type == SignalType.HOLD


def test_reversion_e_modo_swing():
    assert SwingReversionStrategy().mode == "swing_trade"


# ──────────────────────────────── donchian ─────────────────────────────────

def test_donchian_compra_no_rompimento_da_maxima():
    closes = [100_000.0] * 60 + [102_000.0]
    signal = DonchianStrategy({"channel": 20}).generate_signal("WIN", daily(closes))
    assert signal.type == SignalType.BUY
    assert signal.take_profit > signal.entry_price


def test_donchian_vende_no_rompimento_da_minima():
    closes = [100_000.0] * 60 + [98_000.0]
    signal = DonchianStrategy({"channel": 20}).generate_signal("WIN", daily(closes))
    assert signal.type == SignalType.SELL


def test_donchian_hold_dentro_do_canal():
    closes = [100_000.0 + (i % 7) * 30 for i in range(60)]
    signal = DonchianStrategy({"channel": 20}).generate_signal("WIN", daily(closes))
    assert signal.type == SignalType.HOLD


def test_atr_positivo():
    values = atr(daily([100_000.0 + i * 10 for i in range(60)]), 14)
    assert (values.dropna() > 0).all()


# ───────────────────────── persistência de regime ──────────────────────────

def test_swing_regime_persistence_estrutura():
    closes = [100_000.0 + i * 50 for i in range(200)]
    result = swing_regime_persistence(daily(closes), horizons=(1, 5))

    assert list(result["horizonte_barras"]) == [1, 5]
    assert set(result.columns) == {
        "horizonte_barras", "mesmo_regime", "acaso", "ganho_sobre_acaso", "amostras"
    }
    # Tendência limpa: o regime persiste quase sempre
    assert (result["mesmo_regime"] > 0.8).all()
