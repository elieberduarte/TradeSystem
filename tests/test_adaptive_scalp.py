"""Testes do scalp com alvo e stop proporcionais à volatilidade."""

import pandas as pd

from src.bot.strategies.adaptive_scalp import AdaptiveScalpStrategy
from src.bot.strategies.base import SignalType


def candles(closes: list[float], span: float = 60.0) -> pd.DataFrame:
    """Candles de 1 minuto com amplitude fixa `span` (controla o ATR)."""
    closes_s = pd.Series(closes, dtype=float)
    opens = closes_s.shift(1).fillna(closes_s.iloc[0])
    return pd.DataFrame(
        {
            "open": opens.values,
            "high": (pd.concat([opens, closes_s], axis=1).max(axis=1) + span / 2).values,
            "low": (pd.concat([opens, closes_s], axis=1).min(axis=1) - span / 2).values,
            "close": closes_s.values,
            "volume": [100.0] * len(closes),
        },
        index=pd.date_range("2026-08-11 10:00", periods=len(closes), freq="1min"),
    )


def drop(size: float, n: int = 40, span: float = 60.0) -> pd.DataFrame:
    """Série estável seguida de queda abrupta nos últimos 5 candles."""
    return candles([1000.0] * n + [1000.0 - size * i / 5 for i in range(1, 6)], span=span)


def test_veta_mercado_pequeno_demais_para_a_friccao():
    # ATR baixo: a fricção fixa comeria o alvo
    strategy = AdaptiveScalpStrategy({"min_atr_friction": 3.0, "friction_points": 12.5})
    assert strategy.generate_signal("WIN", drop(200.0, span=10.0)).type == SignalType.HOLD


def test_opera_quando_o_mercado_esta_grande():
    strategy = AdaptiveScalpStrategy(
        {"min_atr_friction": 3.0, "friction_points": 12.5, "trigger_atr": 1.0}
    )
    signal = strategy.generate_signal("WIN", drop(400.0, span=120.0))
    assert signal.type == SignalType.BUY  # fade da queda


def test_alvo_e_stop_escalam_com_a_volatilidade():
    strategy = AdaptiveScalpStrategy(
        {"min_atr_friction": 1.0, "trigger_atr": 1.0, "target_atr": 1.0, "stop_atr": 1.0}
    )
    pequeno = strategy.generate_signal("WIN", drop(300.0, span=80.0))
    grande = strategy.generate_signal("WIN", drop(600.0, span=200.0))

    dist_pequeno = pequeno.take_profit - pequeno.entry_price
    dist_grande = grande.take_profit - grande.entry_price
    assert dist_grande > dist_pequeno * 1.5


def test_direcao_follow_inverte_a_aposta():
    params = {"min_atr_friction": 1.0, "trigger_atr": 1.0}
    fade = AdaptiveScalpStrategy({**params, "direction": "fade"})
    follow = AdaptiveScalpStrategy({**params, "direction": "follow"})
    data = drop(400.0, span=120.0)

    assert fade.generate_signal("WIN", data).type == SignalType.BUY
    assert follow.generate_signal("WIN", data).type == SignalType.SELL


def test_hold_sem_movimento_relevante():
    strategy = AdaptiveScalpStrategy({"min_atr_friction": 1.0, "trigger_atr": 3.0})
    assert strategy.generate_signal("WIN", drop(50.0, span=120.0)).type == SignalType.HOLD


def test_hold_sem_dados_suficientes():
    strategy = AdaptiveScalpStrategy()
    assert strategy.generate_signal("WIN", candles([1000.0] * 5)).type == SignalType.HOLD
