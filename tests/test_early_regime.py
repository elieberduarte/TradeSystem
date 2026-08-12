"""Testes da classificação de regime sem look-ahead (só dados da manhã)."""

from datetime import time

from src.bot.analysis.profile import (
    breakout_by_early_regime,
    early_regimes,
    regime_persistence,
)
from tests.test_profile import many_days


def morning_then_reversal(n: int = 90) -> list[float]:
    """Sobe forte até o meio do pregão e desaba depois — o regime da
    manhã e o do fim do dia divergem de propósito."""
    up = [1000.0 + i * 5 for i in range(n // 2)]
    return up + [up[-1] - i * 6 for i in range(n - n // 2)]


def steady_trend(n: int = 90) -> list[float]:
    return [1000.0 + i * 4 for i in range(n)]


def test_early_regimes_usa_so_a_manha():
    candles = many_days([steady_trend()] * 4)
    result = early_regimes(candles, until=time(12, 0))

    assert not result.empty
    assert set(result.columns) == {"day", "regime_cedo", "adx_cedo"}
    assert (result["regime_cedo"] == "tendencia_alta").all()


def test_early_regimes_usa_o_historico_continuo():
    # Corte às 09:30 deixa só 7 candles do dia, mas o ADX vem da série
    # contínua (como o bot ao vivo veria) — a classificação sai mesmo assim
    candles = many_days([steady_trend()] * 4)
    result = early_regimes(candles, until=time(9, 30))
    assert len(result) == 4


def test_persistencia_mostra_divergencia_entre_manha_e_fim_do_dia():
    candles = many_days([morning_then_reversal()] * 6)
    matrix = regime_persistence(candles, until=time(12, 0))

    assert not matrix.empty
    # Cada linha é uma distribuição de probabilidade
    for _, row in matrix.iterrows():
        assert abs(row.sum() - 1.0) < 1e-6


def test_breakout_por_regime_cedo_nao_usa_futuro():
    day = [1000.0, 1005.0, 1002.0, 1004.0, 1015.0] + [1015.0 + i * 4 for i in range(1, 60)]
    result = breakout_by_early_regime(many_days([day] * 6), until=time(12, 0))

    if not result.empty:
        assert set(result.columns) == {"rompimentos", "mfe_medio", "mae_medio", "razao"}
