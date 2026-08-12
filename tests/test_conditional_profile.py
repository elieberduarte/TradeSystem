"""Testes das medidas condicionadas ao regime de mercado."""

from src.bot.analysis.profile import autocorrelation_by_regime, breakout_by_regime
from tests.test_profile import many_days


def trend_day(slope: float = 4.0, n: int = 60) -> list[float]:
    return [1000.0 + i * slope for i in range(n)]


def flat_day(n: int = 60) -> list[float]:
    cycle = [0.0, 3.0, 5.0, 3.0, 0.0, -3.0, -5.0, -3.0]
    return [1000.0 + cycle[i % len(cycle)] for i in range(n)]


def test_autocorrelacao_separa_por_regime():
    candles = many_days([trend_day()] * 5 + [flat_day()] * 5)
    result = autocorrelation_by_regime(candles, lags=(1, 3))

    assert not result.empty
    assert set(result.columns) >= {"regime", "candles", "lag_1", "lag_3"}
    # Cada regime presente aparece uma única vez
    assert result["regime"].is_unique


def test_autocorrelacao_negativa_em_serie_alternada():
    # Zigue-zague candle a candle: cada retorno desfaz o anterior
    zigzag = [1000.0 + (5.0 if i % 2 else 0.0) for i in range(60)]
    result = autocorrelation_by_regime(many_days([zigzag] * 6), lags=(1,))
    assert (result["lag_1"] < 0).all()


def test_autocorrelacao_positiva_em_onda_suave():
    # Ciclo suave: o movimento persiste por algumas barras antes de virar
    result = autocorrelation_by_regime(many_days([flat_day()] * 6), lags=(1,))
    assert (result["lag_1"] > 0).all()


def test_breakout_por_regime_devolve_razao():
    day = [1000.0, 1005.0, 1002.0, 1004.0, 1015.0] + [1015.0 + i * 3 for i in range(1, 30)]
    result = breakout_by_regime(many_days([day] * 6), range_bars=3, horizon_bars=10)

    if not result.empty:
        assert set(result.columns) == {"rompimentos", "mfe_medio", "mae_medio", "razao"}
        assert (result["rompimentos"] > 0).all()


def test_breakout_por_regime_vazio_sem_rompimentos():
    # Dias sem rompimento algum: preço preso no range de abertura
    flat = [1000.0] * 40
    assert breakout_by_regime(many_days([flat] * 3), range_bars=3).empty
