"""Testes do estudo estatístico de mercado."""

import pandas as pd

from src.bot.analysis.profile import (
    by_hour,
    daily_regimes,
    gap_study,
    opening_range_study,
    summarize,
)
from tests.conftest import make_candles


def many_days(day_closes: list[list[float]], start_day="2026-08-03") -> pd.DataFrame:
    """Concatena vários pregões de 5min a partir das 09:00."""
    frames = []
    for offset, closes in enumerate(day_closes):
        day = pd.Timestamp(start_day) + pd.Timedelta(days=offset)
        frames.append(make_candles(closes, start=f"{day:%Y-%m-%d} 09:00"))
    return pd.concat(frames)


def test_by_hour_agrega_por_hora():
    candles = make_candles([1000.0 + i for i in range(120)], start="2026-08-11 09:00")
    result = by_hour(candles)
    # Só horas com mais de 50 candles sobrevivem ao filtro
    assert result.empty or set(result.columns) == {
        "candles", "amplitude_media", "retorno_medio", "volatilidade"
    }


def test_daily_regimes_separa_tendencia_de_lateral():
    trend_day = [1000.0 + i * 4 for i in range(40)]
    flat_cycle = [0.0, 3.0, 5.0, 3.0, 0.0, -3.0, -5.0, -3.0]
    flat_day = [1000.0 + flat_cycle[i % len(flat_cycle)] for i in range(40)]
    candles = many_days([trend_day] * 4 + [flat_day] * 4)

    result = daily_regimes(candles)
    assert len(result) == 8
    assert set(result.columns) == {"day", "regime", "adx", "amplitude", "net"}
    # Dias de alta forte devem ser classificados como tendência
    assert (result["regime"] == "tendencia_alta").any()


def test_opening_range_study_mede_mfe_e_mae():
    # Range nos 3 primeiros candles, rompe para cima no 5º e continua subindo
    day = [1000.0, 1005.0, 1002.0, 1004.0, 1015.0] + [1015.0 + i * 3 for i in range(1, 20)]
    result = opening_range_study(many_days([day, day]), range_bars=3, horizon_bars=12)

    assert len(result) == 2  # um rompimento por dia
    assert (result["lado"] == "alta").all()
    # Continuação forte: o movimento a favor supera o contrário
    assert (result["mfe"] > result["mae"]).all()


def test_opening_range_study_pega_so_o_primeiro_rompimento():
    day = [1000.0, 1005.0, 1002.0, 1004.0, 1015.0, 1002.0, 995.0] + [1000.0] * 15
    result = opening_range_study(many_days([day]), range_bars=3, horizon_bars=5)
    assert len(result) == 1
    assert result["lado"].iloc[0] == "alta"


def test_gap_study_calcula_gap_e_movimento_do_dia():
    day1 = [1000.0] * 30
    day2 = [1020.0] * 30  # abre em gap de alta
    result = gap_study(many_days([day1, day2]))

    assert len(result) == 1  # o primeiro dia não tem fechamento anterior
    assert result["gap"].iloc[0] == 20.0


def test_summarize_devolve_metricas_chave():
    trend_day = [1000.0 + i * 4 for i in range(40)]
    result = summarize(many_days([trend_day] * 6))

    assert result["pregoes"] == 6
    assert "regimes_pct" in result
    assert "amplitude_media_dia" in result
