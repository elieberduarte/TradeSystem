"""Testes da volatilidade dessazonalizada por hora."""

import pandas as pd

from src.bot.analysis.volatility import (
    atr,
    deseasonalized_atr,
    hourly_profile,
    true_range,
)


def intraday(ranges_by_hour: dict[int, float], days: int = 10) -> pd.DataFrame:
    """Candles horários com amplitude controlada por hora do dia."""
    rows, index = [], []
    for day in range(days):
        base = pd.Timestamp("2026-08-03") + pd.Timedelta(days=day)
        for hour, span in ranges_by_hour.items():
            index.append(base + pd.Timedelta(hours=hour))
            rows.append((1000.0, 1000.0 + span / 2, 1000.0 - span / 2, 1000.0))
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [100.0] * len(rows),
        },
        index=pd.DatetimeIndex(index),
    ).sort_index()


def test_true_range_usa_a_maior_das_tres_medidas():
    df = intraday({10: 100.0}, days=2)
    assert (true_range(df).dropna() > 0).all()


def test_perfil_horario_captura_a_diferenca_entre_horas():
    # Manhã com o dobro da amplitude da tarde (40 dias > min_samples)
    df = intraday({10: 200.0, 15: 100.0}, days=40)
    profile = hourly_profile(df)

    assert set(profile.index) == {10, 15}
    assert profile[10] > profile[15] * 1.5


def test_perfil_ignora_horas_com_poucas_amostras():
    df = intraday({10: 200.0, 15: 100.0}, days=5)
    assert hourly_profile(df).empty


def test_dessazonalizado_neutraliza_a_hora():
    # Sem a correção, o ATR das 10h pareceria sempre "mercado agitado"
    df = intraday({10: 200.0, 15: 100.0}, days=30)
    ratio = deseasonalized_atr(df, period=5)

    manha = ratio[ratio.index.hour == 10].tail(10).mean()
    tarde = ratio[ratio.index.hour == 15].tail(10).mean()
    # As razões ficam na mesma ordem de grandeza, ao contrário do ATR bruto
    assert 0.5 < manha / tarde < 2.0


def test_atr_bruto_confunde_hora_com_agitacao():
    # Controle: o ATR sem correção mostra a distorção que motivou o módulo
    df = intraday({10: 200.0, 15: 100.0}, days=30)
    valores = atr(df, period=5)
    assert valores.dropna().max() > valores.dropna().min()


def test_serie_sem_perfil_devolve_um():
    df = intraday({10: 100.0}, days=1)  # amostras insuficientes
    assert (deseasonalized_atr(df, period=3) == 1.0).all()
