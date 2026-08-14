"""Testes das medidas de nowcast de regime."""

import numpy as np
import pandas as pd

from src.bot.analysis.nowcast import (
    kaufman_efficiency,
    max_retrace_fraction,
    microchannel,
    session_vwap,
    side_consistency,
)


def test_eficiencia_linha_reta_e_um():
    closes = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    assert kaufman_efficiency(closes) == 1.0


def test_eficiencia_serrote_e_baixa():
    closes = pd.Series([100.0, 102.0, 100.0, 102.0, 100.0])
    assert kaufman_efficiency(closes) == 0.0


def test_microcanal_conta_a_maior_sequencia():
    closes = pd.Series([101.0, 102.0, 103.0, 99.0, 101.0, 102.0])
    ema = pd.Series([100.0] * 6)
    # acima, acima, acima (3) | abaixo (1) | acima, acima (2)
    assert microchannel(closes, ema) == 3


def test_consistencia_de_lado():
    closes = pd.Series([101.0, 102.0, 103.0, 99.0])
    ema = pd.Series([100.0] * 4)
    assert side_consistency(closes, ema) == 0.75


def test_vwap_pondera_por_volume():
    bars = pd.DataFrame({
        "high": [101.0, 111.0], "low": [99.0, 109.0], "close": [100.0, 110.0],
        "volume": [1.0, 3.0],
    })
    vwap = session_vwap(bars)
    assert vwap.iloc[0] == 100.0
    assert vwap.iloc[1] == (100.0 * 1 + 110.0 * 3) / 4


def test_retracao_de_tendencia_limpa_e_rasa():
    closes = pd.Series([100.0, 102.0, 101.5, 104.0, 106.0])
    frac = max_retrace_fraction(closes, direction=1.0)
    assert frac == 0.5 / 6.0


def test_retracao_total_devolvida_passa_de_um():
    closes = pd.Series([100.0, 106.0, 99.0, 100.5])
    frac = max_retrace_fraction(closes, direction=1.0)
    assert frac > 1.0


def test_retracao_na_direcao_de_baixa():
    closes = pd.Series([100.0, 97.0, 98.0, 94.0])
    frac = max_retrace_fraction(closes, direction=-1.0)
    assert frac == 1.0 / 6.0
