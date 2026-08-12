"""Testes das estratégias de reversão à média (band fade e PFR)."""

import pandas as pd

from src.bot.strategies.band_fade import BandFadeStrategy, bands
from src.bot.strategies.base import SignalType
from src.bot.strategies.pfr import PfrStrategy
from tests.conftest import make_candles


# ─────────────────────────────── band fade ───────────────────────────────

def test_bands_bollinger_envolve_o_preco():
    candles = make_candles([1000.0 + (i % 7) for i in range(60)])
    mid, upper, lower = bands(candles, 20, 2.0, "bollinger")
    assert (upper.dropna() > mid.dropna()).all()
    assert (lower.dropna() < mid.dropna()).all()


def test_bands_keltner_usa_atr():
    candles = make_candles([1000.0 + (i % 7) for i in range(60)])
    _, upper_k, _ = bands(candles, 20, 2.0, "keltner")
    _, upper_b, _ = bands(candles, 20, 2.0, "bollinger")
    # Larguras diferentes: são medidas distintas de dispersão
    assert upper_k.iloc[-1] != upper_b.iloc[-1]


def test_compra_quando_fecha_fora_e_volta_para_dentro():
    # Série estável, um mergulho abaixo da banda e recuperação parcial
    closes = [1000.0] * 40 + [940.0, 985.0]
    strategy = BandFadeStrategy({"period": 20, "mult": 2.0, "target": 1.5})
    signal = strategy.generate_signal("WIN", make_candles(closes))

    assert signal.type == SignalType.BUY
    assert signal.stop_loss < signal.entry_price < signal.take_profit


def test_venda_quando_estoura_acima_e_volta():
    closes = [1000.0] * 40 + [1060.0, 1015.0]
    strategy = BandFadeStrategy({"period": 20, "mult": 2.0, "target": 1.5})
    signal = strategy.generate_signal("WIN", make_candles(closes))

    assert signal.type == SignalType.SELL
    assert signal.stop_loss > signal.entry_price > signal.take_profit


def test_hold_enquanto_o_preco_fica_dentro_das_bandas():
    closes = [1000.0 + (i % 3) for i in range(60)]
    strategy = BandFadeStrategy({"period": 20})
    assert strategy.generate_signal("WIN", make_candles(closes)).type == SignalType.HOLD


def test_hold_sem_dados_suficientes():
    strategy = BandFadeStrategy({"period": 20})
    assert strategy.generate_signal("WIN", make_candles([1000.0] * 5)).type == SignalType.HOLD


def test_alvo_na_media_central():
    closes = [1000.0] * 40 + [940.0, 985.0]
    strategy = BandFadeStrategy({"period": 20, "target": "mid"})
    signal = strategy.generate_signal("WIN", make_candles(closes))
    mid, _, _ = bands(make_candles(closes), 20, 2.0, "bollinger")
    assert signal.take_profit == float(mid.iloc[-1])


def test_hold_quando_a_volta_ja_ultrapassou_a_media():
    # Recuperação total: a reversão já aconteceu, não há o que capturar
    closes = [1000.0] * 40 + [940.0, 1000.0]
    strategy = BandFadeStrategy({"period": 20, "target": "mid"})
    assert strategy.generate_signal("WIN", make_candles(closes)).type == SignalType.HOLD


# ─────────────────────────────────── PFR ──────────────────────────────────

def test_pfr_de_compra():
    # Último candle: mínima abaixo das duas anteriores, fecha acima
    candles = pd.DataFrame(
        {
            "open":   [1000.0, 998.0, 990.0],
            "high":   [1005.0, 1000.0, 999.0],
            "low":    [995.0, 990.0, 985.0],
            "close":  [998.0, 992.0, 997.0],
            "volume": [100.0] * 3,
        },
        index=pd.date_range("2026-08-11 10:00", periods=3, freq="5min"),
    )
    signal = PfrStrategy({"rr": 1.0}).generate_signal("WIN", candles)

    assert signal.type == SignalType.BUY
    assert signal.stop_loss == 985.0
    # Payoff 1:1 sobre o risco
    assert signal.take_profit == 997.0 + (997.0 - 985.0)


def test_pfr_de_venda():
    candles = pd.DataFrame(
        {
            "open":   [1000.0, 1002.0, 1010.0],
            "high":   [1005.0, 1008.0, 1015.0],
            "low":    [995.0, 1000.0, 1002.0],
            "close":  [1002.0, 1007.0, 1003.0],
            "volume": [100.0] * 3,
        },
        index=pd.date_range("2026-08-11 10:00", periods=3, freq="5min"),
    )
    signal = PfrStrategy().generate_signal("WIN", candles)

    assert signal.type == SignalType.SELL
    assert signal.stop_loss == 1015.0


def test_pfr_hold_sem_padrao():
    candles = make_candles([1000.0, 1001.0, 1002.0, 1003.0])
    assert PfrStrategy().generate_signal("WIN", candles).type == SignalType.HOLD
