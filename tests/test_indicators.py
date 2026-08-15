"""Testes do Ichimoku e do squeeze de Bollinger."""

import numpy as np
import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType
from src.bot.strategies.ichimoku import (
    CloudFilterOverlay,
    IchimokuCrossStrategy,
    ichimoku_lines,
)
from src.bot.strategies.squeeze import (
    SqueezeBreakoutStrategy,
    bollinger,
    squeeze_rank,
)


def frame(closes: list[float], spread: float = 1.0) -> pd.DataFrame:
    values = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": np.r_[values[0], values[:-1]],
        "high": values + spread,
        "low": values - spread,
        "close": values,
        "volume": np.full(len(values), 1000.0),
    }, index=pd.date_range("2024-01-01", periods=len(values), freq="B"))


class AlwaysBuy(BaseStrategy):
    mode = "swing_trade"

    def __init__(self):
        super().__init__({})

    def generate_signal(self, symbol, candles):
        close = float(candles["close"].iloc[-1])
        return Signal(symbol=symbol, type=SignalType.BUY, entry_price=close,
                      stop_loss=close - 5, take_profit=close + 15)


# ─────────────────────────── Ichimoku ───────────────────────────

def test_linhas_sao_pontos_medios_de_donchian():
    rng = np.random.default_rng(7)
    closes = list(100 + rng.normal(0, 2, 120).cumsum())
    data = frame(closes)
    lines = ichimoku_lines(data)

    tenkan_manual = (data["high"].rolling(9).max() + data["low"].rolling(9).min()) / 2
    assert lines["tenkan"].iloc[-1] == tenkan_manual.iloc[-1]
    # A nuvem de hoje foi calculada 26 barras atrás (deslocamento causal)
    span_b_manual = (
        (data["high"].rolling(52).max() + data["low"].rolling(52).min()) / 2
    ).shift(26)
    assert lines["cloud_top"].iloc[-1] >= span_b_manual.iloc[-1] or np.isnan(span_b_manual.iloc[-1])


def test_cruzamento_tk_gera_compra():
    # Fundo profundo há mais de 9 barras (pesa só no Kijun) e recuperação
    # recente (pesa no Tenkan): o Tenkan cruza o Kijun para cima no fim
    closes = ([100.0] * 70 + [80.0] * 10
              + [96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0])
    signal = IchimokuCrossStrategy().generate_signal("WIN", frame(closes))
    assert signal.type == SignalType.BUY
    assert signal.stop_loss < signal.entry_price < signal.take_profit


def test_sem_cruzamento_hold():
    closes = [100.0 + 0.1 * i for i in range(100)]   # tendência estável, sem cruzar
    signal = IchimokuCrossStrategy().generate_signal("WIN", frame(closes))
    assert signal.type == SignalType.HOLD


def test_filtro_de_nuvem_veta_compra_abaixo_dela():
    # Preço despenca para bem abaixo da nuvem construída no platô
    closes = [100.0] * 90 + [70.0, 68.0, 66.0]
    filtered = CloudFilterOverlay(AlwaysBuy())
    assert filtered.generate_signal("WIN", frame(closes)).type == SignalType.HOLD


def test_filtro_de_nuvem_deixa_passar_acima():
    closes = [100.0] * 90 + [130.0, 132.0, 134.0]
    filtered = CloudFilterOverlay(AlwaysBuy())
    assert filtered.generate_signal("WIN", frame(closes)).type == SignalType.BUY


# ─────────────────────────── Squeeze ───────────────────────────

def test_bollinger_matematica():
    closes = pd.Series([100.0, 102.0, 98.0, 101.0, 99.0] * 5)
    bands = bollinger(closes, period=5, k=2.0)
    mid = closes.rolling(5).mean().iloc[-1]
    assert abs(bands["middle"].iloc[-1] - mid) < 1e-9
    assert bands["upper"].iloc[-1] > bands["middle"].iloc[-1] > bands["lower"].iloc[-1]


def test_squeeze_rank_usa_a_largura_de_ontem():
    # Largura caindo até ontem; hoje explode — o rank deve ignorar hoje
    bandwidth = pd.Series([0.10] * 100 + [0.02, 0.50])
    rank = squeeze_rank(bandwidth, lookback=100)
    assert rank <= 0.02   # ontem (0,02) era a menor largura da janela


def test_squeeze_breakout_dispara_apos_compressao():
    rng = np.random.default_rng(3)
    quiet = list(100 + rng.normal(0, 0.3, 150).cumsum() * 0.1)
    closes = quiet + [quiet[-1] + 8.0]     # rompimento após 150 barras calmas
    signal = SqueezeBreakoutStrategy({"lookback": 100}).generate_signal(
        "WIN", frame(closes, spread=0.3)
    )
    assert signal.type == SignalType.BUY


def test_sem_squeeze_sem_sinal():
    rng = np.random.default_rng(4)
    noisy = list(100 + rng.normal(0, 3, 150).cumsum())   # largura alta o tempo todo
    closes = noisy + [noisy[-1] + 10.0]
    signal = SqueezeBreakoutStrategy({"lookback": 100}).generate_signal(
        "WIN", frame(closes, spread=3.0)
    )
    assert signal.type == SignalType.HOLD
