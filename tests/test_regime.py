"""Testes do classificador de regime de mercado."""

from src.bot.analysis.regime import Regime, classify
from src.bot.strategies.base import BaseStrategy, Signal, SignalType
from src.bot.strategies.regime_filter import RegimeFilteredStrategy
from tests.conftest import make_candles


def uptrend():
    return make_candles([1000.0 + i * 3 for i in range(80)])


def downtrend():
    return make_candles([1000.0 - i * 3 for i in range(80)])


def sideways():
    # Oscilação cíclica em torno de 1000, sem direção predominante
    cycle = [0.0, 3.0, 5.0, 3.0, 0.0, -3.0, -5.0, -3.0]
    return make_candles([1000.0 + cycle[i % len(cycle)] for i in range(80)])


def test_tendencia_de_alta():
    assert classify(uptrend()) == Regime.TREND_UP


def test_tendencia_de_baixa():
    assert classify(downtrend()) == Regime.TREND_DOWN


def test_mercado_lateral():
    assert classify(sideways()) == Regime.RANGE


def test_poucos_dados_e_tratado_como_lateral():
    assert classify(make_candles([1000.0] * 10)) == Regime.RANGE


class AlwaysBuyStrategy(BaseStrategy):
    def generate_signal(self, symbol, candles):
        close = float(candles["close"].iloc[-1])
        return Signal(
            symbol=symbol, type=SignalType.BUY,
            entry_price=close, stop_loss=close - 10, take_profit=close + 20,
        )


def test_filtro_de_regime_veta_no_mercado_errado():
    strategy = RegimeFilteredStrategy(AlwaysBuyStrategy(), allowed={Regime.TREND_UP})
    # Em alta: sinal passa
    assert strategy.generate_signal("WIN", uptrend()).type == SignalType.BUY
    # Lateral: sinal vetado
    assert strategy.generate_signal("WIN", sideways()).type == SignalType.HOLD
    # Baixa: sinal vetado
    assert strategy.generate_signal("WIN", downtrend()).type == SignalType.HOLD
