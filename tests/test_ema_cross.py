"""Testes da estratégia de cruzamento de EMAs."""

from src.bot.strategies.base import SignalType
from src.bot.strategies.ema_cross import EmaCrossStrategy
from tests.conftest import make_candles


def find_first_signal(strategy, candles):
    for i in range(30, len(candles)):
        signal = strategy.generate_signal("WIN", candles.iloc[: i + 1])
        if signal.type != SignalType.HOLD:
            return signal
    return None


def test_hold_com_poucos_candles():
    strategy = EmaCrossStrategy()
    candles = make_candles([1000.0] * 10)
    assert strategy.generate_signal("WIN", candles).type == SignalType.HOLD


def test_compra_em_reversao_para_alta(trending_up_candles):
    # Sem filtro de tendência para o cruzamento aparecer logo na virada
    strategy = EmaCrossStrategy({"trend": 0})
    signal = find_first_signal(strategy, trending_up_candles)
    assert signal is not None
    assert signal.type == SignalType.BUY
    assert signal.stop_loss < signal.entry_price < signal.take_profit


def test_alvo_respeita_relacao_risco_retorno(trending_up_candles):
    strategy = EmaCrossStrategy({"trend": 0, "rr": 2.0})
    signal = find_first_signal(strategy, trending_up_candles)
    risk = signal.entry_price - signal.stop_loss
    reward = signal.take_profit - signal.entry_price
    assert abs(reward - 2.0 * risk) < 1e-9


def test_filtro_de_tendencia_bloqueia_compra_contra_ema_longa(trending_up_candles):
    # Com EMA de tendência longa demais, o preço ainda está abaixo dela
    # no momento do cruzamento — o sinal deve ser vetado.
    with_filter = EmaCrossStrategy({"trend": 150})
    signal = find_first_signal(with_filter, trending_up_candles)
    without_filter = EmaCrossStrategy({"trend": 0})
    unfiltered = find_first_signal(without_filter, trending_up_candles)
    assert unfiltered is not None
    assert signal is None or signal.entry_price > unfiltered.entry_price
