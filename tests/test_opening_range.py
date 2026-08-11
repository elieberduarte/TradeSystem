"""Testes do rompimento da abertura (ORB)."""

from src.bot.strategies.base import SignalType
from src.bot.strategies.opening_range import OpeningRangeStrategy
from tests.conftest import make_candles


def day_candles(closes):
    """Candles de 5min começando na abertura do pregão (09:00)."""
    return make_candles(closes, start="2026-08-11 09:00")


def test_hold_durante_a_formacao_do_range():
    strategy = OpeningRangeStrategy({"range_bars": 3})
    candles = day_candles([1000.0, 1005.0, 1002.0])
    assert strategy.generate_signal("WIN", candles).type == SignalType.HOLD


def test_compra_no_candle_que_fecha_acima_do_range():
    strategy = OpeningRangeStrategy({"range_bars": 3, "rr": 1.5})
    # Range: closes 1000-1005 (highs até 1006); 4º candle ainda dentro,
    # 5º fecha acima do range → rompimento
    candles = day_candles([1000.0, 1005.0, 1002.0, 1004.0, 1015.0])
    signal = strategy.generate_signal("WIN", candles)

    assert signal.type == SignalType.BUY
    assert signal.entry_price == 1015.0
    # Stop no lado oposto do range (mínima da abertura: 1000 - 1 do gerador)
    assert signal.stop_loss == 999.0
    assert signal.take_profit == 1015.0 + 1.5 * (1015.0 - 999.0)


def test_venda_no_rompimento_para_baixo():
    strategy = OpeningRangeStrategy({"range_bars": 3})
    candles = day_candles([1000.0, 1005.0, 1002.0, 1001.0, 990.0])
    signal = strategy.generate_signal("WIN", candles)

    assert signal.type == SignalType.SELL
    assert signal.stop_loss == 1006.0  # máxima da abertura (1005 + 1)


def test_sem_sinal_repetido_apos_o_rompimento():
    strategy = OpeningRangeStrategy({"range_bars": 3})
    # Rompeu no 5º candle e segue subindo: candles seguintes não geram sinal
    candles = day_candles([1000.0, 1005.0, 1002.0, 1004.0, 1015.0, 1020.0, 1025.0])
    assert strategy.generate_signal("WIN", candles).type == SignalType.HOLD


def test_range_reinicia_a_cada_dia():
    strategy = OpeningRangeStrategy({"range_bars": 3})
    # Dia 1 inteiro + início do dia 2 ainda formando o range
    day1 = [1000.0, 1005.0, 1002.0, 1004.0, 1015.0, 1020.0]
    day2 = [1020.0, 1022.0]
    candles1 = make_candles(day1, start="2026-08-11 09:00")
    candles2 = make_candles(day2, start="2026-08-12 09:00")
    import pandas as pd

    candles = pd.concat([candles1, candles2])
    assert strategy.generate_signal("WIN", candles).type == SignalType.HOLD
