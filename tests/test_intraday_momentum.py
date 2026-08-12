"""Testes do momentum intradiário de horário."""

from datetime import time

import pandas as pd

from src.bot.strategies.base import SignalType
from src.bot.strategies.intraday_momentum import IntradayMomentumStrategy
from tests.conftest import make_candles


def session(day: str, closes: list[float], start="09:00"):
    return make_candles(closes, start=f"{day} {start}")


def two_sessions(day1_closes, day2_closes):
    """Pregão anterior completo + pregão atual até 17:35 (5min)."""
    d1 = session("2026-08-11", day1_closes)
    d2 = session("2026-08-12", day2_closes)
    return pd.concat([d1, d2])


# 09:00 → 17:35 em candles de 5min são 104 barras
BARS_TO_1735 = 104


def build_day(first_half_hour_close: float, rest_close: float) -> list[float]:
    """Dia que fecha a 1ª meia hora (6 candles) num nível e segue estável."""
    return [first_half_hour_close] * 7 + [rest_close] * (BARS_TO_1735 - 6)


def strategy(**params):
    return IntradayMomentumStrategy(
        {"entry_from": time(17, 30), "entry_until": time(17, 40), **params}
    )


def test_compra_quando_a_abertura_foi_de_alta():
    # Ontem fechou em 1000; hoje a 1ª meia hora fecha em 1020 (alta)
    candles = two_sessions([1000.0] * 20, build_day(1020.0, 1015.0))
    signal = strategy().generate_signal("WIN", candles)

    assert signal.type == SignalType.BUY
    assert signal.stop_loss < signal.entry_price


def test_vende_quando_a_abertura_foi_de_baixa():
    candles = two_sessions([1000.0] * 20, build_day(980.0, 985.0))
    signal = strategy().generate_signal("WIN", candles)

    assert signal.type == SignalType.SELL
    assert signal.stop_loss > signal.entry_price


def test_modo_contrarian_inverte_o_sinal():
    candles = two_sessions([1000.0] * 20, build_day(1020.0, 1015.0))
    assert strategy(contrarian=True).generate_signal("WIN", candles).type == SignalType.SELL


def test_hold_fora_da_janela_de_execucao():
    # Janela de execução deslocada: às 17:35 ainda não é hora
    s = IntradayMomentumStrategy({"entry_from": time(17, 50), "entry_until": time(17, 55)})
    candles = two_sessions([1000.0] * 20, build_day(1020.0, 1015.0))
    assert s.generate_signal("WIN", candles).type == SignalType.HOLD


def test_hold_sem_pregao_anterior():
    candles = session("2026-08-12", build_day(1020.0, 1015.0))
    assert strategy().generate_signal("WIN", candles).type == SignalType.HOLD


def test_hold_quando_a_abertura_nao_saiu_do_lugar():
    candles = two_sessions([1000.0] * 20, build_day(1000.0, 1000.0))
    assert strategy().generate_signal("WIN", candles).type == SignalType.HOLD


def test_filtro_de_movimento_minimo_veta_dia_sem_informacao():
    # Movimento de 2 pts contra amplitude média muito maior → vetado
    prev = [1000.0 + (i % 2) * 100 for i in range(20)]  # dias amplos
    candles = two_sessions(prev, build_day(1002.0, 1002.0))
    assert strategy(min_move_mult=1.0).generate_signal("WIN", candles).type == SignalType.HOLD
