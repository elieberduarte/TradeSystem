"""Testes do filtro de horário de sessão."""

from datetime import time

from src.bot.strategies.base import BaseStrategy, Signal, SignalType
from src.bot.strategies.session_filter import SessionFilteredStrategy
from tests.conftest import make_candles


class AlwaysBuy(BaseStrategy):
    mode = "swing_trade"

    def generate_signal(self, symbol, candles):
        close = float(candles["close"].iloc[-1])
        return Signal(
            symbol=symbol, type=SignalType.BUY,
            entry_price=close, stop_loss=close - 10, take_profit=close + 20,
        )


def at(hhmm: str):
    return make_candles([1000.0, 1001.0], start=f"2026-08-11 {hhmm}")


def test_libera_dentro_da_janela():
    strategy = SessionFilteredStrategy(AlwaysBuy(), start=time(9, 0), end=time(11, 0))
    # 2º candle às 09:35 (5min após 09:30)
    assert strategy.generate_signal("WIN", at("09:30")).type == SignalType.BUY


def test_veta_fora_da_janela():
    strategy = SessionFilteredStrategy(AlwaysBuy(), start=time(9, 0), end=time(11, 0))
    assert strategy.generate_signal("WIN", at("14:00")).type == SignalType.HOLD
    assert strategy.generate_signal("WIN", at("08:00")).type == SignalType.HOLD


def test_preserva_o_modo_da_estrategia_interna():
    strategy = SessionFilteredStrategy(AlwaysBuy(), start=time(9, 0), end=time(18, 0))
    assert strategy.mode == "swing_trade"
