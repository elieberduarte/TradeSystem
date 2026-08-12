"""Testes da variante somente-compra do Donchian."""

from src.bot.strategies.base import SignalType
from src.bot.strategies.donchian import DonchianStrategy
from tests.test_swing_strategies import daily


def test_long_only_ignora_rompimento_de_baixa():
    closes = [100_000.0] * 60 + [98_000.0]
    normal = DonchianStrategy({"channel": 20}).generate_signal("WIN", daily(closes))
    long_only = DonchianStrategy({"channel": 20, "long_only": True}).generate_signal("WIN", daily(closes))

    assert normal.type == SignalType.SELL
    assert long_only.type == SignalType.HOLD


def test_long_only_mantem_as_compras():
    closes = [100_000.0] * 60 + [102_000.0]
    signal = DonchianStrategy({"channel": 20, "long_only": True}).generate_signal("WIN", daily(closes))
    assert signal.type == SignalType.BUY
