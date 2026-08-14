"""Testes da saída por ADR e da entrada aleatória de controle."""

import pandas as pd

from src.bot.strategies.adr_exit import AdrExitOverlay, RandomEntryStrategy, adr
from src.bot.strategies.base import BaseStrategy, Signal, SignalType


def candles(n: int = 250, span: float = 10.0, price: float = 100.0) -> pd.DataFrame:
    """Candles com amplitude fixa `span`, para o ADR ser previsível."""
    return pd.DataFrame(
        {
            "open": [price] * n,
            "high": [price + span / 2] * n,
            "low": [price - span / 2] * n,
            "close": [price] * n,
            "volume": [1000.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )


class FixedBuy(BaseStrategy):
    mode = "swing_trade"

    def generate_signal(self, symbol, candles):
        return Signal(
            symbol=symbol, type=SignalType.BUY, entry_price=100.0,
            stop_loss=95.0, take_profit=110.0,
        )


class FixedSell(BaseStrategy):
    mode = "swing_trade"

    def generate_signal(self, symbol, candles):
        return Signal(
            symbol=symbol, type=SignalType.SELL, entry_price=100.0,
            stop_loss=105.0, take_profit=90.0,
        )


def test_adr_e_a_amplitude_media():
    values = adr(candles(span=10.0), period=200)
    assert abs(float(values.iloc[-1]) - 10.0) < 1e-9


def test_grade_lenta_usa_metade_e_um_terco():
    signal = AdrExitOverlay(FixedBuy(), grid="lento").generate_signal("WIN", candles(span=12.0))
    # ADR = 12 → alvo +6 (ADR/2), stop −4 (ADR/3)
    assert abs(signal.take_profit - 106.0) < 1e-9
    assert abs(signal.stop_loss - 96.0) < 1e-9


def test_payoff_e_o_mesmo_nas_tres_grades():
    """O autor apresenta três estratégias; na prática é uma só, com
    payoff ~1,5:1 reescalonado pelo timeframe."""
    payoffs = []
    for grid in ("rapido", "medio", "lento"):
        signal = AdrExitOverlay(FixedBuy(), grid=grid).generate_signal("WIN", candles())
        ganho = signal.take_profit - signal.entry_price
        risco = signal.entry_price - signal.stop_loss
        payoffs.append(ganho / risco)
    assert all(abs(p - 1.5) < 0.15 for p in payoffs)


def test_venda_espelha_os_niveis():
    signal = AdrExitOverlay(FixedSell(), grid="lento").generate_signal("WIN", candles(span=12.0))
    assert abs(signal.take_profit - 94.0) < 1e-9   # −ADR/2
    assert abs(signal.stop_loss - 104.0) < 1e-9    # +ADR/3


def test_hold_sem_historico_para_o_adr():
    signal = AdrExitOverlay(FixedBuy(), period=200).generate_signal("WIN", candles(n=50))
    assert signal.type == SignalType.HOLD


def test_hold_quando_a_amplitude_e_zero():
    flat = candles(span=0.0)
    assert AdrExitOverlay(FixedBuy()).generate_signal("WIN", flat).type == SignalType.HOLD


def test_grade_invalida_e_rejeitada():
    try:
        AdrExitOverlay(FixedBuy(), grid="inexistente")
    except ValueError as exc:
        assert "inexistente" in str(exc)
    else:
        raise AssertionError("deveria ter rejeitado a grade")


# ─────────────────────── entrada aleatória (controle) ───────────────────────

def test_entrada_aleatoria_respeita_a_frequencia():
    data = candles(n=1000)
    strategy = RandomEntryStrategy(probability=0.05, warmup=200)
    signals = sum(
        1 for i in range(200, len(data))
        if strategy.generate_signal("WIN", data.iloc[: i + 1]).type == SignalType.BUY
    )
    frequencia = signals / (len(data) - 200)
    assert 0.02 < frequencia < 0.09  # em torno de 5%


def test_entrada_aleatoria_e_deterministica():
    data = candles(n=300)
    a = RandomEntryStrategy(seed=7)
    b = RandomEntryStrategy(seed=7)
    for i in range(200, 300):
        window = data.iloc[: i + 1]
        assert a.generate_signal("WIN", window).type == b.generate_signal("WIN", window).type


def test_sementes_diferentes_dao_series_diferentes():
    data = candles(n=600)
    a = RandomEntryStrategy(seed=1, probability=0.1)
    b = RandomEntryStrategy(seed=999, probability=0.1)
    diferencas = sum(
        1 for i in range(200, 600)
        if a.generate_signal("WIN", data.iloc[: i + 1]).type
        != b.generate_signal("WIN", data.iloc[: i + 1]).type
    )
    assert diferencas > 0
