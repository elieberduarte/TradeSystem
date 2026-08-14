"""Testes dos setups extraídos dos livros."""

import pandas as pd

from src.bot.strategies.base import SignalType
from src.bot.strategies.book_setups import InsideDayStrategy, OopsStrategy


def frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Cada linha: (open, high, low, close), em candles diários."""
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1000.0] * len(rows),
        },
        index=pd.date_range("2024-01-01", periods=len(rows), freq="B"),
    )


def base(n: int = 30) -> list[tuple[float, float, float, float]]:
    """Fundo neutro para o ATR ter histórico."""
    return [(100.0, 102.0, 98.0, 100.0)] * n


# ─────────────────────────────── Oops ───────────────────────────────

def test_oops_compra_quando_volta_a_minima_de_ontem():
    # Ontem: mínima 98. Hoje abre em 95 (gap abaixo) e sobe até 99.
    rows = base() + [(100.0, 102.0, 98.0, 99.0), (95.0, 99.0, 94.0, 98.5)]
    signal = OopsStrategy().generate_signal("WIN", frame(rows))

    assert signal.type == SignalType.BUY
    assert signal.entry_price == 98.0    # a mínima de ontem
    assert signal.stop_loss == 95.0      # a abertura em gap


def test_oops_nao_dispara_se_o_preco_nao_volta():
    # Abre em gap de baixa mas nunca alcança a mínima de ontem
    rows = base() + [(100.0, 102.0, 98.0, 99.0), (95.0, 97.0, 94.0, 96.0)]
    assert OopsStrategy().generate_signal("WIN", frame(rows)).type == SignalType.HOLD


def test_oops_nao_dispara_sem_gap():
    rows = base() + [(100.0, 102.0, 98.0, 99.0), (99.5, 101.0, 98.5, 100.0)]
    assert OopsStrategy().generate_signal("WIN", frame(rows)).type == SignalType.HOLD


def test_oops_lado_vendido_e_opcional():
    # Abre acima da máxima de ontem e volta a tocá-la
    rows = base() + [(100.0, 102.0, 98.0, 101.0), (105.0, 106.0, 101.5, 102.0)]
    data = frame(rows)

    assert OopsStrategy({"side": "long"}).generate_signal("WIN", data).type == SignalType.HOLD
    vendido = OopsStrategy({"side": "short"}).generate_signal("WIN", data)
    assert vendido.type == SignalType.SELL
    assert vendido.entry_price == 102.0   # a máxima de ontem


def test_oops_filtro_de_gap_minimo():
    # Gap de 1 ponto contra ATR de ~4: pequeno demais para o filtro
    rows = base() + [(100.0, 102.0, 98.0, 99.0), (97.0, 99.0, 96.5, 98.5)]
    data = frame(rows)

    assert OopsStrategy({"min_gap_atr": 0.0}).generate_signal("WIN", data).type == SignalType.BUY
    assert OopsStrategy({"min_gap_atr": 1.0}).generate_signal("WIN", data).type == SignalType.HOLD


# ───────────────────────────── Inside Day ─────────────────────────────

def test_inside_day_compra_no_rompimento_da_maxima():
    rows = base()
    rows += [(100.0, 110.0, 90.0, 105.0)]      # barra-mãe, range amplo
    rows += [(104.0, 106.0, 100.0, 102.0)]     # dia de dentro
    rows += [(102.0, 108.0, 101.0, 107.0)]     # rompe para cima
    signal = InsideDayStrategy({"rr": 2.0}).generate_signal("WIN", frame(rows))

    assert signal.type == SignalType.BUY
    assert signal.entry_price == 106.0         # máxima do dia de dentro
    assert signal.stop_loss == 100.0           # mínima do dia de dentro
    assert signal.take_profit == 106.0 + 2.0 * 6.0


def test_inside_day_vende_no_rompimento_da_minima():
    rows = base()
    rows += [(100.0, 110.0, 90.0, 105.0)]
    rows += [(104.0, 106.0, 100.0, 102.0)]
    rows += [(102.0, 103.0, 98.0, 99.0)]       # rompe para baixo
    signal = InsideDayStrategy().generate_signal("WIN", frame(rows))

    assert signal.type == SignalType.SELL
    assert signal.entry_price == 100.0
    assert signal.stop_loss == 106.0


def test_inside_day_exige_conter_a_barra_anterior():
    rows = base()
    rows += [(100.0, 110.0, 90.0, 105.0)]
    rows += [(104.0, 112.0, 100.0, 102.0)]     # máxima ultrapassa a mãe
    rows += [(102.0, 115.0, 101.0, 114.0)]
    assert InsideDayStrategy().generate_signal("WIN", frame(rows)).type == SignalType.HOLD


def test_inside_day_hold_sem_rompimento():
    rows = base()
    rows += [(100.0, 110.0, 90.0, 105.0)]
    rows += [(104.0, 106.0, 100.0, 102.0)]
    rows += [(102.0, 105.0, 101.0, 103.0)]     # fica dentro
    assert InsideDayStrategy().generate_signal("WIN", frame(rows)).type == SignalType.HOLD


def test_filtro_de_range_estreito():
    # Dia de dentro com range de 6, contra fundo de range 4: não é estreito
    rows = base()
    rows += [(100.0, 110.0, 90.0, 105.0)]
    rows += [(104.0, 106.0, 100.0, 102.0)]
    rows += [(102.0, 108.0, 101.0, 107.0)]
    data = frame(rows)

    assert InsideDayStrategy({"narrow_pct": 0.0}).generate_signal("WIN", data).type == SignalType.BUY
    # Exigindo o percentil 20 dos ranges recentes (≈4), o dia de dentro
    # com range 6 é vetado
    assert InsideDayStrategy({"narrow_pct": 0.2}).generate_signal("WIN", data).type == SignalType.HOLD


def test_setups_sao_modo_swing():
    assert OopsStrategy().mode == "swing_trade"
    assert InsideDayStrategy().mode == "swing_trade"
