"""Testes da estratégia IBS (Internal Bar Strength)."""

import pandas as pd

from src.bot.strategies.base import SignalType
from src.bot.strategies.ibs import IbsStrategy, ibs


def frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Cada linha: (open, high, low, close)."""
    index = pd.date_range("2024-01-01", periods=len(rows), freq="B")
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1000.0] * len(rows),
        },
        index=index,
    )


def base_rows(n: int = 40) -> list[tuple[float, float, float, float]]:
    """Candles neutros: fecham no meio do range."""
    return [(1000.0, 1050.0, 950.0, 1000.0)] * n


def test_ibs_mede_onde_o_candle_fechou():
    df = frame([(1000.0, 1100.0, 900.0, 900.0), (1000.0, 1100.0, 900.0, 1100.0)])
    values = ibs(df)
    assert values.iloc[0] == 0.0  # fechou na mínima
    assert values.iloc[1] == 1.0  # fechou na máxima


def test_ibs_neutro_quando_o_candle_nao_tem_range():
    df = frame([(1000.0, 1000.0, 1000.0, 1000.0)])
    assert ibs(df).iloc[0] == 0.5


def test_compra_quando_fecha_perto_da_minima():
    rows = base_rows() + [(1000.0, 1050.0, 950.0, 955.0)]  # IBS = 0.05
    signal = IbsStrategy({"entry_low": 0.15}).generate_signal("WIN", frame(rows))

    assert signal.type == SignalType.BUY
    assert signal.stop_loss < signal.entry_price < signal.take_profit


def test_vende_quando_fecha_perto_da_maxima():
    rows = base_rows() + [(1000.0, 1050.0, 950.0, 1045.0)]  # IBS = 0.95
    signal = IbsStrategy({"entry_high": 0.85}).generate_signal("WIN", frame(rows))
    assert signal.type == SignalType.SELL


def test_hold_no_meio_do_range():
    rows = base_rows() + [(1000.0, 1050.0, 950.0, 1000.0)]  # IBS = 0.5
    assert IbsStrategy().generate_signal("WIN", frame(rows)).type == SignalType.HOLD


def test_alvo_mais_perto_que_o_stop_por_padrao():
    rows = base_rows() + [(1000.0, 1050.0, 950.0, 955.0)]
    signal = IbsStrategy({"target_atr": 0.5, "stop_atr": 2.0}).generate_signal("WIN", frame(rows))
    ganho = signal.take_profit - signal.entry_price
    perda = signal.entry_price - signal.stop_loss
    # Alvo curto: acerta mais vezes, ganha menos por vez
    assert ganho < perda


def test_filtro_de_tendencia_veta_compra_abaixo_da_media():
    # Série em queda: o preço atual está abaixo da média de 20
    rows = [(1000.0 - i * 5, 1050.0 - i * 5, 950.0 - i * 5, 1000.0 - i * 5) for i in range(40)]
    rows.append((800.0, 850.0, 750.0, 755.0))  # IBS baixo, mas em queda
    signal = IbsStrategy({"entry_low": 0.15, "trend_filter": 20}).generate_signal("WIN", frame(rows))
    assert signal.type == SignalType.HOLD


def test_e_modo_swing():
    assert IbsStrategy().mode == "swing_trade"
