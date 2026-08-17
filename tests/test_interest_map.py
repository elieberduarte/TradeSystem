"""Testes do mapa de interesse (pivôs + redondos + 50%)."""

import numpy as np
import pandas as pd

from src.bot.analysis.interest_map import build_map, nearest, round_levels, round_step_for


def frame(closes):
    values = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": values, "high": values + 1, "low": values - 1, "close": values,
        "volume": np.ones(len(values)),
    }, index=pd.date_range("2024-01-01 09:00", periods=len(values), freq="5min"))


def test_round_step_por_simbolo():
    assert round_step_for("WIN$N") == 1_000.0
    assert round_step_for("WINV26") == 1_000.0
    assert round_step_for("WDO$N") == 10.0
    assert round_step_for("PETR4") == 0.0


def test_round_levels_cobre_a_vizinhanca():
    levels = round_levels(171_435.0, 1_000.0, span=1_500.0)
    prices = [l.price for l in levels]
    assert 170_000.0 in prices and 171_000.0 in prices and 172_000.0 in prices
    assert all(l.kind == "redondo" for l in levels)


def test_build_map_detecta_topo_fundo_e_meio():
    # sobe 100→110, cai a 100 (topo em 110), sobe a 108 (fundo em 100)
    closes = [100, 103, 106, 110, 107, 104, 100, 103, 106, 108]
    levels, pivots = build_map(frame(closes), "WIN$N", threshold=5.0, round_span=0)
    kinds = {(l.kind, l.price) for l in levels}
    assert ("topo", 110.0) in kinds
    assert ("fundo", 100.0) in kinds
    assert ("meio", 105.0) in kinds            # 50% da perna 110→100
    # nenhum pivô "conhecível" no futuro
    assert all(l.known_at <= len(closes) - 1 for l in levels)


def test_nearest_ordena_por_distancia():
    levels, _ = build_map(frame([100, 103, 106, 110, 107, 104, 100, 103, 106, 108]),
                          "WIN$N", threshold=5.0, round_span=0)
    ranked = nearest(levels, price=106.0, n=2)
    assert abs(ranked[0][1]) <= abs(ranked[1][1])
