"""Testes do zigzag causal e dos eventos de estrutura."""

import pandas as pd

from src.bot.analysis.swings import Pivot, structure_events, swing_pivots


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_topo_confirmado_so_apos_o_limiar():
    #        0    1     2     3     4
    closes = series([100, 110, 108, 106, 104])
    pivots = swing_pivots(closes, threshold=5.0)
    # A série começa num extremo real: o fundo da barra 0 é confirmado
    # pela subida de 10; depois vem o topo de 110
    assert [(p.kind, p.price) for p in pivots] == [("fundo", 100), ("topo", 110)]
    top = pivots[1]
    assert top.index == 1           # o extremo foi na barra 1...
    assert top.confirm_index == 4   # ...mas só ficou conhecível na 4 (110-104≥5)


def test_zigzag_alternado():
    closes = series([100, 110, 104, 112, 106, 114, 108])
    pivots = swing_pivots(closes, threshold=5.0)
    kinds = [p.kind for p in pivots]
    prices = [p.price for p in pivots]
    assert kinds == ["fundo", "topo", "fundo", "topo", "fundo", "topo"]
    assert prices == [100, 110, 104, 112, 106, 114]


def test_sem_movimento_sem_pivos():
    closes = series([100, 101, 100, 101, 100])
    assert swing_pivots(closes, threshold=5.0) == []


def test_confirmacao_nunca_antes_do_extremo():
    closes = series([100, 108, 116, 110, 104, 112, 120])
    for pivot in swing_pivots(closes, threshold=6.0):
        assert pivot.confirm_index > pivot.index


def test_estrutura_de_alta_exige_hh_e_hl():
    # fundos 100 → 104 (HL) e topos 110 → 112 (HH): a estrutura fica
    # completa quando o topo 112 é confirmado (barra 4); cada pivô
    # seguinte que a mantém soma uma perna
    closes = series([100, 110, 104, 112, 106, 114, 108])
    events = structure_events(swing_pivots(closes, threshold=5.0))
    assert [e.leg for e in events] == [1, 2, 3]
    first = events[0]
    assert first.direction == 1
    assert first.confirm_index == 4   # 112 confirmado quando fecha 106


def test_estrutura_quebrada_zera_a_contagem():
    # alta (HH+HL), depois fundo abaixo do anterior quebra a estrutura
    closes = series([100, 110, 104, 112, 106, 114, 98, 108, 100, 110])
    events = structure_events(swing_pivots(closes, threshold=5.0))
    legs = [e.leg for e in events]
    # depois da quebra, qualquer estrutura nova recomeça em 1
    assert 1 in legs
    assert all(leg == 1 or legs[i - 1] == leg - 1 for i, leg in enumerate(legs))


def test_estrutura_de_baixa_espelhada():
    closes = series([114, 104, 110, 100, 106, 96, 102])
    events = structure_events(swing_pivots(closes, threshold=5.0))
    assert events
    assert events[0].direction == -1
