"""Testes do estudo de níveis (toque, repique, rompimento, placebo)."""

import pandas as pd

from src.bot.analysis.levels import (
    levels_for,
    session_frame,
    study_levels,
    two_proportion_z,
    walk_level,
)


def bars(rows: list[tuple[float, float, float, float]], day: str = "2024-01-02") -> pd.DataFrame:
    """Cada linha: (open, high, low, close), candles de 5 minutos."""
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [100.0] * len(rows),
        },
        index=pd.date_range(f"{day} 09:00", periods=len(rows), freq="5min"),
    )


# nível 100, banda 1, corrida 5: repique = voltar a 95, rompimento = 105


def test_repique_simples_vindo_de_baixo():
    data = bars([
        (90, 91, 89, 90),      # armado abaixo (high <= 95)
        (90, 99.5, 90, 99),    # toca a banda (99.5 >= 99)
        (99, 100, 94, 94.5),   # cai a 94 <= 95: repique
    ])
    assert walk_level(data, 100.0, band=1.0, race=5.0) == [("below", False)]


def test_rompimento_vindo_de_baixo():
    data = bars([
        (90, 91, 89, 90),
        (90, 99.5, 90, 99),
        (99, 106, 98, 105.5),  # atinge 105 >= 105: rompeu
    ])
    assert walk_level(data, 100.0, band=1.0, race=5.0) == [("below", True)]


def test_sequencia_dois_repiques_e_rompimento():
    data = bars([
        (90, 91, 89, 90),
        (90, 99.2, 90, 99),    # 1º toque
        (99, 99, 94, 94.5),    # repique
        (94, 99.8, 94, 99),    # 2º toque (rearmado: veio de 94 <= 95)
        (99, 99, 94.5, 94.8),  # repique
        (95, 106, 95, 105),    # 3º toque, rompe
    ])
    events = walk_level(data, 100.0, band=1.0, race=5.0)
    assert events == [("below", False), ("below", False), ("below", True)]


def test_toque_como_suporte_vindo_de_cima():
    data = bars([
        (110, 111, 105.5, 110),  # armado acima (low >= 105)
        (110, 110, 100.5, 101),  # toca a banda por cima
        (101, 106, 101, 105.5),  # sobe a 105 >= 105: repique
    ])
    assert walk_level(data, 100.0, band=1.0, race=5.0) == [("above", False)]


def test_barra_ambigua_decide_pelo_fechamento():
    data = bars([
        (90, 91, 89, 90),
        (90, 106, 94, 104),    # toca, atinge 105 E 95 na mesma barra; fecha acima
    ])
    assert walk_level(data, 100.0, band=1.0, race=5.0) == [("below", True)]

    data2 = bars([
        (90, 91, 89, 90),
        (90, 106, 94, 96),     # mesma barra ambígua, mas fecha abaixo do nível
    ])
    assert walk_level(data2, 100.0, band=1.0, race=5.0) == [("below", False)]


def test_sem_toque_sem_eventos():
    data = bars([(90, 91, 89, 90), (90, 93, 89, 92), (92, 94, 90, 93)])
    assert walk_level(data, 100.0, band=1.0, race=5.0) == []


def test_dia_termina_sem_resolucao_descarta_o_toque():
    data = bars([
        (90, 91, 89, 90),
        (90, 99.5, 90, 99),    # toca...
        (99, 101, 98, 100),    # ...e o dia acaba entre 95 e 105
    ])
    assert walk_level(data, 100.0, band=1.0, race=5.0) == []


def test_levels_for_matematica_do_pivo():
    prev = pd.Series({"high": 110.0, "low": 90.0, "close": 100.0})
    prev2 = pd.Series({"high": 120.0, "low": 80.0, "close": 95.0})
    levels = dict(levels_for(prev, prev2))

    assert levels["PP"] == 100.0
    assert levels["R1"] == 110.0          # 2*100 - 90
    assert levels["S1"] == 90.0           # 2*100 - 110
    assert levels["R2"] == 120.0          # 100 + 20
    assert levels["S2"] == 80.0
    assert levels["DBYH"] == 120.0


def test_levels_for_inclui_redondos():
    prev = pd.Series({"high": 110.0, "low": 90.0, "close": 100.0})
    prev2 = pd.Series({"high": 111.0, "low": 89.0, "close": 100.0})
    rounds = [price for name, price in levels_for(prev, prev2, round_step=50.0)
              if name == "RND"]
    # espaço varrido: [90-20, 110+20] = [70, 130] → 100 é o único múltiplo de 50
    assert rounds == [100.0]


def test_session_frame_agrega_ohlc():
    d1 = bars([(100, 105, 95, 102), (102, 110, 101, 108)], day="2024-01-02")
    d2 = bars([(108, 109, 100, 101)], day="2024-01-03")
    daily = session_frame(pd.concat([d1, d2]))

    assert len(daily) == 2
    assert daily.iloc[0]["high"] == 110
    assert daily.iloc[0]["low"] == 95
    assert daily.iloc[0]["close"] == 108
    assert daily.iloc[1]["bars"] == 1


def test_study_levels_gera_reais_e_placebos():
    days = []
    for i, day in enumerate(["2024-01-02", "2024-01-03", "2024-01-04"]):
        base = 100 + i
        rows = [(base, base + 10, base - 10, base + 2)] * 40
        days.append(bars(rows, day=day))
    touches = study_levels(pd.concat(days), min_bars=10)

    if not touches.empty:
        assert set(touches["placebo"].unique()) <= {True, False}
        assert (touches["index"] >= 1).all()


def test_two_proportion_z():
    z, p = two_proportion_z(60, 100, 50, 100)
    assert z > 0
    assert 0 < p < 1
    z_null, p_null = two_proportion_z(50, 100, 50, 100)
    assert z_null == 0.0
    assert p_null == 1.0


# ───────────────────────── Zonas dinâmicas ─────────────────────────

from src.bot.analysis.levels import walk_zone


def test_zona_so_vale_depois_de_nascer():
    # A zona 100 nasce na barra 3; o toque da barra 1 não conta
    data = bars([
        (99, 100.5, 98, 99),     # tocaria, mas a zona ainda não existe
        (99, 106, 99, 106),
        (106, 107, 105, 106),    # armado (low >= 105)
        (106, 107, 100.5, 101),  # toque por cima (low <= 101)
        (101, 106, 101, 105.5),  # sobe a 105: repique
    ])
    events = walk_zone(data, 100.0, band=1.0, race=5.0, start=1, side="above")
    assert events == [(3, False)]


def test_zona_rompida_encerra_o_rastreamento():
    data = bars([
        (106, 107, 105, 106),    # armado
        (106, 106, 100.5, 101),  # toque
        (101, 101, 94, 94.5),    # cai a 95: rompeu (low <= 95)
        (94, 107, 94, 106),      # nova aproximação NÃO conta mais
        (106, 106, 100.5, 101),
    ])
    events = walk_zone(data, 100.0, band=1.0, race=5.0, start=0, side="above")
    assert events == [(1, True)]


def test_zona_como_resistencia_por_baixo():
    data = bars([
        (94, 95, 93, 94),        # armado (high <= 95)
        (94, 99.5, 94, 99),      # toque por baixo
        (99, 100, 94, 94.5),     # cai a 95: repique
    ])
    events = walk_zone(data, 100.0, band=1.0, race=5.0, start=0, side="below")
    assert events == [(1, False)]
