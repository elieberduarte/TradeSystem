"""Testes das partes puras do gravador de fluxo (sem MT5)."""

import numpy as np
import pandas as pd

from src.bot.data.flow_recorder import book_imbalance, cumulative_delta, lee_ready


def test_book_imbalance_extremos_e_equilibrio():
    assert book_imbalance(np.array([100.0]), np.array([0.0]), 1) == 1.0
    assert book_imbalance(np.array([0.0]), np.array([100.0]), 1) == -1.0
    assert book_imbalance(np.array([50.0, 50.0]), np.array([50.0, 50.0]), 2) == 0.0
    assert book_imbalance(np.array([]), np.array([]), 3) == 0.0


def test_book_imbalance_respeita_os_niveis():
    bids = np.array([10.0, 10.0, 1000.0])       # o peso está no 3º nível
    asks = np.array([10.0, 10.0, 10.0])
    assert book_imbalance(bids, asks, 2) == 0.0
    assert book_imbalance(bids, asks, 3) > 0.9


def test_lee_ready_quote_rule():
    bid = np.array([100.0, 100.0, 100.0])
    ask = np.array([101.0, 101.0, 101.0])
    last = np.array([101.0, 100.0, 101.0])       # no ask, no bid, no ask
    assert lee_ready(last, bid, ask).tolist() == [1, -1, 1]


def test_lee_ready_tick_test_no_meio():
    bid = np.array([100.0] * 4)
    ask = np.array([102.0] * 4)
    # 101 é o meio exato: sem lado pela cotação; decide pelo tick anterior
    last = np.array([100.0, 101.0, 101.0, 100.5])
    side = lee_ready(last, bid, ask)
    assert side[0] == -1                       # no bid
    assert side[1] == 1                        # subiu de 100 → 101: compra
    assert side[2] == 1                        # mesmo preço no meio: herda
    assert side[3] == -1                       # abaixo do meio


def test_lee_ready_herda_lado_anterior_entre_chamadas():
    bid = np.array([100.0]); ask = np.array([102.0]); last = np.array([101.0])
    assert lee_ready(last, bid, ask, prev_side=-1)[0] == -1
    assert lee_ready(last, bid, ask, prev_side=1)[0] == 1


def test_cumulative_delta():
    ticks = pd.DataFrame({"volume": [10.0, 5.0, 20.0], "side": [1, -1, 1]})
    assert cumulative_delta(ticks).tolist() == [10.0, 5.0, 25.0]


# ─────────────── Estados do book (leilão, cruzamento, contínuo) ───────────────

from src.bot.data.flow_recorder import classify_book


def test_book_continuo():
    assert classify_book(169_760.0, 169_765.0) == "continuo"   # 1 tick de spread
    assert classify_book(5_217.0, 5_217.5) == "continuo"


def test_book_de_leilao_e_cruzado_por_muito():
    # O caso real de 18/08: ordens nos limites do túnel de ±10%
    assert classify_book(186_845.0, 152_880.0) == "leilao"
    assert classify_book(5_500.0, 5_000.0) == "leilao"


def test_cruzamento_momentaneo_nao_e_leilao():
    # bid 1 tick acima do ask: leitura no meio de uma atualização
    assert classify_book(169_810.0, 169_805.0) == "cruzado"
    assert classify_book(169_760.0, 169_760.0) == "cruzado"    # iguais


def test_limite_de_cruzamento_e_relativo_ao_preco():
    # 0,5% de 170.000 = 850 pontos: abaixo disso é ruído de leitura
    assert classify_book(170_400.0, 170_000.0) == "cruzado"
    assert classify_book(171_000.0, 170_000.0) == "leilao"


def test_cvd_ignora_leilao():
    """Volume de leilão é acumulado, não negócio — não pode entrar no CVD."""
    ticks = pd.DataFrame({
        "volume": [20_770.0, 20_770.0, 10.0, 5.0],
        "side": [1, 1, 1, -1],
        "estado": ["leilao", "leilao", "continuo", "continuo"],
    })
    assert cumulative_delta(ticks).tolist() == [10.0, 5.0]


def test_cvd_sem_coluna_estado_usa_tudo():
    """Compatível com o acervo antigo (antes da classificação)."""
    ticks = pd.DataFrame({"volume": [10.0, 5.0], "side": [1, -1]})
    assert cumulative_delta(ticks).tolist() == [10.0, 5.0]
