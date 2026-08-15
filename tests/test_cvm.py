"""Testes das partes puras do coletor CVM (sem rede)."""

import pandas as pd
import pytest

from src.bot.data.cvm import _find_column


def test_find_column_acha_por_substring():
    frame = pd.DataFrame(columns=["CNPJ_Companhia", "Codigo_Negociacao", "Assunto"])
    assert _find_column(frame, "cnpj") == "CNPJ_Companhia"
    assert _find_column(frame, "codigo_negociacao") == "Codigo_Negociacao"


def test_find_column_tenta_alternativas_em_ordem():
    frame = pd.DataFrame(columns=["Denominacao_Social", "Outra"])
    assert _find_column(frame, "nome_empresarial", "denominacao") == "Denominacao_Social"


def test_find_column_explode_quando_nao_ha():
    frame = pd.DataFrame(columns=["A", "B"])
    with pytest.raises(KeyError):
        _find_column(frame, "cnpj")
