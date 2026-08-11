"""Testes da resolução de contratos futuros da B3."""

from datetime import date

from src.bot.data.contracts import resolve_symbol, wdo_contract, win_contract, win_expiry


def test_win_expiry_quarta_mais_proxima_do_dia_15():
    # Agosto/2026: dia 15 é sábado; quarta mais próxima é 12/08
    assert win_expiry(2026, 8) == date(2026, 8, 12)
    # Abril/2026: dia 15 é quarta-feira
    assert win_expiry(2026, 4) == date(2026, 4, 15)


def test_win_contrato_vigente_antes_do_vencimento():
    # Antes de 12/08/2026 o contrato é o de agosto (Q)
    assert win_contract(date(2026, 8, 10)) == "WINQ26"


def test_win_rola_para_outubro_no_vencimento():
    # No dia do vencimento, o front passa a ser outubro (V)
    assert win_contract(date(2026, 8, 12)) == "WINV26"
    assert win_contract(date(2026, 9, 1)) == "WINV26"


def test_win_vira_o_ano():
    # Em dezembro, após o vencimento, o front é fevereiro do ano seguinte (G)
    assert win_contract(date(2026, 12, 20)) == "WING27"


def test_wdo_contrato_e_o_do_mes_seguinte():
    # Negociado em agosto/2026 → contrato de setembro (U)
    assert wdo_contract(date(2026, 8, 10)) == "WDOU26"


def test_wdo_vira_o_ano():
    # Negociado em dezembro/2026 → contrato de janeiro/2027 (F)
    assert wdo_contract(date(2026, 12, 15)) == "WDOF27"


def test_resolve_symbol():
    ref = date(2026, 8, 10)
    assert resolve_symbol("WIN", ref) == "WINQ26"
    assert resolve_symbol("wdo", ref) == "WDOU26"
    # Símbolo explícito passa direto
    assert resolve_symbol("WINV26", ref) == "WINV26"
    assert resolve_symbol("PETR4", ref) == "PETR4"
