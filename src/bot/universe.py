"""Universo de instrumentos para os testes de replicação.

A divisão por bloco existe por um motivo estatístico, não estético: os
13 instrumentos originais eram quase todos o mesmo mercado (Ibovespa
em roupas diferentes), o que dava só ~7 apostas efetivamente
independentes e deixava nosso melhor resultado com p = 0,082.

Os blocos abaixo são dirigidos por forças distintas — safra e clima,
política monetária, tecnologia americana, China, cripto. É isso que
aumenta o número de apostas independentes e, com ele, o poder do
teste de replicação.
"""

# Ibovespa e derivados — alta correlação interna, contam quase como um
ACOES_BR = ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "B3SA3", "WEGE3", "PRIO3",
            "MGLU3", "SUZB3", "RENT3", "RADL3", "TOTS3"]
INDICES_BR = ["WIN$N", "BOVA11", "SMAL11"]
CAMBIO = ["WDO$N"]

# Blocos genuinamente descorrelacionados do Ibovespa
JUROS = ["DI1F27", "DI1F28", "DI1F29"]
COMMODITIES = ["CCM$N", "BGI$N", "ICF$N"]          # milho, boi gordo, café
EXTERIOR = ["IVVB11", "NASD11", "XINA11"]          # S&P, Nasdaq, China
ALTERNATIVOS = ["GOLD11", "HASH11"]                 # ouro, cripto

BLOCKS = {
    "ações BR": ACOES_BR,
    "índices BR": INDICES_BR,
    "câmbio": CAMBIO,
    "juros": JUROS,
    "commodities": COMMODITIES,
    "exterior": EXTERIOR,
    "alternativos": ALTERNATIVOS,
}

EXPANDED = [s for symbols in BLOCKS.values() for s in symbols]

# Universo original, mantido para comparação
ORIGINAL = ["WIN$N", "WDO$N", "BOVA11", "SMAL11", "IVVB11",
            "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "B3SA3", "WEGE3", "PRIO3"]

# Valor financeiro de 1 unidade de variação do preço cotado.
#
# AUDITADO em 14/08/2026 contra a especificação dos contratos no MT5
# da XP (scripts/verify_contracts.py): valor do ponto = tick_value /
# tick_size. WIN, WDO, CCM e BGI confirmados.
#
# Ações e ETFs: cotados em R$ por papel, então 1,0.
# WIN: 1 ponto de índice = R$ 0,20 · WDO: 1 ponto = R$ 10,00.
# CCM (milho): contrato de 450 sacas, cotado em R$/saca.
# BGI (boi gordo): 330 arrobas, cotado em R$/arroba.
# ICF (café): 100 sacas em USD/saca; MT5 confirma USD 100/ponto, aqui
#   convertido a ~R$ 5,40 (a exposição cambial embutida não é modelada).
# DI1: cotado em TAXA (%). O valor de 1 p.p. NÃO é constante — segue a
#   duration: dPU/dr = 1000·n/(1+r)^(n+1), e no F27 foi de R$ 2.986 a
#   R$ 325 ao longo do histórico. Os valores abaixo são a MÉDIA do
#   período de cada contrato (calculada sobre os candles reais).
#   Aproximação declarada: o exato exigiria backtest em espaço de PU.
POINT_VALUE = {
    "WIN$N": 0.20,
    "WDO$N": 10.00,
    "CCM$N": 450.0,
    "BGI$N": 330.0,
    "ICF$N": 540.0,
    "DI1F27": 1_751.0,
    "DI1F28": 2_140.0,
    "DI1F29": 2_415.0,
}

# Custo em caixa de UMA unidade da posição. Ações e ETFs são pagos
# integralmente (usa o preço, informado em tempo de execução). Futuros
# exigem só a margem — AUDITADA em 14/08/2026 via order_calc_margin na
# XP demo (margem de carrego; a intraday é menor).
MARGIN = {
    "WIN$N": 7_400.0,
    "WDO$N": 7_000.0,
    "CCM$N": 1_600.0,
    "BGI$N": 4_600.0,
    "ICF$N": 11_200.0,
    "DI1F27": 750.0,
    "DI1F28": 1_900.0,
    "DI1F29": 3_100.0,
}


def unit_cost_of(symbol: str) -> float | None:
    """Caixa consumido por unidade. None = usar o preço (mercado à vista)."""
    return MARGIN.get(symbol)


# ─────────────────────── Carteira de futuros ───────────────────────
#
# Futuros exigem apenas margem, não o valor cheio — é a alavancagem
# legítima e barata que torna o trend following eficiente em capital.
# Não é coincidência que fundos de managed futures operem futuros e não
# ações à vista.
#
# AUDITORIA de 14/08/2026 (scripts/verify_contracts.py, XP demo):
# valores de ponto confirmados pelo MT5, exceto DI1 (média da duration,
# ver POINT_VALUE acima) e ICF (USD, convertido). Margens medidas via
# order_calc_margin — margem de CARREGO, e a XP cobra bem mais que as
# estimativas antigas (WIN: 7.377 contra os 2.000 estimados).
# WSP$N: o cálculo de margem não retornou na demo; estimativa mantida.

FUTUROS_BLOCKS = {
    "índice BR": ["WIN$N", "IND$N"],
    "exterior": ["WSP$N"],                       # Micro S&P 500
    "juros BR": ["DI1F27", "DI1F29", "DI1F31", "DI1F33"],
    "juros US": ["T10$N"],                       # T-Note 10 anos
    "câmbio": ["WDO$N", "DOL$N"],
    "commodities": ["CCM$N", "BGI$N", "ICF$N"],
}
FUTUROS = [s for symbols in FUTUROS_BLOCKS.values() for s in symbols]

FUT_POINT_VALUE = {
    "WIN$N": 0.20, "IND$N": 1.00,        # índice: mini e cheio (MT5 ✓)
    "WDO$N": 10.00, "DOL$N": 50.00,      # dólar: mini e cheio (MT5 ✓)
    "WSP$N": 2.50,                       # micro S&P (MT5 ✓)
    "T10$N": 1_000.0,                    # T-Note (MT5 ✓ — era 10, erro de 100x)
    "DI1F27": 1_751.0, "DI1F29": 2_415.0,  # média da duration no período
    "DI1F31": 2_707.0, "DI1F33": 2_765.0,  # (dPU/dr sobre os candles reais)
    "CCM$N": 450.0,                      # 450 sacas (MT5 ✓)
    "BGI$N": 330.0,                      # 330 arrobas (MT5 ✓)
    "ICF$N": 540.0,                      # USD 100/ponto × ~R$ 5,40
}

FUT_MARGIN = {
    "WIN$N": 7_400.0, "IND$N": 36_900.0,
    "WDO$N": 7_000.0, "DOL$N": 35_200.0,
    "WSP$N": 3_000.0,                    # não retornou na demo; estimativa
    "T10$N": 31_600.0,
    "DI1F27": 750.0, "DI1F29": 3_100.0,
    "DI1F31": 4_100.0, "DI1F33": 4_400.0,
    "CCM$N": 1_600.0, "BGI$N": 4_600.0, "ICF$N": 11_200.0,
}


def fut_block_of(symbol: str) -> str:
    for name, symbols in FUTUROS_BLOCKS.items():
        if symbol in symbols:
            return name
    return "outro"


def block_of(symbol: str) -> str:
    for name, symbols in BLOCKS.items():
        if symbol in symbols:
            return name
    return "outro"
