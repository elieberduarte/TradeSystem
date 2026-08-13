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
# Ações e ETFs: cotados em R$ por papel, então 1,0.
# WIN: 1 ponto de índice = R$ 0,20 · WDO: 1 ponto = R$ 10,00.
# CCM (milho): contrato de 450 sacas, cotado em R$/saca.
# BGI (boi gordo): 330 arrobas, cotado em R$/arroba.
# ICF (café): 100 sacas, cotado em USD/saca — convertido por ~R$ 5,40.
# DI1: cotado em TAXA (%). O valor de 1 ponto percentual depende da
#   duration do vencimento; aqui usamos aproximações por contrato
#   (~R$ 800 por ponto percentual em papéis de 1 a 3 anos). É estimativa,
#   não valor exato — DI exige cálculo de PU para precisão.
POINT_VALUE = {
    "WIN$N": 0.20,
    "WDO$N": 10.00,
    "CCM$N": 450.0,
    "BGI$N": 330.0,
    "ICF$N": 540.0,
    "DI1F27": 800.0,
    "DI1F28": 800.0,
    "DI1F29": 800.0,
}

# Custo em caixa de UMA unidade da posição. Ações e ETFs são pagos
# integralmente (usa o preço, informado em tempo de execução). Futuros
# exigem só a margem — valores aproximados de swing (margem cheia).
MARGIN = {
    "WIN$N": 2_000.0,
    "WDO$N": 5_000.0,
    "CCM$N": 3_000.0,
    "BGI$N": 4_000.0,
    "ICF$N": 6_000.0,
    "DI1F27": 1_500.0,
    "DI1F28": 2_000.0,
    "DI1F29": 2_500.0,
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
# ⚠️ Os valores de ponto e margem abaixo são ESTIMATIVAS de ordem de
# grandeza para o teste. Antes de operar, confirmar cada um com a
# especificação da B3 e a margem que a corretora efetivamente exige.

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
    "WIN$N": 0.20, "IND$N": 1.00,        # índice: mini e cheio
    "WDO$N": 10.00, "DOL$N": 50.00,      # dólar: mini e cheio
    "WSP$N": 2.50,                       # micro S&P, estimativa
    "T10$N": 10.00,                      # T-Note, estimativa
    "DI1F27": 800.0, "DI1F29": 800.0,    # por ponto percentual de taxa
    "DI1F31": 900.0, "DI1F33": 950.0,    # duration maior, valor maior
    "CCM$N": 450.0,                      # 450 sacas
    "BGI$N": 330.0,                      # 330 arrobas
    "ICF$N": 540.0,                      # 100 sacas em USD convertidas
}

FUT_MARGIN = {
    "WIN$N": 2_000.0, "IND$N": 12_000.0,
    "WDO$N": 5_000.0, "DOL$N": 25_000.0,
    "WSP$N": 3_000.0, "T10$N": 3_000.0,
    "DI1F27": 800.0, "DI1F29": 1_200.0,
    "DI1F31": 1_800.0, "DI1F33": 2_500.0,
    "CCM$N": 2_500.0, "BGI$N": 3_500.0, "ICF$N": 6_000.0,
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
