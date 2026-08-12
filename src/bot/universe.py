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

# Valor financeiro de 1 ponto. Para os instrumentos novos usamos 1,0:
# o teste de replicação olha SINAL e consistência (Sharpe por trade),
# que são invariantes a escala — não a comparação de PnL em reais.
POINT_VALUE = {"WIN$N": 0.20, "WDO$N": 10.00}


def block_of(symbol: str) -> str:
    for name, symbols in BLOCKS.items():
        if symbol in symbols:
            return name
    return "outro"
