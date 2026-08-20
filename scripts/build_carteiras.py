"""Monta as carteiras teóricas por perfil de risco (R$ 20 mil cada).

Critérios clássicos, aplicados sobre o que já medimos: fundamentos
(painel de 115 ações líquidas com balanço da CVM) + risco (volatilidade,
beta contra o BOVA11, drawdown, 2 anos de janela).

  BAIXO RISCO — "dormir tranquilo"
    blue chip (liquidez ≥ R$ 50 mi/dia), volatilidade no terço inferior,
    beta < 1, paga dividendo (DY ≥ mediana), lucrativa em todos os
    exercícios, dívida ≤ mediana, ROE de 5 anos ≥ mediana.

  RISCO MÉDIO — "qualidade a preço razoável"
    liquidez ≥ R$ 20 mi/dia, ROE 5a ≥ mediana, todos os anos lucrativa,
    dívida ≤ 1,5× mediana, P/L positivo abaixo do 3º quartil,
    volatilidade até a mediana + 1/2 desvio.

  ALTO RISCO — "crescimento lucrativo, small caps"
    liquidez ≥ R$ 5 mi/dia, valor de mercado ≤ R$ 10 bi, CAGR de receita
    ≥ 15%, lucrativa (o filtro que separa small growth de loteria — o
    canto historicamente pior do mercado), sem exigir dividendo.

  REFERÊNCIA — BOVA11 (o índice) e o CDI, para a comparação ser honesta.

Uso: python scripts/build_carteiras.py [aporte=20000]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.data.history import HistoryStore
from src.bot.portfolio.carteiras import metricas_de_risco, montar, salvar

ROOT = Path(__file__).resolve().parents[1]
JANELA = 504          # ~2 anos de pregões para as métricas de risco


def carregar_universo(store: HistoryStore) -> pd.DataFrame:
    fund = json.loads((ROOT / "web" / "fundamentals.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(fund["papeis"])
    indice = store.load("BOVA11", "1d")
    indice_close = indice["close"] if indice is not None else None

    linhas = []
    for _, r in frame.iterrows():
        daily = store.load(r["ticker"], "1d")
        if daily is None or len(daily) < 300:
            continue
        risco = metricas_de_risco(daily, indice_close, JANELA)
        if not risco:
            continue
        linhas.append({**r.to_dict(), **risco, "preco_hoje": float(daily["close"].iloc[-1])})
    return pd.DataFrame(linhas)


def main() -> None:
    aporte = float(sys.argv[1]) if len(sys.argv) > 1 else 20_000.0
    store = HistoryStore(ROOT / "data")
    u = carregar_universo(store)
    print(f"universo: {len(u)} ações com fundamento + risco\n")

    # ROE fora de [-100%, +200%] denuncia patrimônio líquido distorcido
    # (negativo ou quase zero) — o número existe, mas não mede rentabilidade.
    absurdos = u[(u["roe_medio"] < -1.0) | (u["roe_medio"] > 2.0)]
    if len(absurdos):
        print("descartados por ROE distorcido (PL negativo/ínfimo): " +
              ", ".join(f"{r.ticker} ({r.roe_medio:.0%})" for r in absurdos.itertuples()))
    u = u[u["roe_medio"].between(-1.0, 2.0)]

    med = {c: u[c].median() for c in ("pl", "roe_medio", "divida_pl", "dy", "vol_anual")}
    q75_pl = u[u["pl"] > 0]["pl"].quantile(0.75)
    t33_vol, t66_vol = u["vol_anual"].quantile([1/3, 2/3])
    print(f"medianas do universo · P/L {med['pl']:.1f} · ROE 5a {med['roe_medio']:.1%} · "
          f"dív/PL {med['divida_pl']:.2f} · DY {med['dy']:.1%} · vol {med['vol_anual']:.1%}")
    print(f"volatilidade: terço inferior ≤ {t33_vol:.1%} · superior ≥ {t66_vol:.1%}\n")

    def sel(frame: pd.DataFrame, ordem: str, n: int, asc: bool = True) -> pd.DataFrame:
        """Ordena, tira ON/PN do MESMO emissor e corta em n.

        PETR3 e PETR4 são a mesma empresa: mantê-las juntas concentraria
        2/7 do capital num emissor só, com cara de diversificação. O
        emissor é o prefixo de 4 letras; fica a mais líquida.
        """
        frame = frame.sort_values(ordem, ascending=asc).copy()
        frame["emissor"] = frame["ticker"].str[:4]
        frame = (frame.sort_values("volume_mediano_mi", ascending=False)
                      .drop_duplicates(subset="emissor", keep="first")
                      .sort_values(ordem, ascending=asc))
        return frame.head(n)

    # ── BAIXO RISCO ──
    baixo = u[
        (u["volume_mediano_mi"] >= 50) & (u["vol_anual"] <= t33_vol) & (u["beta"] < 1.0)
        & (u["dy"] >= med["dy"]) & (u["roe_medio"] >= med["roe_medio"])
        & (u["divida_pl"] <= med["divida_pl"])
        & (u["anos_lucrativos"] == u["anos_observados"])
    ]
    baixo = sel(baixo, "vol_anual", 8)

    # ── RISCO MÉDIO ──
    medio = u[
        (u["volume_mediano_mi"] >= 20) & (u["roe_medio"] >= med["roe_medio"])
        & (u["anos_lucrativos"] == u["anos_observados"])
        & (u["divida_pl"] <= 1.5 * med["divida_pl"])
        & (u["pl"] > 0) & (u["pl"] <= q75_pl)
        & (u["vol_anual"] <= u["vol_anual"].median() + 0.5 * u["vol_anual"].std())
        & (~u["ticker"].isin(baixo["ticker"]))
    ]
    medio = sel(medio, "roe_medio", 10, asc=False)

    # ── ALTO RISCO ──
    alto = u[
        (u["volume_mediano_mi"] >= 5) & (u["valor_mercado_bi"] <= 10)
        & (u["cresc_receita"].notna()) & (u["cagr_receita"] >= 0.15)
        & (u["anos_lucrativos"] >= u["anos_observados"] - 1) & (u["roe_medio"] > 0.05)
        & (~u["ticker"].isin(baixo["ticker"])) & (~u["ticker"].isin(medio["ticker"]))
    ]
    alto = sel(alto, "cagr_receita", 10, asc=False)

    precos = dict(zip(u["ticker"], u["preco_hoje"]))
    bova = store.load("BOVA11", "1d")
    precos["BOVA11"] = float(bova["close"].iloc[-1])

    def motivos_de(frame: pd.DataFrame, tipo: str) -> dict:
        out = {}
        for _, r in frame.iterrows():
            if tipo == "baixo":
                out[r["ticker"]] = (f"vol {r['vol_anual']:.0%} a.a., beta {r['beta']:.2f}, "
                                    f"DY {r['dy']:.1%}, ROE 5a {r['roe_medio']:.0%}, "
                                    f"dív/PL {r['divida_pl']:.2f}")
            elif tipo == "medio":
                out[r["ticker"]] = (f"ROE 5a {r['roe_medio']:.0%}, P/L {r['pl']:.1f}, "
                                    f"vol {r['vol_anual']:.0%}, {int(r['anos_lucrativos'])}/"
                                    f"{int(r['anos_observados'])} anos lucrativa")
            else:
                out[r["ticker"]] = (f"CAGR receita {r['cagr_receita']:.0%}, ROE 5a "
                                    f"{r['roe_medio']:.0%}, VM R$ {r['valor_mercado_bi']:.1f} bi, "
                                    f"vol {r['vol_anual']:.0%}")
        return out

    carteiras = [
        montar("Defensiva", "baixo",
               "Blue chips de baixa volatilidade que pagam dividendo e lucram todo ano",
               ["liquidez ≥ R$ 50 mi/dia", "volatilidade no terço inferior", "beta < 1",
                "DY ≥ mediana", "ROE 5a ≥ mediana", "dívida ≤ mediana", "lucrativa em todos os anos"],
               list(baixo["ticker"]), precos, aporte, motivos_de(baixo, "baixo")),
        montar("Qualidade", "medio",
               "Empresas rentáveis e consistentes a múltiplo razoável",
               ["liquidez ≥ R$ 20 mi/dia", "ROE 5a ≥ mediana", "lucrativa em todos os anos",
                "dívida ≤ 1,5× mediana", "P/L positivo ≤ 3º quartil", "volatilidade moderada"],
               list(medio["ticker"]), precos, aporte, motivos_de(medio, "medio")),
        montar("Crescimento", "alto",
               "Small caps que crescem receita COM lucro — não crescimento a qualquer preço",
               ["liquidez ≥ R$ 5 mi/dia", "valor de mercado ≤ R$ 10 bi", "CAGR de receita ≥ 15%",
                "lucrativa (o filtro que separa de loteria)", "sem exigir dividendo"],
               list(alto["ticker"]), precos, aporte, motivos_de(alto, "alto")),
        montar("Índice (BOVA11)", "referencia",
               "O mercado inteiro num papel — a régua que qualquer carteira precisa bater",
               ["ETF do Ibovespa", "peso único"],
               ["BOVA11"], precos, aporte, {"BOVA11": "benchmark de renda variável"}),
    ]

    for c, frame in zip(carteiras[:3], (baixo, medio, alto)):
        print(f"── {c.nome} ({c.perfil} risco) · {len(c.posicoes)} papéis · "
              f"caixa R$ {c.caixa:,.2f} ──")
        print(f"{'papel':<8} {'qtd':>5} {'preço':>8} {'valor':>10} {'vol':>6} {'beta':>6} "
              f"{'DY':>6} {'P/L':>6} {'ROE 5a':>7}")
        for pos in c.posicoes:
            r = frame[frame["ticker"] == pos.ticker].iloc[0]
            pl = f"{r['pl']:.1f}" if pd.notna(r["pl"]) else "—"
            dy = f"{r['dy']:.1%}" if pd.notna(r["dy"]) else "—"
            roe = f"{r['roe_medio']:.0%}" if pd.notna(r["roe_medio"]) else "—"
            print(f"{pos.ticker:<8} {pos.quantidade:>5} {pos.preco_entrada:>8.2f} "
                  f"{pos.valor:>10,.2f} {r['vol_anual']:>5.0%} {r['beta']:>6.2f} "
                  f"{dy:>6} {pl:>6} {roe:>7}")
        vol_media = frame["vol_anual"].mean()
        print(f"  volatilidade média da carteira (papéis): {vol_media:.1%} a.a. · "
              f"beta médio {frame['beta'].mean():.2f}\n")

    out = ROOT / "data" / "carteiras.json"
    salvar(carteiras, out)
    print(f"Carteiras registradas em {out}")
    print("Acompanhe com: python scripts/track_carteiras.py")


if __name__ == "__main__":
    main()
