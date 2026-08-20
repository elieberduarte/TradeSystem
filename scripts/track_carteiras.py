"""Acompanha as carteiras teóricas: valor, retorno e risco realizado.

Roda junto com o ciclo diário do bot. Registra um ponto por dia em
data/carteiras_historico.parquet e escreve web/carteiras.json para o
painel — assim a comparação entre perfis vira série, não foto.

⚠️ Os preços do MT5 são AJUSTADOS por proventos: o retorno medido é o
RETORNO TOTAL (valorização + dividendos e JCP reinvestidos). A coluna
"proventos estimados" usa o DY conhecido de cada papel só para mostrar
QUANTO do retorno veio de distribuição — é estimativa, não caixa.

Uso: python scripts/track_carteiras.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.data.history import HistoryStore
from src.bot.portfolio.carteiras import avaliar, carregar

ROOT = Path(__file__).resolve().parents[1]
CDI_ANUAL = 0.1415


def serie_da_carteira(carteira, store: HistoryStore) -> pd.Series:
    """Valor da carteira dia a dia desde a montagem (para risco realizado)."""
    partes = []
    for pos in carteira.posicoes:
        daily = store.load(pos.ticker, "1d")
        if daily is None:
            continue
        serie = daily["close"][daily.index >= carteira.data_inicio] * pos.quantidade
        partes.append(serie.rename(pos.ticker))
    if not partes:
        return pd.Series(dtype=float)
    return pd.concat(partes, axis=1).ffill().sum(axis=1) + carteira.caixa


def main() -> None:
    store = HistoryStore(ROOT / "data")
    carteiras = carregar(ROOT / "data" / "carteiras.json")
    fund = {p["ticker"]: p for p in json.loads(
        (ROOT / "web" / "fundamentals.json").read_text(encoding="utf-8"))["papeis"]}

    precos_hoje = {}
    for c in carteiras:
        for pos in c.posicoes:
            daily = store.load(pos.ticker, "1d")
            if daily is not None:
                precos_hoje[pos.ticker] = float(daily["close"].iloc[-1])

    payload, historico = [], []
    print(f"{'carteira':<18} {'valor':>11} {'retorno':>9} {'vs CDI':>8} "
          f"{'vol real':>9} {'pior queda':>11} {'dias':>5}")
    print("-" * 78)
    for c in carteiras:
        resumo = avaliar(c, precos_hoje, CDI_ANUAL)
        serie = serie_da_carteira(c, store)
        vol = pior = float("nan")
        if len(serie) > 5:
            ret = serie.pct_change().dropna()
            vol = float(ret.std() * np.sqrt(252))
            pior = float((serie / serie.cummax() - 1).min())

        # Quanto do retorno esperado vem de provento (estimativa pelo DY)
        dy_carteira = np.mean([fund.get(p.ticker, {}).get("dy") or 0.0
                               for p in c.posicoes]) if c.posicoes else 0.0
        resumo.update({
            "vol_realizada": None if np.isnan(vol) else round(vol, 4),
            "pior_queda": None if np.isnan(pior) else round(pior, 4),
            "dy_medio_estimado": round(float(dy_carteira), 4),
            "descricao": c.descricao, "criterios": c.criterios,
        })
        payload.append(resumo)
        historico.append({"data": datetime.now().strftime("%Y-%m-%d"),
                          "carteira": c.nome, "valor": resumo["valor_atual"],
                          "retorno_pct": resumo["retorno_pct"]})
        print(f"{c.nome:<18} {resumo['valor_atual']:>11,.2f} {resumo['retorno_pct']:>8.2%} "
              f"{resumo['vs_cdi_pp']:>+7.2%} "
              f"{'—' if np.isnan(vol) else f'{vol:>8.1%}'} "
              f"{'—' if np.isnan(pior) else f'{pior:>10.1%}'} {resumo['dias']:>5}")

    # histórico incremental (uma linha por carteira por dia)
    path = ROOT / "data" / "carteiras_historico.parquet"
    fresh = pd.DataFrame(historico)
    if path.exists():
        fresh = pd.concat([pd.read_parquet(path), fresh], ignore_index=True)
        fresh = fresh.drop_duplicates(subset=["data", "carteira"], keep="last")
    fresh.sort_values(["data", "carteira"]).to_parquet(path)

    out = ROOT / "web" / "carteiras.json"
    out.write_text(json.dumps({
        "atualizado": datetime.now().isoformat(timespec="seconds"),
        "cdi_anual": CDI_ANUAL, "carteiras": payload,
        "historico": fresh.to_dict("records"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(fresh)} pontos no histórico · exportado para {out}")


if __name__ == "__main__":
    main()
