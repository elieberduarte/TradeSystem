"""Como as três carteiras teriam se comportado nos últimos anos.

⚠️ VIÉS DECLARADO — leia antes de olhar o retorno. Os papéis foram
escolhidos com dados de HOJE: fundamentos atuais, volatilidade dos
últimos 2 anos, empresas que existem e são líquidas agora. Aplicar
essa seleção ao passado é *look-ahead* e *viés de sobrevivência* ao
mesmo tempo — as empresas que quebraram no caminho nunca entram, e as
que hoje têm ROE alto tendem a ser justamente as que subiram. O
RETORNO desta simulação é, portanto, otimista por construção e NÃO é
previsão de nada.

O que a simulação mede com honestidade é o **comportamento de risco**:
volatilidade, profundidade de queda, correlação entre os perfis e como
cada um reage nos meses ruins. Essas propriedades dependem da natureza
dos papéis (blue chip vs small cap), não de quais especificamente
foram escolhidos — e é para isso que a tabela serve.

A medição limpa é a que começou hoje, em track_carteiras.py.

Uso: python scripts/carteiras_retroativo.py [anos=3]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.data.history import HistoryStore
from src.bot.portfolio.carteiras import carregar

ROOT = Path(__file__).resolve().parents[1]
CDI_ANUAL = 0.1415


def curva(carteira, store: HistoryStore, inicio: pd.Timestamp) -> pd.Series:
    """Curva de uma carteira com pesos iguais, rebalanceada só na entrada."""
    partes = []
    for pos in carteira.posicoes:
        daily = store.load(pos.ticker, "1d")
        if daily is None:
            continue
        serie = daily["close"][daily.index >= inicio]
        if len(serie) < 30:
            continue
        partes.append((serie / serie.iloc[0]).rename(pos.ticker))
    if not partes:
        return pd.Series(dtype=float)
    frame = pd.concat(partes, axis=1).ffill().dropna()
    return frame.mean(axis=1)          # pesos iguais


def estatisticas(serie: pd.Series) -> dict:
    ret = serie.pct_change().dropna()
    anos = len(ret) / 252
    cagr = float(serie.iloc[-1] ** (1 / anos) - 1) if anos > 0 else 0.0
    vol = float(ret.std() * np.sqrt(252))
    dd = float((serie / serie.cummax() - 1).min())
    pior_mes = float(serie.resample("ME").last().pct_change().min())
    return {"cagr": cagr, "vol": vol, "drawdown": dd, "pior_mes": pior_mes,
            "sharpe_cdi": (cagr - CDI_ANUAL) / vol if vol > 0 else 0.0,
            "calmar": cagr / abs(dd) if dd < 0 else float("nan")}


def main() -> None:
    anos = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    store = HistoryStore(ROOT / "data")
    carteiras = carregar(ROOT / "data" / "carteiras.json")
    inicio = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(anos * 365.25))

    print(f"═══ Simulação retroativa · {anos:.0f} anos · pesos iguais, sem rebalancear ═══")
    print("⚠️  retorno OTIMISTA por construção (papéis escolhidos com dados de hoje);")
    print("    o que vale aqui é a comparação de RISCO entre perfis.\n")
    print(f"{'carteira':<18} {'retorno a.a.':>13} {'vol':>7} {'pior queda':>11} "
          f"{'pior mês':>9} {'Calmar':>7} {'Sharpe/CDI':>11}")
    print("-" * 84)

    curvas, linhas = {}, []
    for c in carteiras:
        serie = curva(c, store, inicio)
        if serie.empty:
            continue
        curvas[c.nome] = serie
        s = estatisticas(serie)
        linhas.append({"carteira": c.nome, "perfil": c.perfil, **{k: round(v, 4) for k, v in s.items()}})
        print(f"{c.nome:<18} {s['cagr']:>12.1%} {s['vol']:>7.1%} {s['drawdown']:>10.1%} "
              f"{s['pior_mes']:>9.1%} {s['calmar']:>7.2f} {s['sharpe_cdi']:>11.2f}")

    cdi_cagr = CDI_ANUAL
    print(f"{'CDI (referência)':<18} {cdi_cagr:>12.1%} {0.0:>7.1%} {0.0:>10.1%} "
          f"{0.0:>9.1%} {'—':>7} {'—':>11}")

    if len(curvas) > 1:
        print("\n── correlação entre as carteiras (retornos diários) ──")
        rets = pd.DataFrame({k: v.pct_change() for k, v in curvas.items()}).dropna()
        print(rets.corr().round(2).to_string())

        print("\n── comportamento nos 5 piores meses do índice ──")
        idx = [k for k in curvas if "BOVA" in k or "ndice" in k]
        if idx:
            mensal = pd.DataFrame({k: v.resample("ME").last().pct_change() for k, v in curvas.items()}).dropna()
            piores = mensal.nsmallest(5, idx[0])
            print((piores * 100).round(1).to_string())

    out = ROOT / "web" / "carteiras_retroativo.json"
    out.write_text(json.dumps({
        "anos": anos, "aviso": "retorno com viés de seleção; use para comparar risco",
        "estatisticas": linhas,
        "curvas": {k: [{"d": str(i.date()), "v": round(float(v), 4)}
                       for i, v in serie.resample("W").last().dropna().items()]
                   for k, serie in curvas.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
