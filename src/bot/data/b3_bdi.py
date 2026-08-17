"""Coletor do Boletim Diário da B3 (BDI) — o fluxo oficial dos players.

A API pública que alimenta o painel da B3 (arquivos.b3.com.br/bdi)
serve, entre outras, duas tabelas que interessam ao projeto:

  SharesInvesVolum       participação por tipo de investidor no
                         mercado à vista (compras e vendas em R$ mil).
                         ⚠️ Os valores são o ACUMULADO DO MÊS, não o
                         dia — descoberto porque as compras "zeram" na
                         virada de mês. O saldo DIÁRIO é a diferença
                         entre acumulados consecutivos (com reset
                         mensal); `daily_flow()` faz essa conta.
  AnalyticalFramework2   contratos em aberto por mercado (futuros)

Limitação estrutural descoberta na engenharia reversa: a API só
retém ~21 pregões. Por isso este coletor é um ACUMULADOR — cada
execução baixa o que existe e mescla ao acervo local em Parquet.
Rodando junto com o ciclo diário do bot, o histórico cresce para
sempre; a B3 esquece, nós não.

Endpoint (descoberto no bundle do SPA):
  POST /bdi/table/{nome}/{dataIni}/{dataFim}/{página}/{tamanho}
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://arquivos.b3.com.br/bdi"
STORE = Path("data/b3")


def _post(url: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "application/json",
                 "User-Agent": "trade-bot/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def workdays(reference: str) -> list[str]:
    url = f"{BASE}/table/workdays?date={reference}"
    request = urllib.request.Request(url, headers={"User-Agent": "trade-bot/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        days = json.loads(response.read().decode("utf-8"))
    return [d[:10] for d in days]


def fetch_table(name: str, day: str, take: int = 500) -> list[list]:
    """Uma tabela do BDI para um pregão (páginas concatenadas).

    Dia sem boletim (fora da retenção, feriado, ou o dia corrente
    antes da publicação) devolve lista vazia em vez de explodir — o
    coletor é um acumulador, e um buraco não pode derrubar o resto.
    """
    rows: list[list] = []
    page = 1
    while True:
        try:
            data = _post(f"{BASE}/table/{name}/{day}/{day}/{page}/{take}")
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            print(f"  aviso: {name} {day} indisponível ({exc})")
            return rows
        values = (data.get("table") or {}).get("values") or []
        rows.extend(values)
        if len(values) < take:
            return rows
        page += 1


def collect_participacao(days: list[str]) -> pd.DataFrame:
    """Participação por investidor no à vista, acumulada no acervo."""
    rows = []
    for day in days:
        for value in fetch_table("SharesInvesVolum", day):
            # layout observado: [categoria, compras_mil, compras_%, vendas_mil, vendas_%, _]
            if len(value) >= 5 and isinstance(value[0], str):
                rows.append({
                    "data": day, "categoria": value[0],
                    "compras_mil": value[1], "compras_pct": value[2],
                    "vendas_mil": value[3], "vendas_pct": value[4],
                })
    fresh = pd.DataFrame(rows)
    return _merge(fresh, STORE / "participacao_investidores.parquet",
                  keys=["data", "categoria"])


def collect_open_interest(days: list[str],
                          assets: tuple[str, ...] = ("WIN", "IND", "WDO", "DOL", "DI1")) -> pd.DataFrame:
    """Contratos em aberto dos futuros que operamos."""
    rows = []
    for day in days:
        for value in fetch_table("AnalyticalFramework2", day):
            # layout: [data, mercado, ativo, contratos_abertos, valor_ref, _]
            if len(value) >= 5 and str(value[2]) in assets and "futuro" in str(value[1]).lower():
                rows.append({
                    "data": day, "mercado": value[1], "ativo": value[2],
                    "contratos_abertos": value[3], "valor_ref_mil": value[4],
                })
    fresh = pd.DataFrame(rows)
    return _merge(fresh, STORE / "contratos_em_aberto.parquet",
                  keys=["data", "mercado"])


def daily_flow(participacao: pd.DataFrame) -> pd.DataFrame:
    """Saldo DIÁRIO por categoria a partir do acumulado mensal da B3.

    saldo_dia = (compras − vendas)(D) − (compras − vendas)(D−1) dentro
    do mesmo mês; no primeiro pregão do mês o acumulado É o dia. Dias
    ausentes no acervo (buracos) invalidam a diferença seguinte, então
    a linha fica NaN em vez de virar um salto falso.
    """
    frame = participacao.copy()
    frame["data"] = pd.to_datetime(frame["data"])
    frame["saldo_acum_mil"] = frame["compras_mil"] - frame["vendas_mil"]
    frame["mes"] = frame["data"].dt.to_period("M")
    frame = frame.sort_values(["categoria", "data"])

    out = []
    for _, group in frame.groupby(["categoria", "mes"], sort=False):
        group = group.copy()
        previous_day = group["data"].shift(1)
        previous_saldo = group["saldo_acum_mil"].shift(1)
        gap = (group["data"] - previous_day).dt.days
        # primeiro dia do mês no acervo: se for dia 1-3 útil, o acumulado é o dia
        first = previous_day.isna()
        diff = group["saldo_acum_mil"] - previous_saldo
        diff[first] = group.loc[first, "saldo_acum_mil"]
        # buraco maior que um fim de semana prolongado (> 4 dias corridos) = não confiável
        diff[(gap > 4) & ~first] = float("nan")
        group["saldo_dia_mil"] = diff
        out.append(group)
    result = pd.concat(out).sort_values(["data", "categoria"]).reset_index(drop=True)
    result["data"] = result["data"].dt.strftime("%Y-%m-%d")
    return result.drop(columns=["mes"])


def _merge(fresh: pd.DataFrame, path: Path, keys: list[str]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        merged = pd.concat([old, fresh], ignore_index=True)
        merged = merged.drop_duplicates(subset=keys, keep="last")
    else:
        merged = fresh
    merged = merged.sort_values(keys).reset_index(drop=True)
    merged.to_parquet(path)
    return merged
