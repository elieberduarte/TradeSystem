"""Carteiras teóricas por perfil de risco — construção e medição.

Três carteiras de R$ 20 mil, montadas por critérios CLÁSSICOS de
análise fundamentalista e risco, mais os dois benchmarks que tornam a
comparação honesta (índice e CDI). A pergunta que elas respondem não é
"qual sobe mais" — é **quanto de retorno cada perfil entrega por
unidade de risco**, medido do mesmo jeito para todos.

⚠️ RETORNO TOTAL, NÃO SÓ PREÇO. A série de preços do MT5/XP é
AJUSTADA por proventos: o dividendo pago aparece como um ajuste
retroativo na série inteira, então a valorização medida JÁ INCLUI
dividendos e JCP reinvestidos. Verificado empiricamente: BBSE3 aparece
saindo de R$ 12,33 em ago/2021 quando o preço real era ~R$ 21 — a
diferença é exatamente o acumulado de proventos. Somar dividendos por
fora seria contar duas vezes. Onde este módulo estima a PARCELA de
proventos, ele o faz pelo DY conhecido e marca como estimativa.

Pesos iguais (1/N) de propósito: é o que a literatura mostra ser
difícil de bater e não embute opinião sobre qual papel é melhor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PREGOES_ANO = 252


@dataclass
class Posicao:
    ticker: str
    quantidade: int
    preco_entrada: float
    valor: float
    # o que motivou a escolha — auditável, como todo número do projeto
    motivo: str = ""


@dataclass
class Carteira:
    nome: str
    perfil: str            # "baixo" | "medio" | "alto" | "referencia"
    descricao: str
    criterios: list[str]
    data_inicio: str
    aporte: float
    posicoes: list[Posicao] = field(default_factory=list)
    caixa: float = 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "posicoes": [asdict(p) for p in self.posicoes]}


def metricas_de_risco(daily: pd.DataFrame, indice: pd.Series | None = None,
                      janela: int = 504) -> dict:
    """Volatilidade, beta, drawdown e retorno — a régua de risco do papel."""
    closes = daily["close"].tail(janela)
    if len(closes) < 60:
        return {}
    ret = closes.pct_change().dropna()
    vol = float(ret.std() * np.sqrt(PREGOES_ANO))
    equity = (1 + ret).cumprod()
    drawdown = float((equity / equity.cummax() - 1).min())
    anos = len(ret) / PREGOES_ANO
    cagr = float((closes.iloc[-1] / closes.iloc[0]) ** (1 / anos) - 1) if anos > 0 else 0.0

    beta = float("nan")
    if indice is not None:
        par = pd.DataFrame({"a": ret, "b": indice.pct_change()}).dropna()
        if len(par) > 60 and par["b"].var() > 0:
            beta = float(par["a"].cov(par["b"]) / par["b"].var())

    return {"vol_anual": vol, "beta": beta, "max_drawdown": drawdown,
            "cagr": cagr, "sharpe_bruto": cagr / vol if vol > 0 else 0.0}


def montar(nome: str, perfil: str, descricao: str, criterios: list[str],
           tickers: list[str], precos: dict[str, float], aporte: float,
           motivos: dict[str, str] | None = None,
           data: str | None = None) -> Carteira:
    """Distribui o aporte em pesos iguais, respeitando lotes inteiros.

    O que não couber em ação inteira vira caixa — é assim na corretora
    de verdade, e ignorar isso inflaria o retorno de papéis caros.
    """
    motivos = motivos or {}
    carteira = Carteira(
        nome=nome, perfil=perfil, descricao=descricao, criterios=criterios,
        data_inicio=data or datetime.now().strftime("%Y-%m-%d"), aporte=aporte,
    )
    if not tickers:
        carteira.caixa = aporte
        return carteira

    alvo = aporte / len(tickers)
    gasto = 0.0
    for ticker in tickers:
        preco = precos.get(ticker, 0.0)
        if preco <= 0:
            continue
        quantidade = int(alvo // preco)
        if quantidade < 1:
            continue
        valor = quantidade * preco
        gasto += valor
        carteira.posicoes.append(Posicao(
            ticker=ticker, quantidade=quantidade, preco_entrada=round(preco, 2),
            valor=round(valor, 2), motivo=motivos.get(ticker, ""),
        ))
    carteira.caixa = round(aporte - gasto, 2)
    return carteira


def avaliar(carteira: Carteira, precos_hoje: dict[str, float],
            cdi_anual: float = 0.1415) -> dict:
    """Valor atual, retorno e contribuição de cada posição."""
    linhas = []
    valor = carteira.caixa
    for pos in carteira.posicoes:
        atual = precos_hoje.get(pos.ticker, pos.preco_entrada)
        valor_atual = pos.quantidade * atual
        valor += valor_atual
        linhas.append({
            "ticker": pos.ticker, "quantidade": pos.quantidade,
            "preco_entrada": pos.preco_entrada, "preco_atual": round(atual, 2),
            "valor_inicial": pos.valor, "valor_atual": round(valor_atual, 2),
            "retorno_pct": round(atual / pos.preco_entrada - 1, 4) if pos.preco_entrada else 0.0,
            "resultado": round(valor_atual - pos.valor, 2),
            "motivo": pos.motivo,
        })
    dias = max((datetime.now() - datetime.strptime(carteira.data_inicio, "%Y-%m-%d")).days, 0)
    cdi_periodo = (1 + cdi_anual) ** (dias / 365.25) - 1
    return {
        "nome": carteira.nome, "perfil": carteira.perfil,
        "aporte": carteira.aporte, "caixa": carteira.caixa,
        "valor_atual": round(valor, 2),
        "resultado": round(valor - carteira.aporte, 2),
        "retorno_pct": round(valor / carteira.aporte - 1, 4),
        "dias": dias,
        "cdi_periodo_pct": round(cdi_periodo, 4),
        "vs_cdi_pp": round((valor / carteira.aporte - 1) - cdi_periodo, 4),
        "posicoes": sorted(linhas, key=lambda r: -r["retorno_pct"]),
    }


def salvar(carteiras: list[Carteira], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"criado": datetime.now().isoformat(timespec="seconds"),
         "carteiras": [c.to_dict() for c in carteiras]},
        ensure_ascii=False, indent=1), encoding="utf-8")


def carregar(path: Path) -> list[Carteira]:
    data = json.loads(path.read_text(encoding="utf-8"))
    saida = []
    for item in data["carteiras"]:
        posicoes = [Posicao(**p) for p in item.pop("posicoes")]
        saida.append(Carteira(**item, posicoes=posicoes))
    return saida
