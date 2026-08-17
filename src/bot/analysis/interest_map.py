"""Mapa de interesse: topos/fundos automáticos + níveis redondos + 50%.

Compõe três coisas que o projeto já validou ou mediu:

  pivôs      topos e fundos pelo zigzag CAUSAL (swings.py) — cada um
             carrega a barra em que ficou conhecível; nada aqui usa
             informação futura.
  redondos   múltiplos de um passo (WIN: 1.000 pts; WDO: 10 pts) na
             vizinhança do preço — o único tipo de nível que superou
             o placebo no estudo de níveis (+2,5 p.p., p=0,0002).
  50%        o meio de cada perna confirmada (topo↔fundo) — o "ponto
             focal sem cálculo" da conversa sobre Fibonacci; a única
             retração que não depende de qual swing cada um desenha.

Tudo é DESCRIÇÃO do terreno, não sinal: os estudos mostraram que a
"defesa" de nível repica na taxa da geometria (random walk). O mapa
serve para leitura, para o painel e como matéria-prima de futuros
testes com fluxo — não como gatilho por si.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.bot.analysis.swings import Pivot, swing_pivots

ROUND_STEP = {"WIN": 1_000.0, "IND": 1_000.0, "WDO": 10.0, "DOL": 10.0}


@dataclass(frozen=True)
class Level:
    kind: str          # "topo" | "fundo" | "redondo" | "meio"
    price: float
    origin: str        # texto curto: de onde veio
    known_at: int      # índice da barra em que ficou conhecível (-1 = sempre)


def round_step_for(symbol: str) -> float:
    root = symbol.replace("$N", "")[:3]
    return ROUND_STEP.get(root, 0.0)


def round_levels(price: float, step: float, span: float) -> list[Level]:
    """Múltiplos de `step` dentro de ±span do preço."""
    if step <= 0:
        return []
    lo = np.floor((price - span) / step) * step
    hi = np.ceil((price + span) / step) * step
    marks = np.arange(lo, hi + step / 2, step)
    return [Level("redondo", float(m), f"redondo {m:,.0f}", -1) for m in marks]


def build_map(
    candles: pd.DataFrame,
    symbol: str,
    threshold: float,
    lookback_pivots: int = 6,
    round_span: float | None = None,
) -> tuple[list[Level], list[Pivot]]:
    """Níveis de interesse conhecíveis na ÚLTIMA barra do DataFrame."""
    closes = candles["close"].reset_index(drop=True)
    pivots = swing_pivots(closes, threshold)
    last_index = len(closes) - 1
    known = [p for p in pivots if p.confirm_index <= last_index][-lookback_pivots:]

    levels: list[Level] = []
    for pivot in known:
        levels.append(Level(pivot.kind, float(pivot.price),
                            f"{pivot.kind} confirmado na barra {pivot.confirm_index}",
                            pivot.confirm_index))
    # 50% de cada perna entre pivôs consecutivos
    for a, b in zip(known, known[1:]):
        mid = (a.price + b.price) / 2.0
        levels.append(Level("meio", float(mid),
                            f"50% da perna {a.price:,.0f}→{b.price:,.0f}",
                            b.confirm_index))
    price = float(closes.iloc[-1])
    span = round_span if round_span is not None else 3 * threshold
    levels.extend(round_levels(price, round_step_for(symbol), span))
    return levels, pivots


def nearest(levels: list[Level], price: float, n: int = 6) -> list[tuple[Level, float]]:
    """Os N níveis mais próximos do preço, com a distância assinada."""
    ranked = sorted(((lvl, lvl.price - price) for lvl in levels), key=lambda t: abs(t[1]))
    return ranked[:n]


def render(symbol: str, price: float, levels: list[Level], n: int = 8) -> str:
    """Texto do mapa: acima e abaixo do preço, mais perto primeiro."""
    above = sorted((l for l in levels if l.price > price), key=lambda l: l.price)[:n]
    below = sorted((l for l in levels if l.price <= price), key=lambda l: -l.price)[:n]
    lines = [f"── {symbol} · preço {price:,.1f} ──"]
    for lvl in reversed(above):
        lines.append(f"  ▲ {lvl.price:>10,.1f}  {lvl.price - price:>+8,.1f}  {lvl.kind:<8} {lvl.origin}")
    lines.append(f"  ● {price:>10,.1f}  {'':>8}  preço")
    for lvl in below:
        lines.append(f"  ▼ {lvl.price:>10,.1f}  {lvl.price - price:>+8,.1f}  {lvl.kind:<8} {lvl.origin}")
    return "\n".join(lines)
