"""Topos e fundos em tempo real, sem olhar o futuro.

Um pivô de zigzag só passa a EXISTIR quando o preço anda o limiar de
reversão a partir do extremo — antes disso é só o preço de agora. O
erro clássico dos estudos com zigzag é usar o pivô na barra em que
ele ocorreu; aqui cada pivô carrega dois índices:

  index          onde o extremo aconteceu (para desenhar)
  confirm_index  onde ele ficou CONHECÍVEL (para decidir)

Toda análise honesta usa o segundo. Os pivôs são calculados sobre
FECHAMENTOS (dentro da barra não se sabe a ordem dos preços).
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Pivot:
    kind: str            # "topo" | "fundo"
    index: int           # barra do extremo
    price: float
    confirm_index: int   # barra em que o pivô ficou conhecível


def swing_pivots(closes: pd.Series, threshold: float) -> list[Pivot]:
    """Zigzag incremental com carimbo de confirmação.

    Um topo é confirmado quando o fechamento cai `threshold` abaixo do
    maior fechamento da perna; um fundo, quando sobe `threshold` acima
    do menor. A primeira perna nasce sem direção e é definida pelo
    primeiro movimento que atingir o limiar.
    """
    values = closes.to_numpy(dtype=float)
    if len(values) < 2 or threshold <= 0:
        return []

    pivots: list[Pivot] = []
    direction = 0                     # 0 = indefinida, +1 subindo, -1 descendo
    hi_idx = lo_idx = 0
    hi = lo = values[0]

    for i in range(1, len(values)):
        price = values[i]
        if price > hi:
            hi, hi_idx = price, i
        if price < lo:
            lo, lo_idx = price, i

        if direction >= 0 and hi - price >= threshold:
            # A perna de alta acabou: o topo em hi_idx fica conhecível agora
            pivots.append(Pivot("topo", hi_idx, hi, i))
            direction = -1
            lo, lo_idx = price, i
        elif direction <= 0 and price - lo >= threshold:
            pivots.append(Pivot("fundo", lo_idx, lo, i))
            direction = 1
            hi, hi_idx = price, i

    return pivots


@dataclass(frozen=True)
class StructureEvent:
    """Estrutura de tendência completa, no momento em que ficou conhecível.

    Alta: o fundo recém-confirmado é MAIS ALTO que o anterior e o
    último topo é mais alto que o topo que o antecedeu (HH + HL).
    Baixa: o espelho. `leg` conta a perna da estrutura (1ª confirmação,
    2ª confirmação seguida...) — mede quanto tempo estruturas duram.
    """
    direction: int       # +1 alta, -1 baixa
    confirm_index: int
    leg: int


def structure_events(pivots: list[Pivot]) -> list[StructureEvent]:
    """Eventos de estrutura confirmada, na ordem em que ficam conhecíveis."""
    events: list[StructureEvent] = []
    tops: list[Pivot] = []
    bottoms: list[Pivot] = []
    streak_dir = 0
    streak = 0

    for pivot in pivots:
        (tops if pivot.kind == "topo" else bottoms).append(pivot)
        if len(tops) < 2 or len(bottoms) < 2:
            continue

        hh = tops[-1].price > tops[-2].price
        hl = bottoms[-1].price > bottoms[-2].price
        ll = bottoms[-1].price < bottoms[-2].price
        lh = tops[-1].price < tops[-2].price

        direction = 1 if (hh and hl) else (-1 if (ll and lh) else 0)
        if direction == 0:
            streak_dir, streak = 0, 0
            continue
        streak = streak + 1 if direction == streak_dir else 1
        streak_dir = direction
        events.append(StructureEvent(direction, pivot.confirm_index, streak))

    return events
