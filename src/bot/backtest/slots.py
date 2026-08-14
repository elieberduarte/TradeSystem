"""Concorrência por vagas: o que fazer quando há mais sinais que capital.

O backtest por instrumento assume que todo sinal vira posição. Na
conta real há N vagas de caixa — e em parte dos dias os sinais
excedem as vagas livres. Alguém precisa decidir quem entra, e essa
decisão precisa ser uma REGRA, não um humano na frente da tela (o
projeto existe para remover exatamente esse tipo de decisão).

Como cada instrumento é backtestado de forma independente, pular um
trade não altera os demais — então a carteira com teto de vagas é
exatamente uma seleção de subconjunto dos trades originais, o que
permite comparar regras de seleção sem re-simular estratégia.

Regras disponíveis (todas usam só informação da hora da decisão):
  alfabetica — ordem fixa de símbolo (baseline arbitrária)
  aleatoria  — sorteio (mede quanto a escolha importa; com várias
               sementes vira a distribuição do azar)
  bloco      — prioriza o bloco econômico MENOS representado na
               carteira atual (diversificação primeiro)
  margem     — prioriza o contrato de margem mais barata (maximiza
               o aproveitamento do caixa da vaga)
"""

import random
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SlotTrade:
    symbol: str
    block: str
    entry: pd.Timestamp
    exit: pd.Timestamp
    pnl: float
    margin: float = 0.0


def _rank_alfabetica(candidates, held, rng):
    return sorted(candidates, key=lambda t: t.symbol)


def _rank_aleatoria(candidates, held, rng):
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled


def _rank_bloco(candidates, held, rng):
    held_blocks = defaultdict(int)
    for trade in held:
        held_blocks[trade.block] += 1
    # Menos representado primeiro; empate decide por símbolo (estável)
    return sorted(candidates, key=lambda t: (held_blocks[t.block], t.symbol))


def _rank_margem(candidates, held, rng):
    return sorted(candidates, key=lambda t: (t.margin, t.symbol))


RULES = {
    "alfabetica": _rank_alfabetica,
    "aleatoria": _rank_aleatoria,
    "bloco": _rank_bloco,
    "margem": _rank_margem,
}


@dataclass
class SlotResult:
    taken: list[SlotTrade]
    skipped: list[SlotTrade]
    total_pnl: float
    max_drawdown: float
    calmar: float
    contention_days: int  # dias em que a regra precisou escolher


def simulate_slots(
    trades: list[SlotTrade], slots: int, rule: str, seed: int = 0
) -> SlotResult:
    """Reproduz a carteira com teto de `slots` posições simultâneas.

    Uma vaga é liberada no dia da saída (a saída acontece durante o
    pregão; a entrada nova é no fechamento). Sinais do dia disputam as
    vagas livres na ordem definida pela regra.
    """
    rank = RULES[rule]
    rng = random.Random(seed)

    by_entry: dict[pd.Timestamp, list[SlotTrade]] = defaultdict(list)
    for trade in trades:
        by_entry[trade.entry].append(trade)

    held: list[SlotTrade] = []
    taken: list[SlotTrade] = []
    skipped: list[SlotTrade] = []
    contention = 0

    for day in sorted(by_entry):
        held = [t for t in held if t.exit > day]
        candidates = by_entry[day]
        free = slots - len(held)
        if len(candidates) > free:
            contention += 1
        chosen = rank(candidates, held, rng)[: max(free, 0)]
        for trade in candidates:
            (taken if trade in chosen else skipped).append(trade)
        held.extend(chosen)

    daily = defaultdict(float)
    for trade in taken:
        daily[trade.exit] += trade.pnl
    series = pd.Series(daily).sort_index()
    equity = series.cumsum()
    peak, max_dd = float("-inf"), 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)
    total = float(equity.iloc[-1]) if len(equity) else 0.0

    return SlotResult(
        taken=taken,
        skipped=skipped,
        total_pnl=total,
        max_drawdown=max_dd,
        calmar=total / max_dd if max_dd > 0 else float("inf"),
        contention_days=contention,
    )
