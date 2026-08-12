"""Viabilidade aritmética do scalping no WIN.

Antes de testar estratégias, medir o terreno: quanto o WIN anda em 1 a
10 minutos, e qual taxa de acerto seria necessária para pagar a
fricção com alvos curtos. Se a taxa exigida for absurda, nenhuma
estratégia resolve — é aritmética, não sinal.

Uso: python scripts/scalping_feasibility.py [SIMBOLO]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.data.history import HistoryStore

# WIN: tick de 5 pts = R$ 1,00; 1 pt = R$ 0,20
TICK_POINTS = 5.0
# Cruzar o spread (1 tick) + emolumentos e liquidação (~R$ 1,50 = 7,5 pts)
FRICTION_POINTS = TICK_POINTS + 7.5


def move_distribution(candles: pd.DataFrame, horizons=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    """Quanto o preço se move em N minutos, em pontos."""
    close = candles["close"]
    rows = []
    for horizon in horizons:
        move = (close.shift(-horizon) - close).dropna()
        excursion = (
            candles["high"].rolling(horizon).max().shift(-horizon)
            - candles["low"].rolling(horizon).min().shift(-horizon)
        ).dropna()
        rows.append(
            {
                "minutos": horizon,
                "movimento_medio": round(float(move.abs().mean()), 1),
                "mediana": round(float(move.abs().median()), 1),
                "p75": round(float(move.abs().quantile(0.75)), 1),
                "amplitude_media": round(float(excursion.mean()), 1),
                "friccao_%_do_movimento": round(FRICTION_POINTS / float(move.abs().mean()) * 100, 1),
            }
        )
    return pd.DataFrame(rows)


def breakeven_table(targets=(10, 20, 30, 50, 80, 120)) -> pd.DataFrame:
    """Taxa de acerto necessária para empatar, por tamanho de alvo.

    Assume stop do mesmo tamanho do alvo (payoff 1:1 bruto). Cada
    operação paga a fricção nas duas pontas.
    """
    rows = []
    for target in targets:
        # ganho líquido = alvo − fricção; perda líquida = stop + fricção
        gain = target - FRICTION_POINTS
        loss = target + FRICTION_POINTS
        if gain <= 0:
            rows.append({"alvo_pts": target, "ganho_liquido": round(gain, 1),
                         "perda_liquida": round(loss, 1), "acerto_para_empatar": "impossível"})
            continue
        breakeven = loss / (gain + loss)
        rows.append(
            {
                "alvo_pts": target,
                "ganho_liquido": round(gain, 1),
                "perda_liquida": round(loss, 1),
                "acerto_para_empatar": f"{breakeven:.1%}",
            }
        )
    return pd.DataFrame(rows)


def hit_rate_reality(candles: pd.DataFrame, targets=(20, 30, 50, 80), max_bars=15) -> pd.DataFrame:
    """Com alvo e stop simétricos, o que o mercado entrega de verdade?

    Para cada candle, verifica o que vem primeiro nas próximas barras:
    o alvo acima ou o stop abaixo. É o teto de acerto de um scalp
    comprado com entrada aleatória — a régua contra a qual qualquer
    sinal precisa provar valor.
    """
    high = candles["high"].to_numpy()
    low = candles["low"].to_numpy()
    close = candles["close"].to_numpy()
    n = len(close)
    rows = []
    for target in targets:
        wins = losses = 0
        # Amostra a cada 20 candles para não estourar o tempo
        for i in range(0, n - max_bars - 1, 20):
            entry = close[i]
            up, down = entry + target, entry - target
            for j in range(i + 1, i + 1 + max_bars):
                if high[j] >= up:
                    wins += 1
                    break
                if low[j] <= down:
                    losses += 1
                    break
        total = wins + losses
        if total == 0:
            continue
        rate = wins / total
        gain = target - FRICTION_POINTS
        loss_amount = target + FRICTION_POINTS
        expectancy = rate * gain - (1 - rate) * loss_amount
        rows.append(
            {
                "alvo_pts": target,
                "resolvidos": total,
                "acerto_real": f"{rate:.1%}",
                "expectativa_pts": round(expectancy, 1),
                "expectativa_R$": round(expectancy * 0.20, 2),
            }
        )
    return pd.DataFrame(rows)


def limit_order_scenario(candles: pd.DataFrame, targets=(20, 30, 50, 80), max_bars=15) -> pd.DataFrame:
    """E se o scalper NÃO cruzar o spread?

    Quem entra e sai com ordem limitada paga só emolumentos, não o
    spread — é assim que o melhor track record público de B3 declara
    operar. O preço é o risco de execução: a ordem pode não ser
    executada, e o stop continua saindo a mercado.
    """
    friction_entry = 0.0            # limitada: não cruza o spread
    friction_exit_target = 0.0      # alvo também limitado
    friction_exit_stop = TICK_POINTS  # stop sai a mercado
    fees = 7.5                      # emolumentos + liquidação, ida e volta

    high, low, close = candles["high"].to_numpy(), candles["low"].to_numpy(), candles["close"].to_numpy()
    rows = []
    for target in targets:
        wins = losses = 0
        for i in range(0, len(close) - max_bars - 1, 20):
            entry = close[i]
            up, down = entry + target, entry - target
            for j in range(i + 1, i + 1 + max_bars):
                if high[j] >= up:
                    wins += 1
                    break
                if low[j] <= down:
                    losses += 1
                    break
        total = wins + losses
        if total == 0:
            continue
        rate = wins / total
        gain = target - friction_entry - friction_exit_target - fees
        loss = target + friction_entry + friction_exit_stop + fees
        expectancy = rate * gain - (1 - rate) * loss
        breakeven = loss / (gain + loss)
        rows.append(
            {
                "alvo_pts": target,
                "acerto_real": f"{rate:.1%}",
                "acerto_para_empatar": f"{breakeven:.1%}",
                "expectativa_pts": round(expectancy, 1),
                "expectativa_R$": round(expectancy * 0.20, 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "WIN$N"
    candles = HistoryStore().load(symbol, "1m")
    if candles is None:
        raise SystemExit(f"Sem acervo de 1 minuto para {symbol} — rode o download primeiro")

    print(f"═══ Viabilidade de scalping · {symbol} 1min ═══")
    print(f"{len(candles):,} candles ({candles.index.min()} → {candles.index.max()})")
    print(f"Fricção assumida: {FRICTION_POINTS:.1f} pts por operação "
          f"(1 tick de spread + emolumentos) = R$ {FRICTION_POINTS * 0.20:.2f}\n")

    print("── Quanto o WIN anda em poucos minutos ──")
    print(move_distribution(candles).to_string(index=False))
    print()

    print("── Taxa de acerto necessária só para EMPATAR (stop = alvo) ──")
    print(breakeven_table().to_string(index=False))
    print()

    print("── O que o mercado entrega: alvo ou stop, o que vier primeiro ──")
    print("(entrada a cada 20 candles, comprado, até 15 minutos para resolver)")
    print(hit_rate_reality(candles).to_string(index=False))
    print()
    print("Expectativa negativa aqui significa: nem com entrada perfeita")
    print("em termos de simetria o alvo curto paga a fricção.")
    print()

    print("── E se entrar e sair com ORDEM LIMITADA (sem cruzar o spread)? ──")
    print("(paga só emolumentos; o stop continua saindo a mercado)")
    print(limit_order_scenario(candles).to_string(index=False))


if __name__ == "__main__":
    main()
