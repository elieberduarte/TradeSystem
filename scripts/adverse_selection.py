"""Mede a seleção adversa da ordem limitada.

A ordem limitada economiza o spread. Mas ela só é executada quando o
preço vem até você — ou seja, quando o mercado está indo contra a sua
direção. Este script quantifica o custo dessa troca comparando, no
mesmo histórico e com o mesmo alvo:

  A) entrada no fechamento (paga o spread, sem seleção)
  B) entrada por limitada N pontos atrás (não paga spread, mas só
     executa quando o preço avança contra você)

Se a taxa de acerto de (B) cair mais do que o spread economizado vale,
a ordem limitada é uma armadilha, não uma vantagem.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.data.history import HistoryStore

TICK = 5.0
FEES_POINTS = 7.5          # emolumentos + liquidação, ida e volta
STEP = 10                  # amostragem de candles
MAX_BARS = 15              # janela para resolver alvo ou stop
FILL_WINDOW = 3            # barras para a limitada ser executada


def market_entry(high, low, close, target: float) -> dict:
    """Entrada no fechamento, alvo e stop simétricos."""
    wins = losses = 0
    for i in range(0, len(close) - MAX_BARS - 1, STEP):
        entry = close[i]
        up, down = entry + target, entry - target
        for j in range(i + 1, i + 1 + MAX_BARS):
            if high[j] >= up:
                wins += 1
                break
            if low[j] <= down:
                losses += 1
                break
    total = wins + losses
    rate = wins / total if total else 0.0
    # Paga o spread na entrada e no stop; alvo é limitado
    gain = target - TICK - FEES_POINTS
    loss = target + TICK + TICK + FEES_POINTS
    return {
        "modo": "mercado (paga spread)",
        "oportunidades": total,
        "executadas": total,
        "execucao": "100%",
        "acerto": rate,
        "expectativa_pts": rate * gain - (1 - rate) * loss,
    }


def limit_entry(high, low, close, target: float, offset: float) -> dict:
    """Entrada por limitada `offset` pontos abaixo do fechamento."""
    filled = wins = losses = 0
    opportunities = 0
    for i in range(0, len(close) - MAX_BARS - FILL_WINDOW - 1, STEP):
        opportunities += 1
        limit = close[i] - offset
        # Procura execução nas próximas FILL_WINDOW barras
        fill_bar = None
        for j in range(i + 1, i + 1 + FILL_WINDOW):
            if low[j] < limit:
                fill_bar = j
                break
        if fill_bar is None:
            continue
        filled += 1
        up, down = limit + target, limit - target

        # Dentro do candle da execução não se sabe a ordem dos preços: o
        # preço que desceu até a limitada pode ter subido antes ou depois.
        # Assumir que a máxima da MESMA barra atingiu o alvo é olhar o
        # futuro dentro da barra — o viés clássico que infla backtest de
        # scalping. Aqui só o stop conta nessa barra (pior caso).
        resolved = False
        if low[fill_bar] <= down:
            losses += 1
            resolved = True

        if not resolved:
            for j in range(fill_bar + 1, fill_bar + 1 + MAX_BARS):
                if j >= len(close):
                    break
                if high[j] >= up:
                    wins += 1
                    break
                if low[j] <= down:
                    losses += 1
                    break
    total = wins + losses
    rate = wins / total if total else 0.0
    # Não paga spread na entrada nem no alvo; só o stop paga
    gain = target - FEES_POINTS
    loss = target + TICK + FEES_POINTS
    return {
        "modo": f"limitada {offset:.0f} pts atrás",
        "oportunidades": opportunities,
        "executadas": filled,
        "execucao": f"{filled / opportunities:.0%}" if opportunities else "—",
        "acerto": rate,
        "expectativa_pts": rate * gain - (1 - rate) * loss,
    }


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "WIN$N"
    candles = HistoryStore().load(symbol, "1m")
    if candles is None:
        raise SystemExit(f"Sem acervo de 1 minuto para {symbol}")

    high = candles["high"].to_numpy()
    low = candles["low"].to_numpy()
    close = candles["close"].to_numpy()

    print(f"═══ Seleção adversa da ordem limitada · {symbol} 1min ═══")
    print(f"{len(candles):,} candles · alvo e stop simétricos · até {MAX_BARS} min para resolver\n")

    for target in (20.0, 30.0, 50.0):
        print(f"── Alvo e stop de {target:.0f} pontos ──")
        rows = [market_entry(high, low, close, target)]
        for offset in (5.0, 10.0, 20.0):
            rows.append(limit_entry(high, low, close, target, offset))
        frame = pd.DataFrame(rows)
        frame["acerto"] = frame["acerto"].map(lambda v: f"{v:.1%}")
        frame["expectativa_pts"] = frame["expectativa_pts"].round(1)
        frame["expectativa_R$"] = (frame["expectativa_pts"] * 0.20).round(2)
        print(frame.to_string(index=False))
        print()

    print("Leitura: se o acerto da limitada cai bem abaixo do acerto a mercado,")
    print("a economia do spread não compensa — você só é executado quando o")
    print("mercado já decidiu ir contra a sua posição.")


if __name__ == "__main__":
    main()
