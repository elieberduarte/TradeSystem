"""Saída por ADR — amplitude diária média (Marcio Santos, cap. 7).

O único bloco 100% mecânico extraído dos livros de price action: alvo
e stop como frações da amplitude média do ativo, em vez de múltiplos
do ATR ou valores fixos.

    ADR = média de (máxima − mínima) dos últimos 200 pregões

    gráficos rápidos (5–15 min):  alvo ADR/10 · stop ADR/15
    gráficos médios (1–4 h):      alvo ADR/5  · stop ADR/8
    gráficos lentos (8 h–diário): alvo ADR/2  · stop ADR/3

Duas observações que o autor não faz e que importam para nós:

1. As três "estratégias" têm o MESMO payoff (~1,5:1). Não são três
   abordagens — é uma aposta 1,5:1 reescalonada pelo timeframe. Logo o
   único parâmetro real aqui é o divisor.

2. Na classe lenta os valores são grandes o bastante para a fricção
   virar 2–3% do risco, contra ~100% que a mata no intradiário. É
   justamente por isso que vale testar esta e não as outras.

Como o ADR normaliza pela volatilidade do próprio ativo, tem chance
real de replicar entre instrumentos diferentes — que é o nosso
critério de aprovação.
"""

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType

# Divisores por classe de gráfico, do livro
GRIDS = {
    "rapido": (10.0, 15.0),   # alvo ADR/10, stop ADR/15
    "medio": (5.0, 8.0),
    "lento": (2.0, 3.0),
}


def adr(candles: pd.DataFrame, period: int = 200) -> pd.Series:
    """Amplitude diária média: média simples de (máxima − mínima)."""
    return (candles["high"] - candles["low"]).rolling(period).mean()


class AdrExitOverlay(BaseStrategy):
    """Substitui alvo e stop de qualquer estratégia por frações do ADR.

    A estratégia interna continua decidindo QUANDO e para QUE LADO
    entrar; esta camada só redefine onde a operação termina. Isso
    permite testar a hipótese "o padrão não vale nada mas a saída sim"
    separadamente.
    """

    def __init__(self, inner: BaseStrategy, grid: str = "lento", period: int = 200):
        super().__init__(inner.params)
        self.inner = inner
        self.mode = inner.mode
        if grid not in GRIDS:
            raise ValueError(f"Grade desconhecida: {grid} (use {list(GRIDS)})")
        self.target_div, self.stop_div = GRIDS[grid]
        self.period = period

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        signal = self.inner.generate_signal(symbol, candles)
        if signal.type == SignalType.HOLD:
            return signal
        if len(candles) < self.period:
            return Signal(symbol=symbol, type=SignalType.HOLD)

        amplitude = float(adr(candles, self.period).iloc[-1])
        if not amplitude > 0:
            return Signal(symbol=symbol, type=SignalType.HOLD)

        entry = signal.entry_price
        direction = 1 if signal.type == SignalType.BUY else -1
        signal.stop_loss = entry - direction * amplitude / self.stop_div
        signal.take_profit = entry + direction * amplitude / self.target_div
        return signal


class RandomEntryStrategy(BaseStrategy):
    """Entrada aleatória com frequência controlada — o controle do teste.

    Serve para separar duas hipóteses que costumam ser confundidas:
    o resultado vem do SINAL ou da GESTÃO DE SAÍDA? Se uma saída
    "funciona" tanto sobre um padrão quanto sobre ruído puro, ela é
    neutra e o padrão é que estava carregando (ou vice-versa).

    A direção segue a deriva de longo prazo (só compras), para não
    comparar contra uma aposta estruturalmente perdedora.
    """

    mode = "swing_trade"

    def __init__(self, probability: float = 0.02, seed: int = 42, warmup: int = 200):
        super().__init__({"probability": probability, "seed": seed})
        self.probability = probability
        self.warmup = warmup
        self._seed = seed

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        if len(candles) < self.warmup:
            return hold
        # Pseudoaleatório determinístico: mesma barra sempre dá o mesmo
        # resultado, para o backtest ser reprodutível
        key = (hash((symbol, candles.index[-1].value, self._seed)) & 0xFFFFFFFF) / 0xFFFFFFFF
        if key >= self.probability:
            return hold
        close = float(candles["close"].iloc[-1])
        # Stop e alvo provisórios: a camada de saída os substitui
        return Signal(
            symbol=symbol, type=SignalType.BUY, entry_price=close,
            stop_loss=close * 0.97, take_profit=close * 1.05,
        )
