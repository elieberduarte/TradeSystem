"""Camadas de gestão aplicáveis sobre qualquer estratégia.

A estratégia decide QUANDO entrar. Estas camadas decidem COMO a
posição é conduzida — e são independentes do sinal, o que permite
testar cada ideia isoladamente sobre uma base já aprovada.

Cada camada é uma transformação do sinal, não uma estratégia nova.
Isso importa para o orçamento estatístico: variar a gestão sobre uma
base validada custa menos testes do que inventar sinais do zero.
"""

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType


class RewardRatioOverlay(BaseStrategy):
    """Reescala o alvo mantendo o stop — testa outras relações risco/retorno.

    `ratio` é o alvo em múltiplos da distância do stop: 3.0 busca três
    vezes o risco, 0.5 busca metade (alvo curto, acerto alto, perdas
    grandes — a configuração 'catar moedas').
    """

    def __init__(self, inner: BaseStrategy, ratio: float):
        super().__init__(inner.params)
        self.inner = inner
        self.mode = inner.mode
        self.ratio = ratio

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        signal = self.inner.generate_signal(symbol, candles)
        if signal.type == SignalType.HOLD:
            return signal
        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            return Signal(symbol=symbol, type=SignalType.HOLD)
        direction = 1 if signal.type == SignalType.BUY else -1
        signal.take_profit = signal.entry_price + direction * self.ratio * risk
        return signal


class StopWidthOverlay(BaseStrategy):
    """Multiplica a distância do stop, preservando a relação alvo/risco.

    Stop mais largo tolera mais ruído e reduz a mão (o dimensionamento
    divide pelo stop), então não é apenas 'arriscar mais'.
    """

    def __init__(self, inner: BaseStrategy, factor: float):
        super().__init__(inner.params)
        self.inner = inner
        self.mode = inner.mode
        self.factor = factor

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        signal = self.inner.generate_signal(symbol, candles)
        if signal.type == SignalType.HOLD:
            return signal
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        if risk <= 0:
            return Signal(symbol=symbol, type=SignalType.HOLD)
        ratio = reward / risk
        direction = 1 if signal.type == SignalType.BUY else -1
        new_risk = risk * self.factor
        signal.stop_loss = signal.entry_price - direction * new_risk
        signal.take_profit = signal.entry_price + direction * ratio * new_risk
        return signal


class NoTargetOverlay(BaseStrategy):
    """Remove o alvo: a posição só sai por stop, sinal contrário ou tempo.

    É a política das tartarugas — deixar o ganho correr até o mercado
    virar. Testa se o alvo fixo está cortando as tendências longas que
    justificam o setup.
    """

    def __init__(self, inner: BaseStrategy, far_multiple: float = 100.0):
        super().__init__(inner.params)
        self.inner = inner
        self.mode = inner.mode
        self.far_multiple = far_multiple

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        signal = self.inner.generate_signal(symbol, candles)
        if signal.type == SignalType.HOLD:
            return signal
        risk = abs(signal.entry_price - signal.stop_loss)
        direction = 1 if signal.type == SignalType.BUY else -1
        signal.take_profit = signal.entry_price + direction * self.far_multiple * risk
        return signal
