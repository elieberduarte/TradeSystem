"""Ichimoku Kinko Hyo — o primo japonês do Donchian.

Pouca gente que usa a nuvem repara: Tenkan e Kijun são PONTOS MÉDIOS
de canais de Donchian ((máxima+mínima)/2 de 9 e 26 períodos), e a
nuvem são esses médios projetados 26 barras à frente. A família toda
é a do nosso único edge aprovado — o que torna o teste barato de
interpretar: ou agrega ao Donchian, ou é ele de quimono.

Deslocamento causal: a nuvem "de hoje" foi calculada com dados de 26
barras atrás (shift positivo). Nada aqui olha o futuro — o Chikou
Span, que é o preço deslocado PARA TRÁS (e por isso vive gerando
look-ahead em backtests alheios), ficou de fora de propósito.

SAÍDAS FIXADAS POR NÓS, iguais às da base aprovada (stop 2×ATR,
alvo 3R), para a comparação isolar só o GATILHO.
"""

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType
from src.bot.strategies.swing_reversion import atr

DEFAULTS = {
    "tenkan": 9,
    "kijun": 26,
    "senkou": 52,
    "displacement": 26,
    "stop_atr": 2.0,
    "rr": 3.0,
    "atr_period": 14,
    "long_only": True,
}


def ichimoku_lines(candles: pd.DataFrame, p: dict | None = None) -> pd.DataFrame:
    """Tenkan, Kijun e os limites da nuvem VÁLIDOS em cada barra."""
    p = {**DEFAULTS, **(p or {})}
    high, low = candles["high"], candles["low"]

    def midpoint(n: int) -> pd.Series:
        return (high.rolling(n).max() + low.rolling(n).min()) / 2

    tenkan = midpoint(p["tenkan"])
    kijun = midpoint(p["kijun"])
    span_a = ((tenkan + kijun) / 2).shift(p["displacement"])
    span_b = midpoint(p["senkou"]).shift(p["displacement"])
    return pd.DataFrame({
        "tenkan": tenkan, "kijun": kijun,
        "cloud_top": pd.concat([span_a, span_b], axis=1).max(axis=1),
        "cloud_bottom": pd.concat([span_a, span_b], axis=1).min(axis=1),
    })


def _exit_levels(candles: pd.DataFrame, p: dict, direction: int) -> tuple[float, float, float]:
    close = float(candles["close"].iloc[-1])
    stop_distance = float(atr(candles, p["atr_period"]).iloc[-1]) * p["stop_atr"]
    return close, close - direction * stop_distance, close + direction * p["rr"] * stop_distance


class IchimokuCrossStrategy(BaseStrategy):
    """Cruzamento Tenkan × Kijun (o "sinal de compra" clássico)."""

    mode = "swing_trade"

    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        if len(candles) < p["senkou"] + p["displacement"] + 2:
            return hold
        lines = ichimoku_lines(candles, p)
        now, before = lines.iloc[-1], lines.iloc[-2]
        if now.isna().any() or before.isna().any():
            return hold

        crossed_up = before["tenkan"] <= before["kijun"] and now["tenkan"] > now["kijun"]
        crossed_down = before["tenkan"] >= before["kijun"] and now["tenkan"] < now["kijun"]

        if crossed_up:
            entry, stop, target = _exit_levels(candles, p, 1)
            if stop < entry:
                return Signal(symbol=symbol, type=SignalType.BUY, entry_price=entry,
                              stop_loss=stop, take_profit=target)
        if crossed_down and not p["long_only"]:
            entry, stop, target = _exit_levels(candles, p, -1)
            if stop > entry:
                return Signal(symbol=symbol, type=SignalType.SELL, entry_price=entry,
                              stop_loss=stop, take_profit=target)
        return hold


class CloudCrossStrategy(BaseStrategy):
    """Fechamento cruza a nuvem: sai de dentro/abaixo para acima dela."""

    mode = "swing_trade"

    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        if len(candles) < p["senkou"] + p["displacement"] + 2:
            return hold
        lines = ichimoku_lines(candles, p)
        now, before = lines.iloc[-1], lines.iloc[-2]
        if now.isna().any() or before.isna().any():
            return hold
        close_now = float(candles["close"].iloc[-1])
        close_before = float(candles["close"].iloc[-2])

        broke_up = close_before <= before["cloud_top"] and close_now > now["cloud_top"]
        broke_down = close_before >= before["cloud_bottom"] and close_now < now["cloud_bottom"]

        if broke_up:
            entry, stop, target = _exit_levels(candles, p, 1)
            if stop < entry:
                return Signal(symbol=symbol, type=SignalType.BUY, entry_price=entry,
                              stop_loss=stop, take_profit=target)
        if broke_down and not p["long_only"]:
            entry, stop, target = _exit_levels(candles, p, -1)
            if stop > entry:
                return Signal(symbol=symbol, type=SignalType.SELL, entry_price=entry,
                              stop_loss=stop, take_profit=target)
        return hold


class CloudFilterOverlay(BaseStrategy):
    """Deixa passar compras só ACIMA da nuvem (vendas, só abaixo).

    O uso mais citado do Ichimoku: a nuvem como juiz do regime. Aqui
    ela filtra os sinais de uma estratégia-base, para medir se o
    filtro agrega algo à entrada que já aprovamos.
    """

    def __init__(self, inner: BaseStrategy, params: dict | None = None):
        super().__init__(inner.params)
        self.inner = inner
        self.mode = inner.mode
        self.cloud_params = {**DEFAULTS, **(params or {})}

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        signal = self.inner.generate_signal(symbol, candles)
        if signal.type == SignalType.HOLD:
            return signal
        lines = ichimoku_lines(candles, self.cloud_params)
        now = lines.iloc[-1]
        if now.isna().any():
            return Signal(symbol=symbol, type=SignalType.HOLD)
        close = float(candles["close"].iloc[-1])
        if signal.type == SignalType.BUY and close > now["cloud_top"]:
            return signal
        if signal.type == SignalType.SELL and close < now["cloud_bottom"]:
            return signal
        return Signal(symbol=symbol, type=SignalType.HOLD)
