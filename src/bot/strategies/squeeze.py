"""Squeeze de Bollinger: compressão de volatilidade antes do movimento.

A metade da banda de Bollinger que nunca testamos. A reversão na
banda (band_fade) já foi reprovada por replicação (5/15); o squeeze
é a tese OPOSTA e tem o melhor pedigree interno: é a versão contínua
do Inside Day/Narrow Range — o único sinal dos livros que replicou
(79%). Compressão medida no próprio ativo (percentil da largura de
banda), porque largura absoluta não transfere entre instrumentos.

Regra pré-declarada, sem otimização: largura de banda de ONTEM no
quinto inferior dos últimos 120 pregões (o squeeze precisa existir
ANTES do rompimento — a barra do rompimento já infla as bandas) e
fechamento rompendo a banda. Saídas iguais à base aprovada
(stop 2×ATR, alvo 3R).
"""

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType
from src.bot.strategies.swing_reversion import atr

DEFAULTS = {
    "period": 20,
    "width_k": 2.0,
    "lookback": 120,
    # Percentil máximo da largura de banda de ontem para contar squeeze
    "squeeze_pct": 0.20,
    "stop_atr": 2.0,
    "rr": 3.0,
    "atr_period": 14,
    "long_only": True,
}


def bollinger(closes: pd.Series, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    middle = closes.rolling(period).mean()
    deviation = closes.rolling(period).std(ddof=0)
    return pd.DataFrame({
        "middle": middle,
        "upper": middle + k * deviation,
        "lower": middle - k * deviation,
        "bandwidth": (2 * k * deviation) / middle.abs(),
    })


def squeeze_rank(bandwidth: pd.Series, lookback: int) -> float:
    """Percentil da largura de ONTEM entre os últimos `lookback` pregões."""
    window = bandwidth.iloc[-(lookback + 1) : -1].dropna()
    if len(window) < lookback // 2:
        return float("nan")
    yesterday = float(bandwidth.iloc[-2])
    return float((window <= yesterday).mean())


class SqueezeBreakoutStrategy(BaseStrategy):
    """Rompimento da banda vindo de compressão."""

    mode = "swing_trade"

    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        if len(candles) < p["lookback"] + p["period"] + 2:
            return hold

        bands = bollinger(candles["close"], p["period"], p["width_k"])
        rank = squeeze_rank(bands["bandwidth"], p["lookback"])
        if pd.isna(rank) or rank > p["squeeze_pct"]:
            return hold

        close_now = float(candles["close"].iloc[-1])
        stop_distance = float(atr(candles, p["atr_period"]).iloc[-1]) * p["stop_atr"]
        if stop_distance <= 0:
            return hold

        if close_now > float(bands["upper"].iloc[-1]):
            return Signal(
                symbol=symbol, type=SignalType.BUY, entry_price=close_now,
                stop_loss=close_now - stop_distance,
                take_profit=close_now + p["rr"] * stop_distance,
            )
        if close_now < float(bands["lower"].iloc[-1]) and not p["long_only"]:
            return Signal(
                symbol=symbol, type=SignalType.SELL, entry_price=close_now,
                stop_loss=close_now + stop_distance,
                take_profit=close_now - p["rr"] * stop_distance,
            )
        return hold


class SqueezeFilterOverlay(BaseStrategy):
    """Só deixa a estratégia-base entrar quando o mercado vem comprimido.

    Testa se "romper vindo de squeeze" é melhor que "romper" — a
    mesma pergunta que o filtro de range estreito respondeu para o
    Inside Day (lá, melhorou o Calmar de 0,07 para 0,15).
    """

    def __init__(self, inner: BaseStrategy, params: dict | None = None):
        super().__init__(inner.params)
        self.inner = inner
        self.mode = inner.mode
        self.squeeze_params = {**DEFAULTS, **(params or {})}

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        signal = self.inner.generate_signal(symbol, candles)
        if signal.type == SignalType.HOLD:
            return signal
        p = self.squeeze_params
        if len(candles) < p["lookback"] + p["period"] + 2:
            return Signal(symbol=signal.symbol, type=SignalType.HOLD)
        bands = bollinger(candles["close"], p["period"], p["width_k"])
        rank = squeeze_rank(bands["bandwidth"], p["lookback"])
        if pd.isna(rank) or rank > p["squeeze_pct"]:
            return Signal(symbol=signal.symbol, type=SignalType.HOLD)
        return signal
