"""Momentum intradiário de horário (Gao, Han, Li & Zhou, JFE 2018).

A direção do início do pregão (gap da noite + primeira meia hora)
prevê a direção da última meia hora. É a única família com evidência
acadêmica forte cujo edge bruto supera com folga a fricção do WIN —
porque opera UMA vez por dia, não a cada balanço de 5 minutos.

Réplicas internacionais mostram o efeito significativo em 12 de 16
mercados desenvolvidos, mas com sinal INVERTIDO no Canadá — por isso
o parâmetro `contrarian`, que permite testar as duas hipóteses. O
Brasil nunca foi testado: aqui isso é hipótese a falsificar.
"""

from datetime import time

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType

DEFAULTS = {
    # Fim da janela de formação do sinal (abertura + 30min no WIN: 09:30)
    "signal_until": time(9, 30),
    # Início da janela de execução (últimos 30 min antes da zeragem)
    "entry_from": time(17, 30),
    "entry_until": time(17, 40),
    # Inverte o sinal (hipótese de reversão, como no mercado canadense)
    "contrarian": False,
    # Só opera quando o movimento inicial supera este múltiplo do
    # movimento médio recente — filtra dias sem informação
    "min_move_mult": 0.0,
    # Stop em múltiplos do movimento inicial (0 = sem stop; a zeragem
    # de fim de pregão fecha a posição)
    "stop_mult": 3.0,
}


class IntradayMomentumStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)

        now = candles.index[-1]
        if not (p["entry_from"] <= now.time() <= p["entry_until"]):
            return hold

        today = candles[candles.index.normalize() == now.normalize()]
        previous = candles[candles.index.normalize() < now.normalize()]
        if today.empty or previous.empty:
            return hold

        # Sinal: do fechamento de ontem até o fim da primeira meia hora.
        # Inclui o gap da noite — a evidência internacional indica que é
        # daí que vem a maior parte do poder preditivo.
        opening = today[today.index.time <= p["signal_until"]]
        if opening.empty:
            return hold
        prev_close = float(previous["close"].iloc[-1])
        first_move = float(opening["close"].iloc[-1]) - prev_close
        if first_move == 0:
            return hold

        if p["min_move_mult"]:
            # Compara com a amplitude média dos pregões anteriores
            daily_range = previous.groupby(previous.index.normalize()).apply(
                lambda d: d["high"].max() - d["low"].min(), include_groups=False
            )
            if daily_range.empty:
                return hold
            reference = float(daily_range.tail(14).mean())
            if abs(first_move) < p["min_move_mult"] * reference:
                return hold

        bullish = first_move > 0
        if p["contrarian"]:
            bullish = not bullish

        entry = float(candles["close"].iloc[-1])
        stop_distance = abs(first_move) * p["stop_mult"] if p["stop_mult"] else None

        if bullish:
            return Signal(
                symbol=symbol, type=SignalType.BUY, entry_price=entry,
                stop_loss=entry - stop_distance if stop_distance else entry * 0.9,
                # O alvo real é a zeragem de fim de pregão; deixamos longe
                take_profit=entry * 1.10,
            )
        return Signal(
            symbol=symbol, type=SignalType.SELL, entry_price=entry,
            stop_loss=entry + stop_distance if stop_distance else entry * 1.1,
            take_profit=entry * 0.90,
        )
