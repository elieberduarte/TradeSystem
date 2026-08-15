"""Reversão de médio prazo (swing, candles diários).

O estudo do horizonte diário encontrou correlação de −0,37 entre o
retorno dos últimos 60 pregões e o dos 60 seguintes no WIN: quedas
longas tendem a ser seguidas de altas e vice-versa. Esta estratégia
aposta nessa reversão.

Ressalva registrada em código porque importa: janelas de 60 dias se
sobrepõem, então 1.128 amostras equivalem a ~19 observações
independentes. O sinal é forte na medida, frágil na estatística — por
isso o walk-forward é o juiz, não a correlação.
"""

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType

DEFAULTS = {
    # Janela do retorno passado que dispara a aposta contrária
    "lookback": 60,
    # Só opera quando o movimento passado excede este desvio-padrão
    # da própria série de retornos de mesma janela
    "threshold_std": 1.0,
    # Stop em múltiplos do ATR diário
    "stop_atr": 3.0,
    "rr": 2.0,
    "atr_period": 14,
}


def atr(candles: pd.DataFrame, period: int) -> pd.Series:
    prev_close = candles["close"].shift(1)
    true_range = pd.concat(
        [
            candles["high"] - candles["low"],
            (candles["high"] - prev_close).abs(),
            (candles["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


class SwingReversionStrategy(BaseStrategy):
    mode = "swing_trade"

    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        needed = p["lookback"] * 2 + p["atr_period"]
        if len(candles) < needed:
            return hold

        returns = candles["close"].pct_change(p["lookback"])
        past = float(returns.iloc[-1])
        if pd.isna(past):
            return hold
        # Referência de "movimento grande": desvio da própria série
        dispersion = float(returns.dropna().std())
        if dispersion == 0 or abs(past) < p["threshold_std"] * dispersion:
            return hold

        entry = float(candles["close"].iloc[-1])
        stop_distance = float(atr(candles, p["atr_period"]).iloc[-1]) * p["stop_atr"]
        if stop_distance <= 0:
            return hold

        # Aposta contrária ao movimento longo
        if past < 0:
            return Signal(
                symbol=symbol, type=SignalType.BUY, entry_price=entry,
                stop_loss=entry - stop_distance,
                take_profit=entry + p["rr"] * stop_distance,
                reason=(f"queda de {abs(past):.1%} em {p['lookback']} pregões, além de "
                        f"{p['threshold_std']}σ da própria série — movimento longo demais, "
                        f"aposta contrária (reversão de médio prazo)"),
            )
        return Signal(
            symbol=symbol, type=SignalType.SELL, entry_price=entry,
            stop_loss=entry + stop_distance,
            take_profit=entry - p["rr"] * stop_distance,
            reason=(f"alta de {past:.1%} em {p['lookback']} pregões, além de "
                    f"{p['threshold_std']}σ da própria série — movimento longo demais, "
                    f"aposta contrária (reversão de médio prazo)"),
        )
