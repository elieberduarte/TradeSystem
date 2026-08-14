"""Testes da saída por falha de Fibonacci (Eykyn E-5) no motor."""

from datetime import time

import pandas as pd

from src.bot.backtest.engine import BacktestEngine
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.base import BaseStrategy, Signal, SignalType


class OneShotLong(BaseStrategy):
    """Compra uma única vez, na barra pedida, sem alvo (alvo longe)."""

    mode = "swing_trade"

    def __init__(self, at: int, entry: float, stop: float):
        super().__init__({})
        self.at = at
        self.entry = entry
        self.stop = stop
        self.fired = False

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        if not self.fired and len(candles) >= self.at:
            self.fired = True
            return Signal(
                symbol=symbol, type=SignalType.BUY, entry_price=self.entry,
                stop_loss=self.stop, take_profit=self.entry + 1000.0,
            )
        return Signal(symbol=symbol, type=SignalType.HOLD)


def frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1000.0] * len(rows),
        },
        index=pd.date_range("2024-01-01", periods=len(rows), freq="B"),
    )


def swing_risk() -> RiskManager:
    return RiskManager(
        RiskConfig(
            capital=100_000.0, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=100.0, max_open_positions=1,
            mode="swing_trade", trading_start=time(0, 0),
            trading_end=time(23, 59), max_consecutive_losses=0,
        )
    )


def run(rows, fib):
    engine = BacktestEngine(
        OneShotLong(at=3, entry=100.0, stop=90.0), swing_risk(),
        warmup=2, fib_exit=fib,
    )
    return engine.run("WIN", frame(rows))


def test_fib_fecha_quando_devolve_mais_de_618():
    # Entra a 100, melhor fechamento 120 (swing 20). Retração de 61,8%
    # do swing = fechar em 107,64 ou menos.
    rows = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),   # sinal no fechamento
        (100, 121, 100, 120),  # pico: swing = 20
        (120, 121, 106, 107),  # devolve 13 > 12,36: sai a 107
        (107, 108, 106, 107),
    ]
    result = run(rows, fib=0.618)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "fib"
    assert trade.exit_price == 107.0


def test_fib_segura_enquanto_a_retracao_e_menor():
    rows = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 121, 100, 120),  # swing = 20; gatilho em 107,64
        (120, 121, 109, 110),  # devolve 10 < 12,36: segura
        (110, 122, 110, 121),  # novo pico 121
        (121, 122, 120, 121),
    ]
    result = run(rows, fib=0.618)
    # nada de "fib": a posição morre no fim do histórico
    assert result.trades[0].exit_reason == "fim do histórico"


def test_fib_nao_age_sem_lucro_no_fechamento():
    # O preço nunca fecha acima da entrada: swing = 0, regra inerte,
    # quem age é o stop normal.
    rows = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 100, 95, 96),
        (96, 97, 89, 90),      # stop 90
    ]
    result = run(rows, fib=0.618)
    assert result.trades[0].exit_reason == "stop"


def test_fib_desligado_nao_interfere():
    rows = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 121, 100, 120),
        (120, 121, 106, 107),  # devolveria 65% do swing
        (107, 108, 106, 107),
    ]
    result = run(rows, fib=0.0)
    assert result.trades[0].exit_reason == "fim do histórico"
