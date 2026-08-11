"""Otimização de parâmetros com validação walk-forward.

Percorre o histórico em janelas deslizantes: otimiza os parâmetros da
estratégia no trecho de treino (in-sample) e mede o resultado no trecho
seguinte, que a otimização nunca viu (out-of-sample). O desempenho real
esperado é o agregado dos trechos out-of-sample — nunca o do treino.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import product

import pandas as pd

from src.bot.backtest.engine import BacktestEngine, BacktestResult, Trade
from src.bot.risk.manager import RiskManager
from src.bot.strategies.base import BaseStrategy


def param_grid(grid: dict[str, list]) -> Iterator[dict]:
    """Expande {'a': [1, 2], 'b': [3]} em [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]."""
    keys = list(grid)
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_result: BacktestResult
    test_result: BacktestResult


@dataclass
class WalkForwardReport:
    windows: list[WalkForwardWindow] = field(default_factory=list)

    @property
    def oos_trades(self) -> list[Trade]:
        return [t for w in self.windows for t in w.test_result.trades]

    @property
    def oos_pnl(self) -> float:
        return sum(t.pnl for t in self.oos_trades)

    def summary(self) -> str:
        lines = [
            f"Janelas: {len(self.windows)} | Trades out-of-sample: "
            f"{len(self.oos_trades)} | PnL out-of-sample: {self.oos_pnl:.2f}"
        ]
        for w in self.windows:
            lines.append(
                f"  {w.test_start:%Y-%m-%d} → {w.test_end:%Y-%m-%d} | "
                f"params {w.best_params} | treino {w.train_result.total_pnl:.2f} | "
                f"teste {w.test_result.total_pnl:.2f}"
            )
        return "\n".join(lines)


class WalkForward:
    def __init__(
        self,
        strategy_factory: Callable[[dict], BaseStrategy],
        risk_factory: Callable[[], RiskManager],
        point_value: float = 1.0,
        warmup: int = 100,
        # Critério de otimização; padrão: PnL total do treino
        metric: Callable[[BacktestResult], float] | None = None,
    ):
        self.strategy_factory = strategy_factory
        self.risk_factory = risk_factory
        self.point_value = point_value
        self.warmup = warmup
        self.metric = metric or (lambda r: r.total_pnl)

    def _backtest(self, symbol: str, candles: pd.DataFrame, params: dict) -> BacktestResult:
        engine = BacktestEngine(
            self.strategy_factory(params),
            self.risk_factory(),
            point_value=self.point_value,
            warmup=self.warmup,
        )
        return engine.run(symbol, candles)

    def optimize(
        self, symbol: str, candles: pd.DataFrame, grid: dict[str, list]
    ) -> tuple[dict, BacktestResult]:
        best_params, best_result, best_score = None, None, float("-inf")
        for params in param_grid(grid):
            result = self._backtest(symbol, candles, params)
            score = self.metric(result)
            if score > best_score:
                best_params, best_result, best_score = params, result, score
        return best_params, best_result

    def run(
        self,
        symbol: str,
        candles: pd.DataFrame,
        grid: dict[str, list],
        train_bars: int,
        test_bars: int,
    ) -> WalkForwardReport:
        report = WalkForwardReport()
        total = len(candles)
        start = 0
        while start + train_bars + test_bars <= total:
            train = candles.iloc[start : start + train_bars]
            # O teste recebe o warmup final do treino como contexto; o motor
            # só decide a partir do candle `warmup`, ou seja, dentro do teste.
            test_from = start + train_bars - self.warmup
            test = candles.iloc[max(test_from, 0) : start + train_bars + test_bars]

            best_params, train_result = self.optimize(symbol, train, grid)
            test_result = self._backtest(symbol, test, best_params)

            report.windows.append(
                WalkForwardWindow(
                    train_start=train.index[0],
                    test_start=candles.index[start + train_bars],
                    test_end=test.index[-1],
                    best_params=best_params,
                    train_result=train_result,
                    test_result=test_result,
                )
            )
            start += test_bars
        return report
