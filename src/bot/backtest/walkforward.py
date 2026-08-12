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


def robust_score(result: BacktestResult, min_trades: int = 10) -> float:
    """Critério de otimização resistente a sorte.

    Otimizar por PnL bruto premia a janela em que dois trades deram
    certo. Aqui exigimos amostra mínima e dividimos o lucro pelo pior
    drawdown — preferindo o parâmetro que ganhou de forma estável ao
    que ganhou mais com um susto pelo caminho.
    """
    if len(result.trades) < min_trades:
        return float("-inf")
    drawdown = max(result.max_drawdown, 1.0)
    return result.total_pnl / drawdown


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

    def risk_metrics(self) -> dict:
        """Métricas de segurança do resultado out-of-sample agregado.

        A curva OOS é a emenda das janelas de teste: é ela que
        representa o que o operador teria vivido.
        """
        trades = self.oos_trades
        if not trades:
            return {}
        equity, peak, max_dd, cum = [], float("-inf"), 0.0, 0.0
        streak = longest = 0
        for trade in trades:
            cum += trade.pnl
            equity.append(cum)
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
            streak = streak + 1 if trade.pnl < 0 else 0
            longest = max(longest, streak)
        mean = cum / len(trades)
        variance = sum((t.pnl - mean) ** 2 for t in trades) / max(len(trades) - 1, 1)
        std = variance**0.5
        return {
            "expectancy": round(mean, 2),
            "pnl_std": round(std, 2),
            "trade_sharpe": round(mean / std, 3) if std > 0 else 0.0,
            "max_drawdown": round(max_dd, 2),
            "calmar": round(cum / max_dd, 2) if max_dd > 0 else None,
            "longest_losing_streak": longest,
            "windows_positive": sum(1 for w in self.windows if w.test_result.total_pnl >= 0),
        }

    def summary(self) -> str:
        metrics = self.risk_metrics()
        lines = [
            f"Janelas: {len(self.windows)} | Trades out-of-sample: "
            f"{len(self.oos_trades)} | PnL out-of-sample: {self.oos_pnl:.2f}"
        ]
        if metrics:
            lines.append(
                f"  Drawdown máx: {metrics['max_drawdown']:.0f} | "
                f"Calmar: {metrics['calmar']} | "
                f"Expectativa/trade: {metrics['expectancy']:.0f} ± {metrics['pnl_std']:.0f} | "
                f"Consistência: {metrics['trade_sharpe']} | "
                f"Pior sequência: {metrics['longest_losing_streak']} perdas"
            )
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
        slippage_points: float = 0.0,
        cost_per_contract: float = 0.0,
        max_holding_bars: int = 0,
        # Critério de otimização; padrão: PnL sobre drawdown com amostra mínima
        metric: Callable[[BacktestResult], float] | None = None,
    ):
        self.strategy_factory = strategy_factory
        self.risk_factory = risk_factory
        self.point_value = point_value
        self.warmup = warmup
        self.slippage_points = slippage_points
        self.cost_per_contract = cost_per_contract
        self.max_holding_bars = max_holding_bars
        self.metric = metric or robust_score

    def _backtest(self, symbol: str, candles: pd.DataFrame, params: dict) -> BacktestResult:
        engine = BacktestEngine(
            self.strategy_factory(params),
            self.risk_factory(),
            point_value=self.point_value,
            warmup=self.warmup,
            slippage_points=self.slippage_points,
            cost_per_contract=self.cost_per_contract,
            max_holding_bars=self.max_holding_bars,
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
        if best_params is None:
            # Nenhuma combinação atingiu a amostra mínima: fica com a primeira
            best_params = next(param_grid(grid))
            best_result = self._backtest(symbol, candles, best_params)
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
