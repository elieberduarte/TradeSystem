"""Testes da otimização walk-forward."""

from src.bot.backtest.walkforward import WalkForward, param_grid
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.base import BaseStrategy, Signal, SignalType
from tests.conftest import make_candles


class DirectionalStrategy(BaseStrategy):
    """Entra periodicamente na direção do parâmetro — em tendência de alta,
    'buy' lucra e 'sell' perde, dando à otimização algo real para escolher."""

    def generate_signal(self, symbol, candles):
        if len(candles) % 20 != 0:
            return Signal(symbol=symbol, type=SignalType.HOLD)
        close = float(candles["close"].iloc[-1])
        if self.params["direction"] == "buy":
            return Signal(
                symbol=symbol, type=SignalType.BUY,
                entry_price=close, stop_loss=close - 10, take_profit=close + 20,
            )
        return Signal(
            symbol=symbol, type=SignalType.SELL,
            entry_price=close, stop_loss=close + 10, take_profit=close - 20,
        )


def make_walkforward():
    return WalkForward(
        strategy_factory=lambda p: DirectionalStrategy(p),
        risk_factory=lambda: RiskManager(
            RiskConfig(
                capital=100_000.0,
                max_risk_per_trade_pct=1.0,
                max_daily_loss_pct=100.0,
                max_open_positions=1,
            )
        ),
        warmup=10,
    )


def test_param_grid_expande_combinacoes():
    combos = list(param_grid({"a": [1, 2], "b": ["x"]}))
    assert combos == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


def test_optimize_escolhe_a_direcao_lucrativa():
    # Tendência de alta consistente: comprar lucra, vender perde
    candles = make_candles([1000.0 + i * 2 for i in range(200)])
    wf = make_walkforward()
    best_params, best_result = wf.optimize(
        "WIN", candles, {"direction": ["buy", "sell"]}
    )
    assert best_params == {"direction": "buy"}
    assert best_result.total_pnl > 0


def test_walkforward_gera_janelas_e_resultado_oos():
    candles = make_candles([1000.0 + i * 2 for i in range(400)])
    wf = make_walkforward()
    report = wf.run(
        "WIN", candles, {"direction": ["buy", "sell"]},
        train_bars=150, test_bars=100,
    )

    # 400 candles, treino 150 + teste 100, passo 100 → 2 janelas
    assert len(report.windows) == 2
    for window in report.windows:
        assert window.best_params == {"direction": "buy"}
        assert window.test_end > window.test_start
    # Em alta constante, o out-of-sample agregado deve ser lucrativo
    assert report.oos_pnl > 0
    assert "Janelas: 2" in report.summary()
