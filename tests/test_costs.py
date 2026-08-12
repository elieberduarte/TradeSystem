"""Testes de slippage e custos no motor de backtest."""

from src.bot.backtest.engine import BacktestEngine
from src.bot.risk.manager import RiskConfig, RiskManager
from tests.conftest import make_candles
from tests.test_backtest_engine import BuyOnceStrategy


def engine_with_costs(strategy, slippage=0.0, cost=0.0):
    risk = RiskManager(
        RiskConfig(
            capital=100_000.0,
            max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=50.0,
            max_open_positions=1,
        )
    )
    return BacktestEngine(
        strategy, risk, warmup=5, slippage_points=slippage, cost_per_contract=cost
    )


def test_slippage_piora_entrada_e_stop():
    closes = [1000.0] * 10 + [998, 996, 994]
    engine = engine_with_costs(
        BuyOnceStrategy(at=10, entry=1000.0, stop=995.0, target=1010.0), slippage=2.0
    )
    result = engine.run("WIN", make_candles(closes))

    trade = result.trades[0]
    assert trade.entry_price == 1002.0  # comprou 2 pts pior
    assert trade.exit_price == 993.0    # stop saiu 2 pts pior
    assert trade.exit_reason == "stop"


def test_alvo_e_ordem_limitada_sem_slippage():
    closes = [1000.0] * 10 + [1002, 1004, 1006, 1008, 1012]
    engine = engine_with_costs(
        BuyOnceStrategy(at=10, entry=1000.0, stop=995.0, target=1010.0), slippage=2.0
    )
    result = engine.run("WIN", make_candles(closes))

    trade = result.trades[0]
    assert trade.entry_price == 1002.0  # entrada a mercado paga slippage
    assert trade.exit_price == 1010.0   # alvo limitado não paga


def test_custo_por_contrato_e_descontado():
    closes = [1000.0] * 10 + [1002, 1004, 1006, 1008, 1012]
    engine = engine_with_costs(
        BuyOnceStrategy(at=10, entry=1000.0, stop=995.0, target=1010.0), cost=1.5
    )
    result = engine.run("WIN", make_candles(closes))

    trade = result.trades[0]
    gross = (1010.0 - 1000.0) * trade.quantity
    assert trade.pnl == gross - 1.5 * trade.quantity
