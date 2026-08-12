"""Testes do critério de otimização resistente a sorte."""

from src.bot.backtest.engine import BacktestResult, Trade
from src.bot.backtest.walkforward import robust_score


def result_with(pnls: list[float], equity: list[float]) -> BacktestResult:
    trades = [
        Trade(
            symbol="WIN", side="buy", entry_time=None, entry_price=0,
            quantity=1, stop_loss=0, take_profit=0, pnl=pnl,
        )
        for pnl in pnls
    ]
    return BacktestResult(trades=trades, equity_curve=equity)


def test_rejeita_amostra_pequena():
    # Lucro alto, mas só 3 trades: não é evidência
    lucky = result_with([500.0, 400.0, 300.0], [0, 500, 900, 1200])
    assert robust_score(lucky) == float("-inf")


def test_prefere_lucro_estavel_ao_lucro_com_susto():
    pnls = [100.0] * 12
    steady = result_with(pnls, [0, 100, 200, 300, 400, 500, 600])
    # Mesmo lucro final, mas com um mergulho pelo caminho
    bumpy = result_with(pnls, [0, 300, -200, 100, 400, 300, 600])
    assert robust_score(steady) > robust_score(bumpy)


def test_prejuizo_pontua_negativo():
    losing = result_with([-50.0] * 12, [0, -100, -200, -300])
    assert robust_score(losing) < 0


def test_min_trades_configuravel():
    r = result_with([10.0] * 5, [0, 10, 20, 30, 40, 50])
    assert robust_score(r, min_trades=10) == float("-inf")
    assert robust_score(r, min_trades=3) > 0
