"""Testes das métricas de risco do resultado de backtest."""

from src.bot.backtest.engine import BacktestResult, Trade


def result_with(pnls: list[float], equity: list[float] | None = None) -> BacktestResult:
    trades = [
        Trade(symbol="WIN", side="buy", entry_time=None, entry_price=0,
              quantity=1, stop_loss=0, take_profit=0, pnl=pnl)
        for pnl in pnls
    ]
    if equity is None:
        equity, cum = [], 0.0
        for pnl in pnls:
            cum += pnl
            equity.append(cum)
    return BacktestResult(trades=trades, equity_curve=equity)


def test_expectancy_e_a_media_por_trade():
    assert result_with([100.0, -50.0, 30.0]).expectancy == 80.0 / 3


def test_pior_sequencia_de_perdas():
    r = result_with([10.0, -5.0, -5.0, -5.0, 20.0, -5.0, -5.0])
    assert r.longest_losing_streak == 3


def test_calmar_penaliza_mergulho():
    # Mesmo lucro final; a segunda sofre um mergulho no meio
    steady = result_with([50.0] * 4)
    bumpy = result_with([200.0, -150.0, 100.0, 50.0])
    assert steady.calmar > bumpy.calmar


def test_consistencia_premia_resultados_parecidos():
    steady = result_with([48.0, 52.0, 49.0, 51.0])
    erratic = result_with([400.0, -100.0, -100.0, 0.0])
    # Mesma média (50), dispersões muito diferentes
    assert steady.trade_sharpe > erratic.trade_sharpe


def test_consistencia_zero_quando_nao_ha_dispersao():
    # Guarda contra divisão por zero: resultados idênticos não pontuam
    assert result_with([50.0, 50.0, 50.0]).trade_sharpe == 0.0


def test_metricas_vazias_sem_trades():
    empty = BacktestResult()
    assert empty.expectancy == 0.0
    assert empty.longest_losing_streak == 0
    assert empty.pnl_std == 0.0
