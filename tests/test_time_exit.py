"""Testes da saída por tempo (holding máximo em barras)."""

from src.bot.backtest.engine import BacktestEngine
from src.bot.risk.manager import RiskConfig, RiskManager
from tests.conftest import make_candles
from tests.test_backtest_engine import BuyOnceStrategy


def engine_with_time_exit(bars: int):
    risk = RiskManager(
        RiskConfig(
            capital=100_000.0,
            max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=50.0,
            max_open_positions=1,
        )
    )
    # Stop e alvo longe: só a saída por tempo pode fechar
    return BacktestEngine(
        BuyOnceStrategy(at=10, entry=1000.0, stop=900.0, target=1100.0),
        risk, warmup=5, max_holding_bars=bars,
    )


def test_fecha_apos_o_numero_de_barras():
    candles = make_candles([1000.0] * 30)
    result = engine_with_time_exit(13).run("WIN", candles)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "tempo"
    assert trade.bars_held == 13


def test_sem_limite_a_posicao_segue_ate_o_fim():
    candles = make_candles([1000.0] * 30)
    result = engine_with_time_exit(0).run("WIN", candles)
    assert result.trades[0].exit_reason == "fim do histórico"


def test_stop_tem_prioridade_sobre_a_saida_por_tempo():
    # Preço despenca no 2º candle após a entrada: stop bate antes do tempo
    closes = [1000.0] * 11 + [800.0] + [1000.0] * 20
    risk = RiskManager(
        RiskConfig(
            capital=100_000.0, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=50.0, max_open_positions=1,
        )
    )
    engine = BacktestEngine(
        BuyOnceStrategy(at=10, entry=1000.0, stop=990.0, target=1100.0),
        risk, warmup=5, max_holding_bars=13,
    )
    result = engine.run("WIN", make_candles(closes))
    assert result.trades[0].exit_reason in ("stop", "stop (gap)")
