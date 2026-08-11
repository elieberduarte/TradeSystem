"""Testes do motor de backtest."""

from src.bot.backtest.engine import BacktestEngine
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.base import BaseStrategy, Signal, SignalType
from tests.conftest import make_candles


class BuyOnceStrategy(BaseStrategy):
    """Compra no candle `at` com stop/alvo fixos; depois fica em HOLD."""

    def __init__(self, at: int, entry: float, stop: float, target: float):
        super().__init__()
        self.at = at
        self.entry, self.stop, self.target = entry, stop, target

    def generate_signal(self, symbol, candles):
        if len(candles) == self.at:
            return Signal(
                symbol=symbol,
                type=SignalType.BUY,
                entry_price=self.entry,
                stop_loss=self.stop,
                take_profit=self.target,
            )
        return Signal(symbol=symbol, type=SignalType.HOLD)


def make_engine(strategy, point_value=1.0):
    risk = RiskManager(
        RiskConfig(
            capital=100_000.0,
            max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=50.0,
            max_open_positions=1,
        )
    )
    return BacktestEngine(strategy, risk, point_value=point_value, warmup=5)


def test_alvo_atingido_gera_lucro():
    # Preço sobe direto: entra a 1000, alvo 1010 é atingido
    closes = [1000.0] * 10 + [1002, 1004, 1006, 1008, 1012]
    candles = make_candles(closes)
    engine = make_engine(BuyOnceStrategy(at=10, entry=1000.0, stop=995.0, target=1010.0))
    result = engine.run("WIN", candles)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "alvo"
    assert trade.exit_price == 1010.0
    # Risco de 1% de 100k = 1000; stop de 5 pontos → 200 contratos
    assert trade.quantity == 200
    assert trade.pnl == (1010.0 - 1000.0) * 200


def test_stop_atingido_gera_perda_limitada():
    closes = [1000.0] * 10 + [998, 996, 994]
    candles = make_candles(closes)
    engine = make_engine(BuyOnceStrategy(at=10, entry=1000.0, stop=995.0, target=1010.0))
    result = engine.run("WIN", candles)

    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    # Perda = exatamente o risco configurado (1% do capital)
    assert trade.pnl == (995.0 - 1000.0) * 200 == -1000.0


def test_stop_tem_prioridade_sobre_alvo_no_mesmo_candle():
    # Candle largo que atinge stop e alvo: assume o pior caso (stop).
    # O sinal sai no candle 10; a saída é avaliada a partir do candle 11.
    closes = [1000.0] * 10 + [1000.0, 1000.0]
    candles = make_candles(closes)
    candles.loc[candles.index[11], "high"] = 1020.0
    candles.loc[candles.index[11], "low"] = 990.0
    engine = make_engine(BuyOnceStrategy(at=10, entry=1000.0, stop=995.0, target=1010.0))
    result = engine.run("WIN", candles)

    assert result.trades[0].exit_reason == "stop"


def test_posicao_aberta_fecha_no_fim_do_historico():
    closes = [1000.0] * 10 + [1001.0, 1002.0]
    candles = make_candles(closes)
    engine = make_engine(BuyOnceStrategy(at=10, entry=1000.0, stop=990.0, target=1050.0))
    result = engine.run("WIN", candles)

    assert result.trades[0].exit_reason == "fim do histórico"


def test_metricas_do_resultado():
    closes = [1000.0] * 10 + [1002, 1004, 1006, 1008, 1012]
    candles = make_candles(closes)
    engine = make_engine(BuyOnceStrategy(at=10, entry=1000.0, stop=995.0, target=1010.0))
    result = engine.run("WIN", candles)

    assert result.win_rate == 1.0
    assert result.total_pnl == 2000.0
    assert result.profit_factor == float("inf")
    assert "Trades: 1" in result.summary()
