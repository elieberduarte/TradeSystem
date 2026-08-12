"""Testes do motor de execução com ordem limitada."""

from datetime import time

import pandas as pd

from src.bot.backtest.limit_engine import LimitOrderEngine
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.base import BaseStrategy, Signal, SignalType


def minute_candles(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Cada linha: (open, high, low, close), em candles de 1 minuto."""
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [100.0] * len(rows),
        },
        index=pd.date_range("2026-08-11 10:00", periods=len(rows), freq="1min"),
    )


class BuyAt(BaseStrategy):
    """Emite compra no candle `at`, com preço de referência fixo."""

    def __init__(self, at: int, price: float, stop: float, target: float):
        super().__init__()
        self.at, self.price, self.stop, self.target = at, price, stop, target

    def generate_signal(self, symbol, candles):
        if len(candles) == self.at:
            return Signal(
                symbol=symbol, type=SignalType.BUY, entry_price=self.price,
                stop_loss=self.stop, take_profit=self.target,
            )
        return Signal(symbol=symbol, type=SignalType.HOLD)


def engine(strategy, **kwargs):
    risk = RiskManager(
        RiskConfig(
            capital=100_000.0, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=100.0, max_open_positions=1,
            trading_start=time(0, 0), trading_end=time(23, 59),
            max_consecutive_losses=0,
        )
    )
    defaults = dict(
        warmup=3, limit_offset=10.0, limit_timeout_bars=3,
        stop_slippage=5.0, cost_per_contract=0.0, max_holding_bars=50,
    )
    return LimitOrderEngine(strategy, risk, **{**defaults, **kwargs})


FLAT = (1000.0, 1000.0, 1000.0, 1000.0)

# Com warmup=3, a primeira barra avaliada é a de índice 3, onde a janela
# tem 4 candles. Logo `at=4` dispara o sinal ali, e a barra de índice 4 é
# a primeira chance de execução da limitada.
SIGNAL_AT = 4


def buyer(price=1000.0, stop=960.0, target=1020.0) -> BuyAt:
    return BuyAt(at=SIGNAL_AT, price=price, stop=stop, target=target)


def test_limitada_nao_executa_se_o_preco_apenas_tocar():
    # Limitada de compra em 990; mínima exatamente 990 não basta (fila do book)
    rows = [FLAT] * 4 + [(1000.0, 1000.0, 990.0, 995.0)] * 4
    eng = engine(buyer())
    result = eng.run("WIN", minute_candles(rows))

    assert eng.orders_placed == 1
    assert eng.orders_filled == 0
    assert not result.trades


def test_limitada_executa_quando_o_preco_negocia_alem():
    rows = [FLAT] * 4 + [(1000.0, 1000.0, 985.0, 995.0)] + [FLAT] * 6
    eng = engine(buyer())
    result = eng.run("WIN", minute_candles(rows))

    assert eng.orders_filled == 1
    # Preenchida no preço da limitada, não no preço do sinal
    assert result.trades[0].entry_price == 990.0


def test_ordem_expira_sem_execucao():
    rows = [FLAT] * 4 + [(1000.0, 1010.0, 999.0, 1008.0)] * 6
    eng = engine(buyer(), limit_timeout_bars=2)
    eng.run("WIN", minute_candles(rows))

    assert eng.orders_expired == 1
    assert eng.fill_rate == 0.0


def test_stop_paga_slippage_e_alvo_nao():
    # Executa em 990; stop deslocado para 950, sai a 945 com slippage
    rows = [FLAT] * 4 + [(1000.0, 1000.0, 985.0, 995.0)]
    rows += [(990.0, 990.0, 940.0, 945.0)] + [FLAT] * 4
    eng = engine(buyer())
    result = eng.run("WIN", minute_candles(rows))

    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_price == 945.0


def test_alvo_exige_negociar_alem_do_preco():
    # Executa em 990, alvo deslocado para 1010: máxima igual a 1010 não
    # fecha; só quando negocia acima
    rows = [FLAT] * 4 + [(1000.0, 1000.0, 985.0, 995.0)]
    rows += [(990.0, 1010.0, 990.0, 1005.0)] * 2
    rows += [(1005.0, 1015.0, 1005.0, 1012.0)] + [FLAT] * 3
    eng = engine(buyer())
    result = eng.run("WIN", minute_candles(rows))

    trade = result.trades[0]
    assert trade.exit_reason == "alvo"
    assert trade.exit_price == 1010.0  # ordem limitada: sem slippage


def test_taxa_de_execucao_e_contabilizada():
    rows = [FLAT] * 4 + [(1000.0, 1000.0, 985.0, 995.0)] + [FLAT] * 10
    eng = engine(buyer())
    eng.run("WIN", minute_candles(rows))
    assert eng.fill_rate == 1.0


def test_saida_por_tempo_paga_slippage():
    rows = [FLAT] * 4 + [(1000.0, 1000.0, 985.0, 995.0)] + [FLAT] * 8
    eng = engine(buyer(stop=900.0, target=1100.0), max_holding_bars=3)
    result = eng.run("WIN", minute_candles(rows))

    trade = result.trades[0]
    assert trade.exit_reason == "tempo"
    assert trade.exit_price == 995.0  # fechamento 1000 − 5 de slippage
