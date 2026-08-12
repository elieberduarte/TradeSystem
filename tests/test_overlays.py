"""Testes das camadas de gestão e das variações de saída."""

from datetime import time

import pandas as pd

from src.bot.backtest.engine import BacktestEngine
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.base import BaseStrategy, Signal, SignalType
from src.bot.strategies.overlays import (
    NoTargetOverlay,
    RewardRatioOverlay,
    StopWidthOverlay,
)
from tests.conftest import make_candles


class FixedBuy(BaseStrategy):
    mode = "swing_trade"

    def generate_signal(self, symbol, candles):
        return Signal(
            symbol=symbol, type=SignalType.BUY, entry_price=1000.0,
            stop_loss=990.0, take_profit=1020.0,
        )


# ────────────────────────────── overlays ──────────────────────────────

def test_reward_ratio_reescala_so_o_alvo():
    signal = RewardRatioOverlay(FixedBuy(), ratio=3.0).generate_signal("WIN", make_candles([1000.0]))
    assert signal.stop_loss == 990.0          # stop intacto
    assert signal.take_profit == 1030.0       # 3x o risco de 10


def test_reward_ratio_invertido_encurta_o_alvo():
    signal = RewardRatioOverlay(FixedBuy(), ratio=0.33).generate_signal("WIN", make_candles([1000.0]))
    assert signal.take_profit == 1000.0 + 0.33 * 10


def test_stop_width_preserva_a_relacao_alvo_risco():
    base = FixedBuy().generate_signal("WIN", make_candles([1000.0]))
    ratio_base = (base.take_profit - base.entry_price) / (base.entry_price - base.stop_loss)

    wide = StopWidthOverlay(FixedBuy(), factor=2.0).generate_signal("WIN", make_candles([1000.0]))
    ratio_wide = (wide.take_profit - wide.entry_price) / (wide.entry_price - wide.stop_loss)

    assert wide.stop_loss == 980.0            # stop dobrado
    assert abs(ratio_wide - ratio_base) < 1e-9


def test_no_target_afasta_o_alvo_para_longe():
    signal = NoTargetOverlay(FixedBuy(), far_multiple=100.0).generate_signal("WIN", make_candles([1000.0]))
    assert signal.take_profit == 1000.0 + 100 * 10


def test_overlay_preserva_o_modo():
    assert RewardRatioOverlay(FixedBuy(), ratio=2.0).mode == "swing_trade"


# ───────────────────────── gestão dentro do motor ─────────────────────────

class BuyOnce(BaseStrategy):
    def __init__(self, at: int):
        super().__init__()
        self.at = at

    def generate_signal(self, symbol, candles):
        if len(candles) == self.at:
            return Signal(
                symbol=symbol, type=SignalType.BUY, entry_price=1000.0,
                stop_loss=990.0, take_profit=1100.0,
            )
        return Signal(symbol=symbol, type=SignalType.HOLD)


def engine(**kwargs):
    risk = RiskManager(
        RiskConfig(
            capital=100_000.0, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=100.0, max_open_positions=1,
            trading_start=time(0, 0), trading_end=time(23, 59),
            max_consecutive_losses=0,
        )
    )
    defaults = dict(warmup=5, slippage_points=0.0, cost_per_contract=0.0)
    return BacktestEngine(BuyOnce(at=6), risk, **{**defaults, **kwargs})


def test_breakeven_move_o_stop_para_a_entrada():
    # Sobe 2x o risco (1020), depois desaba: sai no zero, não em -10
    closes = [1000.0] * 6 + [1020.0, 1020.0] + [980.0] * 4
    result = engine(breakeven_at=2.0).run("WIN", make_candles(closes))

    trade = result.trades[0]
    assert trade.stop_loss == 1000.0
    assert trade.exit_reason in ("stop", "stop (gap)")


def test_sem_breakeven_a_perda_e_cheia():
    closes = [1000.0] * 6 + [1020.0, 1020.0] + [980.0] * 4
    result = engine().run("WIN", make_candles(closes))
    assert result.trades[0].stop_loss == 990.0


def test_trailing_acompanha_o_preco():
    closes = [1000.0] * 6 + [1010.0, 1030.0, 1050.0] + [1000.0] * 4
    result = engine(trailing_atr=1.0).run("WIN", make_candles(closes))

    trade = result.trades[0]
    # O stop subiu bem acima do original de 990
    assert trade.stop_loss > 1020.0
    assert trade.pnl > 0  # saiu no lucro, não no prejuízo


def test_parcial_realiza_metade_e_segue():
    # Sobe 2x o risco (realiza metade a 1020), depois volta e estopa a 990
    closes = [1000.0] * 6 + [1020.0] * 2 + [990.0] * 4
    result = engine(partial_at=2.0).run("WIN", make_candles(closes))

    trade = result.trades[0]
    assert trade.partial_done
    assert trade.quantity == 50           # metade dos 100 originais
    assert trade.realized_pnl == 1000.0   # 50 x 20 pts de lucro
    # O total soma a parcial ganha com a perda da metade restante
    assert trade.pnl == 1000.0 - 500.0


def test_parcial_reduz_a_perda_de_um_trade_que_vira():
    closes = [1000.0] * 6 + [1020.0] * 2 + [990.0] * 4
    com_parcial = engine(partial_at=2.0).run("WIN", make_candles(closes))
    sem_parcial = engine().run("WIN", make_candles(closes))

    assert com_parcial.trades[0].pnl > sem_parcial.trades[0].pnl


def test_parcial_nao_dispara_sem_avanco():
    closes = [1000.0] * 6 + [1002.0] * 6
    result = engine(partial_at=2.0).run("WIN", make_candles(closes))
    assert not result.trades[0].partial_done
