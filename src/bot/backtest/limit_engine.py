"""Motor de backtest com execução por ordem limitada.

Scalping só tem margem do lado passivo do book: entrar e sair com
ordem limitada evita cruzar o spread, que é a maior parte da fricção.
Mas isso traz um problema que os backtests de scalping normalmente
escondem — a SELEÇÃO ADVERSA.

Uma ordem limitada de compra abaixo do mercado só é executada quando o
preço CAI até ela. Ou seja: você é preenchido justamente nas vezes em
que o mercado estava indo contra. As ordens que "teriam dado certo"
(preço subiu direto) simplesmente não executam. Assumir que toda
limitada é preenchida infla o resultado de forma grosseira.

Modelagem adotada, deliberadamente conservadora:

1. A limitada só é preenchida se o preço NEGOCIAR ALÉM dela
   (`low < preço` na compra), não apenas tocar. Tocar significa fila:
   pode não ter havido negócio suficiente para chegar na sua ordem.
2. Ordem não preenchida em `limit_timeout_bars` é cancelada.
3. O alvo também é ordem limitada — mesma regra de "negociar além".
4. O stop sai a mercado e paga o spread. Não existe stop passivo.
"""

from dataclasses import dataclass

import pandas as pd

from src.bot.backtest.engine import BacktestResult, Trade
from src.bot.risk.manager import RiskManager
from src.bot.strategies.base import BaseStrategy, SignalType


@dataclass
class PendingOrder:
    side: str
    limit_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    placed_at: pd.Timestamp
    bars_waiting: int = 0


class LimitOrderEngine:
    def __init__(
        self,
        strategy: BaseStrategy,
        risk: RiskManager,
        point_value: float = 1.0,
        warmup: int = 100,
        lookback: int = 300,
        # Quão longe do preço atual a ordem é colocada (em pontos).
        # Maior = mais barato quando executa, menos execuções.
        limit_offset: float = 10.0,
        # Barras de espera antes de cancelar a ordem não executada
        limit_timeout_bars: int = 3,
        # Fricção do stop, que sai a mercado (spread + agressão)
        stop_slippage: float = 5.0,
        # Emolumentos e liquidação em R$ por contrato, ida e volta
        cost_per_contract: float = 1.5,
        # Fecha a posição após N barras (scalp não carrega posição)
        max_holding_bars: int = 15,
    ):
        self.strategy = strategy
        self.risk = risk
        self.point_value = point_value
        self.warmup = warmup
        self.lookback = lookback
        self.limit_offset = limit_offset
        self.limit_timeout_bars = limit_timeout_bars
        self.stop_slippage = stop_slippage
        self.cost_per_contract = cost_per_contract
        self.max_holding_bars = max_holding_bars
        # Diagnóstico: quantas ordens morreram sem execução
        self.orders_placed = 0
        self.orders_filled = 0
        self.orders_expired = 0

    @property
    def fill_rate(self) -> float:
        return self.orders_filled / self.orders_placed if self.orders_placed else 0.0

    def run(self, symbol: str, candles: pd.DataFrame) -> BacktestResult:
        result = BacktestResult()
        pending: PendingOrder | None = None
        trade: Trade | None = None
        equity = 0.0
        current_day = None

        for i in range(self.warmup, len(candles)):
            candle = candles.iloc[i]
            ts = candles.index[i]
            high, low = float(candle["high"]), float(candle["low"])

            if current_day != ts.date():
                current_day = ts.date()
                self.risk.reset_day()

            # 1. Posição aberta: verifica saídas antes de qualquer coisa
            if trade is not None:
                trade.bars_held += 1
                trade, equity = self._check_exit(trade, high, low, ts, result, equity)

            if trade is not None and trade.bars_held >= self.max_holding_bars:
                trade, equity = self._close(
                    trade, float(candle["close"]), ts, "tempo", result, equity, market=True
                )

            # 2. Ordem pendente: executa se o preço negociou ALÉM dela
            if pending is not None and trade is None:
                filled = (
                    low < pending.limit_price
                    if pending.side == "buy"
                    else high > pending.limit_price
                )
                if filled:
                    self.orders_filled += 1
                    self.risk.open_positions_count += 1
                    trade = Trade(
                        symbol=symbol, side=pending.side, entry_time=ts,
                        entry_price=pending.limit_price, quantity=pending.quantity,
                        stop_loss=pending.stop_loss, take_profit=pending.take_profit,
                    )
                    pending = None
                else:
                    pending.bars_waiting += 1
                    if pending.bars_waiting >= self.limit_timeout_bars:
                        self.orders_expired += 1
                        pending = None

            # 3. Novo sinal só quando não há posição nem ordem viva
            if trade is None and pending is None:
                window = candles.iloc[max(0, i + 1 - self.lookback) : i + 1]
                signal = self.strategy.generate_signal(symbol, window)
                if signal.type in (SignalType.BUY, SignalType.SELL):
                    pending = self._place(signal, ts)

            result.equity_curve.append(
                equity + (self._unrealized(trade, float(candle["close"])) if trade else 0.0)
            )

        if trade is not None:
            self._close(
                trade, float(candles["close"].iloc[-1]), candles.index[-1],
                "fim do histórico", result, equity, market=True,
            )
        return result

    # ---------------------------------------------------------------- internos

    def _place(self, signal, ts) -> PendingOrder | None:
        allowed, _ = self.risk.can_open_position(now=ts.to_pydatetime())
        if not allowed:
            return None

        side = "buy" if signal.type == SignalType.BUY else "sell"
        # A limitada fica ATRÁS do preço: compra abaixo, venda acima
        limit = (
            signal.entry_price - self.limit_offset
            if side == "buy"
            else signal.entry_price + self.limit_offset
        )
        # Stop e alvo acompanham o preço real de entrada
        shift = limit - signal.entry_price
        stop = signal.stop_loss + shift
        target = signal.take_profit + shift

        quantity = int(self.risk.position_size(limit, stop, self.point_value))
        if quantity <= 0:
            return None

        self.orders_placed += 1
        return PendingOrder(
            side=side, limit_price=limit, stop_loss=stop,
            take_profit=target, quantity=quantity, placed_at=ts,
        )

    def _check_exit(self, trade: Trade, high: float, low: float, ts, result, equity):
        if trade.side == "buy":
            # Stop tem prioridade: no mesmo candle, assume o pior caso
            if low <= trade.stop_loss:
                return self._close(trade, trade.stop_loss, ts, "stop", result, equity, market=True)
            # Alvo é limitada: exige negociar ALÉM do preço
            if high > trade.take_profit:
                return self._close(trade, trade.take_profit, ts, "alvo", result, equity, market=False)
        else:
            if high >= trade.stop_loss:
                return self._close(trade, trade.stop_loss, ts, "stop", result, equity, market=True)
            if low < trade.take_profit:
                return self._close(trade, trade.take_profit, ts, "alvo", result, equity, market=False)
        return trade, equity

    def _close(self, trade: Trade, price: float, ts, reason: str, result, equity, market: bool):
        direction = 1 if trade.side == "buy" else -1
        if market:
            price -= direction * self.stop_slippage
        trade.exit_time = ts
        trade.exit_price = price
        trade.exit_reason = reason
        gross = direction * (price - trade.entry_price) * trade.quantity * self.point_value
        trade.pnl = gross - self.cost_per_contract * trade.quantity
        result.trades.append(trade)
        self.risk.register_trade_result(trade.pnl)
        self.risk.open_positions_count -= 1
        return None, equity + trade.pnl

    def _unrealized(self, trade: Trade, price: float) -> float:
        direction = 1 if trade.side == "buy" else -1
        return direction * (price - trade.entry_price) * trade.quantity * self.point_value
