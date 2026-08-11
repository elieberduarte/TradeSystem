"""Motor de backtest candle a candle.

Percorre o histórico simulando o fluxo real do bot: a cada candle fechado
a estratégia opina, o risk manager valida e dimensiona, e a execução é
simulada com regras conservadoras:

- Entrada no fechamento do candle do sinal.
- Stop e alvo verificados contra high/low dos candles seguintes; se os
  dois caberiam no mesmo candle, assume-se o STOP (pior caso).
- Sinal contrário fecha a posição no fechamento do candle.
- PnL diário alimenta o risk manager (que pode vetar novas entradas).
"""

from dataclasses import dataclass, field

import pandas as pd

from src.bot.risk.manager import RiskManager
from src.bot.strategies.base import BaseStrategy, SignalType


@dataclass
class Trade:
    symbol: str
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    pnl: float = 0.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gains = sum(t.pnl for t in self.trades if t.pnl > 0)
        losses = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gains / losses if losses else float("inf")

    @property
    def max_drawdown(self) -> float:
        peak, max_dd = float("-inf"), 0.0
        for value in self.equity_curve:
            peak = max(peak, value)
            max_dd = max(max_dd, peak - value)
        return max_dd

    def summary(self) -> str:
        return (
            f"Trades: {len(self.trades)} | Win rate: {self.win_rate:.1%} | "
            f"PnL: {self.total_pnl:.2f} | Profit factor: {self.profit_factor:.2f} | "
            f"Max drawdown: {self.max_drawdown:.2f}"
        )


class BacktestEngine:
    def __init__(
        self,
        strategy: BaseStrategy,
        risk: RiskManager,
        # Valor financeiro de 1 ponto por contrato (WIN: 0.20; WDO: 10.00)
        point_value: float = 1.0,
        warmup: int = 100,
    ):
        self.strategy = strategy
        self.risk = risk
        self.point_value = point_value
        self.warmup = warmup

    def run(self, symbol: str, candles: pd.DataFrame) -> BacktestResult:
        result = BacktestResult()
        open_trade: Trade | None = None
        equity = 0.0
        current_day = None
        current_week = None

        for i in range(self.warmup, len(candles)):
            candle = candles.iloc[i]
            ts = candles.index[i]

            if current_day != ts.date():
                current_day = ts.date()
                self.risk.reset_day()
            week = ts.isocalendar()[:2]
            if current_week != week:
                current_week = week
                self.risk.reset_week()

            if open_trade is not None and self.risk.should_flatten(ts.to_pydatetime()):
                open_trade, equity = self._close(
                    open_trade, float(candle["open"]), ts, "zeragem diária", result, equity
                )

            if open_trade is not None:
                open_trade, equity = self._check_exit(open_trade, candle, ts, result, equity)

            signal = self.strategy.generate_signal(symbol, candles.iloc[: i + 1])

            if open_trade is not None and self._is_opposite(signal.type, open_trade.side):
                open_trade, equity = self._close(
                    open_trade, float(candle["close"]), ts, "sinal contrário", result, equity
                )

            if open_trade is None and signal.type in (SignalType.BUY, SignalType.SELL):
                open_trade = self._try_open(signal, ts)

            result.equity_curve.append(
                equity + (self._unrealized(open_trade, float(candle["close"])) if open_trade else 0.0)
            )

        if open_trade is not None:
            last = candles.iloc[-1]
            self._close(
                open_trade, float(last["close"]), candles.index[-1], "fim do histórico", result, equity
            )
        return result

    # ---------------------------------------------------------------- internos

    def _try_open(self, signal, ts) -> Trade | None:
        allowed, _ = self.risk.can_open_position(now=ts.to_pydatetime())
        if not allowed:
            return None
        quantity = self.risk.position_size(signal.entry_price, signal.stop_loss)
        # Contratos são negociados em quantidades inteiras
        quantity = int(quantity)
        if quantity <= 0:
            return None
        self.risk.open_positions_count += 1
        return Trade(
            symbol=signal.symbol,
            side="buy" if signal.type == SignalType.BUY else "sell",
            entry_time=ts,
            entry_price=signal.entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    def _check_exit(self, trade: Trade, candle, ts, result, equity):
        """Stop/alvo contra o candle. Gap de abertura que pula o nível sai
        pelo preço de abertura — no swing, o gap contra é pior que o stop
        nominal, e o backtest precisa refletir isso."""
        open_ = float(candle["open"])
        low, high = float(candle["low"]), float(candle["high"])
        if trade.side == "buy":
            if open_ <= trade.stop_loss:
                return self._close(trade, open_, ts, "stop (gap)", result, equity)
            if low <= trade.stop_loss:
                return self._close(trade, trade.stop_loss, ts, "stop", result, equity)
            if open_ >= trade.take_profit:
                return self._close(trade, open_, ts, "alvo (gap)", result, equity)
            if high >= trade.take_profit:
                return self._close(trade, trade.take_profit, ts, "alvo", result, equity)
        else:
            if open_ >= trade.stop_loss:
                return self._close(trade, open_, ts, "stop (gap)", result, equity)
            if high >= trade.stop_loss:
                return self._close(trade, trade.stop_loss, ts, "stop", result, equity)
            if open_ <= trade.take_profit:
                return self._close(trade, open_, ts, "alvo (gap)", result, equity)
            if low <= trade.take_profit:
                return self._close(trade, trade.take_profit, ts, "alvo", result, equity)
        return trade, equity

    def _close(self, trade: Trade, price: float, ts, reason: str, result, equity):
        trade.exit_time = ts
        trade.exit_price = price
        trade.exit_reason = reason
        direction = 1 if trade.side == "buy" else -1
        trade.pnl = direction * (price - trade.entry_price) * trade.quantity * self.point_value
        result.trades.append(trade)
        self.risk.register_trade_result(trade.pnl)
        self.risk.open_positions_count -= 1
        return None, equity + trade.pnl

    def _unrealized(self, trade: Trade, price: float) -> float:
        direction = 1 if trade.side == "buy" else -1
        return direction * (price - trade.entry_price) * trade.quantity * self.point_value

    @staticmethod
    def _is_opposite(signal_type: SignalType, side: str) -> bool:
        return (signal_type == SignalType.SELL and side == "buy") or (
            signal_type == SignalType.BUY and side == "sell"
        )
