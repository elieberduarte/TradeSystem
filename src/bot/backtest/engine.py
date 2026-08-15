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
    # Barras decorridas desde a entrada (para a saída por tempo)
    bars_held: int = 0
    # Distância original do stop, base dos gatilhos de gestão
    initial_risk: float = 0.0
    # Resultado já realizado numa saída parcial
    realized_pnl: float = 0.0
    partial_done: bool = False
    # Melhor fechamento a favor desde a entrada (base da saída por
    # retração de Fibonacci)
    peak_close: float = 0.0
    # O gatilho da entrada, escrito pela estratégia no momento do sinal
    entry_reason: str = ""


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

    @property
    def expectancy(self) -> float:
        """Resultado médio por trade — o número que multiplica pela frequência."""
        return self.total_pnl / len(self.trades) if self.trades else 0.0

    @property
    def longest_losing_streak(self) -> int:
        """Maior sequência de trades perdedores — o que testa a paciência."""
        longest = streak = 0
        for trade in self.trades:
            streak = streak + 1 if trade.pnl < 0 else 0
            longest = max(longest, streak)
        return longest

    @property
    def calmar(self) -> float:
        """Lucro dividido pelo pior mergulho: retorno por unidade de dor."""
        return self.total_pnl / self.max_drawdown if self.max_drawdown > 0 else float("inf")

    @property
    def pnl_std(self) -> float:
        """Desvio-padrão do resultado por trade — dispersão dos resultados."""
        if len(self.trades) < 2:
            return 0.0
        mean = self.expectancy
        variance = sum((t.pnl - mean) ** 2 for t in self.trades) / (len(self.trades) - 1)
        return variance**0.5

    @property
    def trade_sharpe(self) -> float:
        """Expectativa sobre dispersão: quão consistente é o resultado."""
        return self.expectancy / self.pnl_std if self.pnl_std > 0 else 0.0

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
        # Candles passados à estratégia a cada barra (limita custo de CPU
        # em históricos longos; 500 x 5min cobre ~2 pregões)
        lookback: int = 500,
        # Pontos perdidos em cada execução A MERCADO (entrada, stop, zeragem).
        # Saída no alvo é ordem limitada — sem slippage.
        slippage_points: float = 0.0,
        # Custo fixo em R$ por contrato na ida e volta (emolumentos etc.)
        cost_per_contract: float = 0.0,
        # Saída por tempo: fecha a posição após N barras (0 = desativado).
        # Evidência recorrente: sinais que seguram 12–15 barras superam a
        # fricção; os que saem em 1–6 barras não.
        max_holding_bars: int = 0,
        # Stop móvel: após o preço andar N vezes o risco a favor, o stop
        # passa a seguir a máxima/mínima a essa distância (0 = desativado).
        trailing_atr: float = 0.0,
        # Move o stop para o preço de entrada quando o lucro atinge N
        # vezes o risco (0 = desativado). Elimina a perda, ao custo de
        # ser estopado no zero em movimentos que voltariam.
        breakeven_at: float = 0.0,
        # Realiza metade da posição a N vezes o risco (0 = desativado).
        partial_at: float = 0.0,
        # Saída por falha de Fibonacci (Eykyn): fechamento que devolve
        # esta fração do swing entrada→melhor fechamento encerra a
        # posição (0 = desativado; o livro usa 0.618).
        fib_exit: float = 0.0,
        # Custo em caixa de UMA unidade. None = preço de entrada (ações e
        # ETFs, pagos integralmente). Nos futuros, a margem por contrato.
        # Só tem efeito com RiskConfig.enforce_cash ligado.
        unit_cost: float | None = None,
    ):
        self.strategy = strategy
        self.risk = risk
        self.point_value = point_value
        self.warmup = warmup
        self.lookback = lookback
        self.slippage_points = slippage_points
        self.cost_per_contract = cost_per_contract
        self.max_holding_bars = max_holding_bars
        self.trailing_atr = trailing_atr
        self.breakeven_at = breakeven_at
        self.partial_at = partial_at
        self.fib_exit = fib_exit
        self.unit_cost = unit_cost

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
                open_trade.bars_held += 1
                open_trade, equity = self._check_exit(open_trade, candle, ts, result, equity)

            # A gestão vem DEPOIS das saídas: ela lê o fechamento do
            # candle, então só pode valer a partir do próximo. Aplicá-la
            # antes deixaria o stop móvel ser comparado com a abertura da
            # mesma barra, que aconteceu antes do fechamento que o moveu.
            if open_trade is not None:
                open_trade, equity = self._manage(open_trade, candle, ts, result, equity)

            # Saída por tempo: a posição já teve as barras que o setup previa
            if (
                open_trade is not None
                and self.max_holding_bars
                and open_trade.bars_held >= self.max_holding_bars
            ):
                open_trade, equity = self._close(
                    open_trade, float(candle["close"]), ts, "tempo", result, equity
                )

            window_start = max(0, i + 1 - self.lookback)
            signal = self.strategy.generate_signal(symbol, candles.iloc[window_start : i + 1])

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
        quantity = self.risk.position_size(
            signal.entry_price, signal.stop_loss, self.point_value, self.unit_cost
        )
        # Contratos são negociados em quantidades inteiras
        quantity = int(quantity)
        if quantity <= 0:
            return None
        self.risk.open_positions_count += 1
        side = "buy" if signal.type == SignalType.BUY else "sell"
        # Entrada a mercado: paga slippage contra a direção da ordem
        entry = signal.entry_price + (
            self.slippage_points if side == "buy" else -self.slippage_points
        )
        return Trade(
            symbol=signal.symbol,
            side=side,
            entry_time=ts,
            entry_price=entry,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            initial_risk=abs(entry - signal.stop_loss),
            entry_reason=signal.reason,
        )

    def _manage(self, trade: Trade, candle, ts, result, equity: float):
        """Aplica saída parcial, breakeven, stop móvel e saída de Fibonacci.

        Todos usam o fechamento do candle como referência — nunca a
        máxima/mínima, porque dentro da barra não se sabe a ordem dos
        preços e usar o extremo seria olhar o futuro.
        """
        if trade.initial_risk <= 0:
            return trade, equity
        direction = 1 if trade.side == "buy" else -1
        close = float(candle["close"])
        progress = direction * (close - trade.entry_price) / trade.initial_risk

        # Realização parcial: metade da posição sai, o resto segue
        if self.partial_at and not trade.partial_done and progress >= self.partial_at:
            half = trade.quantity // 2
            if half >= 1:
                price = close - direction * self.slippage_points
                gross = direction * (price - trade.entry_price) * half * self.point_value
                realized = gross - self.cost_per_contract * half
                trade.realized_pnl += realized
                trade.quantity -= half
                self.risk.register_trade_result(realized)
                equity += realized
            trade.partial_done = True

        # Breakeven: elimina a perda depois de um avanço mínimo
        if self.breakeven_at and progress >= self.breakeven_at:
            if direction * (trade.entry_price - trade.stop_loss) > 0:
                trade.stop_loss = trade.entry_price

        # Stop móvel: acompanha o preço a uma distância fixa
        if self.trailing_atr and progress >= self.trailing_atr:
            trail = close - direction * self.trailing_atr * trade.initial_risk
            if direction * (trail - trade.stop_loss) > 0:
                trade.stop_loss = trail

        # Falha de Fibonacci: devolver mais que a fração do swing a
        # favor (entrada → melhor fechamento) encerra no fechamento
        if self.fib_exit:
            if trade.peak_close == 0.0:
                trade.peak_close = trade.entry_price
            if direction * (close - trade.peak_close) > 0:
                trade.peak_close = close
            swing = direction * (trade.peak_close - trade.entry_price)
            retraced = direction * (trade.peak_close - close)
            if swing > 0 and retraced >= self.fib_exit * swing:
                return self._close(trade, close, ts, "fib", result, equity)

        return trade, equity

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
        direction = 1 if trade.side == "buy" else -1
        # Só a saída no alvo é ordem limitada; o resto executa a mercado
        # e paga slippage contra a posição
        if reason != "alvo":
            price -= direction * self.slippage_points
        trade.exit_time = ts
        trade.exit_price = price
        trade.exit_reason = reason
        gross = direction * (price - trade.entry_price) * trade.quantity * self.point_value
        # O resultado do trade inclui o que já saiu na parcial
        trade.pnl = gross - self.cost_per_contract * trade.quantity + trade.realized_pnl
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
