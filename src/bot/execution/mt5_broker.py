"""Conector MetaTrader 5 para futuros da B3 (WIN/WDO).

Requer o terminal MetaTrader 5 instalado e logado numa conta de corretora
que ofereça B3 (XP, Clear, Rico, etc.), e o pacote `MetaTrader5` do pip.
"""

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None

from src.bot.execution.broker import BrokerInterface, Order, Position

TIMEFRAME_MAP = {
    "1m": lambda: mt5.TIMEFRAME_M1,
    "5m": lambda: mt5.TIMEFRAME_M5,
    "15m": lambda: mt5.TIMEFRAME_M15,
    "30m": lambda: mt5.TIMEFRAME_M30,
    "1h": lambda: mt5.TIMEFRAME_H1,
    "4h": lambda: mt5.TIMEFRAME_H4,
    "1d": lambda: mt5.TIMEFRAME_D1,
}


class MT5Broker(BrokerInterface):
    def __init__(
        self,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        terminal_path: str | None = None,
        magic: int = 20260810,
    ):
        if mt5 is None:
            raise ImportError(
                "Pacote MetaTrader5 não instalado. Rode: pip install MetaTrader5"
            )
        self.login = login
        self.password = password
        self.server = server
        self.terminal_path = terminal_path
        # Identificador das ordens deste bot no terminal
        self.magic = magic

    # ------------------------------------------------------------------ conexão

    def connect(self) -> None:
        kwargs = {}
        if self.terminal_path:
            kwargs["path"] = self.terminal_path
        if self.login:
            kwargs.update(login=self.login, password=self.password, server=self.server)
        if not mt5.initialize(**kwargs):
            raise ConnectionError(f"Falha ao conectar ao MetaTrader 5: {mt5.last_error()}")

    def disconnect(self) -> None:
        mt5.shutdown()

    # -------------------------------------------------------------------- dados

    def get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"Timeframe inválido: {timeframe} (use {list(TIMEFRAME_MAP)})")
        self._ensure_symbol(symbol)
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe](), 0, limit)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Sem candles para {symbol}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        # B3 fornece volume financeiro em real_volume; tick_volume é fallback
        volume = df["real_volume"].where(df["real_volume"] > 0, df["tick_volume"])
        return pd.DataFrame(
            {
                "open": df["open"],
                "high": df["high"],
                "low": df["low"],
                "close": df["close"],
                "volume": volume,
            }
        )

    # ------------------------------------------------------------------- ordens

    def place_order(self, order: Order) -> str:
        self._ensure_symbol(order.symbol)
        is_market = order.price is None
        order_type = self._order_type(order.side, is_market, order)

        request = {
            "action": mt5.TRADE_ACTION_DEAL if is_market else mt5.TRADE_ACTION_PENDING,
            "symbol": order.symbol,
            "volume": float(order.quantity),
            "type": order_type,
            "magic": self.magic,
            "comment": "trade-bot",
            # B3 opera com preenchimento RETURN (parcial permitido)
            "type_filling": mt5.ORDER_FILLING_RETURN,
            "type_time": mt5.ORDER_TIME_DAY,
        }
        if is_market:
            tick = mt5.symbol_info_tick(order.symbol)
            request["price"] = tick.ask if order.side == "buy" else tick.bid
            request["deviation"] = 5
        else:
            request["price"] = float(order.price)
        if order.stop_loss:
            request["sl"] = float(order.stop_loss)
        if order.take_profit:
            request["tp"] = float(order.take_profit)

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else mt5.last_error()
            raise RuntimeError(f"Ordem rejeitada ({code}): {getattr(result, 'comment', '')}")
        return str(result.order)

    def close_position(self, symbol: str) -> None:
        positions = mt5.positions_get(symbol=symbol) or []
        for pos in positions:
            side = "sell" if pos.type == mt5.POSITION_TYPE_BUY else "buy"
            tick = mt5.symbol_info_tick(symbol)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if side == "sell" else mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": tick.bid if side == "sell" else tick.ask,
                "deviation": 5,
                "magic": self.magic,
                "comment": "trade-bot close",
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                code = result.retcode if result else mt5.last_error()
                raise RuntimeError(f"Falha ao fechar posição {pos.ticket} ({code})")

    def get_open_positions(self) -> list[Position]:
        positions = mt5.positions_get() or []
        return [
            Position(
                symbol=p.symbol,
                side="buy" if p.type == mt5.POSITION_TYPE_BUY else "sell",
                quantity=p.volume,
                entry_price=p.price_open,
                unrealized_pnl=p.profit,
            )
            for p in positions
        ]

    def get_balance(self) -> float:
        info = mt5.account_info()
        if info is None:
            raise ConnectionError(f"Sem informação de conta: {mt5.last_error()}")
        return info.balance

    # ---------------------------------------------------------------- internos

    def _ensure_symbol(self, symbol: str) -> None:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Símbolo {symbol} não existe na corretora")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Não foi possível ativar o símbolo {symbol}")

    @staticmethod
    def _order_type(side: str, is_market: bool, order: Order):
        if is_market:
            return mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        return mt5.ORDER_TYPE_BUY_LIMIT if side == "buy" else mt5.ORDER_TYPE_SELL_LIMIT
