"""Interface abstrata de corretora/exchange.

Implementações concretas (MetaTrader 5, Binance via ccxt, etc.) devem
herdar de BrokerInterface, mantendo estratégias e risco independentes
da corretora escolhida.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    quantity: float
    # None = ordem a mercado
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class Position:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    unrealized_pnl: float
    # Stop e alvo anexados à posição na corretora (0.0 = sem)
    stop_loss: float = 0.0
    take_profit: float = 0.0


class BrokerInterface(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Retorna candles com colunas open, high, low, close, volume."""

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Envia a ordem e retorna o id gerado pela corretora."""

    @abstractmethod
    def close_position(self, symbol: str) -> None: ...

    @abstractmethod
    def get_open_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_balance(self) -> float: ...

    # Métodos com implementação padrão — corretoras que souberem mais,
    # sobrescrevem; as demais continuam funcionando.

    def is_demo(self) -> bool:
        """Conta de demonstração? O runner recusa conta real sem opt-in."""
        return False

    def last_price(self, symbol: str) -> float:
        """Último preço negociado (para ancorar stops em rolagens)."""
        raise NotImplementedError

    def front_contract(self, root: str, min_days: int = 3) -> str:
        """Contrato vigente de um futuro (ex.: 'WIN' → 'WINV26')."""
        raise NotImplementedError

    def realized_pnl(self, since_days: int) -> float:
        """Resultado realizado pelas ordens DESTE bot nos últimos N dias."""
        return 0.0
