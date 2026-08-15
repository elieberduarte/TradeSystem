"""Armazenamento local de candles históricos em Parquet.

Cada download do MT5 é mesclado ao acervo local (`data/`), que só cresce
com o tempo — os backtests leem daqui, sem depender do servidor.
"""

from pathlib import Path

import pandas as pd

from src.bot.execution.broker import BrokerInterface


class HistoryStore:
    def __init__(self, root: Path | str = "data"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, symbol: str, timeframe: str) -> Path:
        return self.root / f"{symbol.upper()}_{timeframe}.parquet"

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        path = self.path(symbol, timeframe)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def save(self, symbol: str, timeframe: str, candles: pd.DataFrame) -> pd.DataFrame:
        """Mescla os candles novos com o acervo existente e persiste.

        Em timestamps duplicados, o dado novo prevalece (candles antigos
        podem ter sido parciais no momento do download).
        """
        existing = self.load(symbol, timeframe)
        if existing is not None:
            merged = pd.concat([existing, candles])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = candles.sort_index()
        merged.to_parquet(self.path(symbol, timeframe))
        return merged

    def update_from_broker(
        self,
        broker: BrokerInterface,
        symbol: str,
        timeframe: str,
        limit: int = 20_000,
    ) -> pd.DataFrame:
        """Baixa o máximo de histórico do broker e mescla ao acervo local."""
        candles = broker.get_candles(symbol, timeframe, limit)
        return self.save(symbol, timeframe, candles)
