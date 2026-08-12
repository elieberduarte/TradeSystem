"""Baixa o histórico diário de todos os instrumentos do universo de teste."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.data.history import HistoryStore
from src.bot.execution.mt5_broker import MT5Broker

UNIVERSE = [
    "WIN$N", "WDO$N", "IND$N", "DOL$N",
    "BOVA11", "SMAL11", "IVVB11",
    "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "B3SA3", "WEGE3", "PRIO3",
]


def main() -> None:
    broker = MT5Broker()
    broker.connect()
    store = HistoryStore()

    for symbol in UNIVERSE:
        try:
            merged = store.update_from_broker(broker, symbol, "1d", limit=99_999)
            print(f"{symbol}: {len(merged)} candles ({merged.index.min().date()} → {merged.index.max().date()})")
        except Exception as exc:  # noqa: BLE001
            print(f"{symbol}: FALHOU — {exc}")

    broker.disconnect()


if __name__ == "__main__":
    main()
