"""Baixa o histórico diário do universo ampliado."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.data.history import HistoryStore
from src.bot.execution.mt5_broker import MT5Broker
from src.bot.universe import BLOCKS


def main() -> None:
    broker = MT5Broker()
    broker.connect()
    store = HistoryStore()

    for block, symbols in BLOCKS.items():
        print(f"\n── {block} ──")
        for symbol in symbols:
            try:
                merged = store.update_from_broker(broker, symbol, "1d", limit=99_999)
                print(f"  {symbol:<10} {len(merged):>5} candles "
                      f"({merged.index.min().date()} → {merged.index.max().date()})")
            except Exception as exc:  # noqa: BLE001
                print(f"  {symbol:<10} FALHOU — {exc}")

    broker.disconnect()


if __name__ == "__main__":
    main()
