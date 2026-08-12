"""Baixa o histórico de candles do MT5 e alimenta o acervo local.

Uso:
    python scripts/download_history.py                # símbolos padrão
    python scripts/download_history.py WIN$N 15m      # símbolo/timeframe específico
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.data.history import HistoryStore
from src.bot.execution.mt5_broker import MT5Broker

# Contratos vigentes + séries contínuas (emendam vencimentos — backtest longo)
DEFAULT_SYMBOLS = ["WINQ26", "WDOU26", "WIN$N", "WDO$N"]
DEFAULT_TIMEFRAMES = ["5m", "15m", "1d"]


def main() -> None:
    args = sys.argv[1:]
    symbols = [args[0]] if args else DEFAULT_SYMBOLS
    timeframes = [args[1]] if len(args) > 1 else DEFAULT_TIMEFRAMES

    broker = MT5Broker()
    broker.connect()
    store = HistoryStore()

    for symbol in symbols:
        for timeframe in timeframes:
            try:
                merged = store.update_from_broker(broker, symbol, timeframe, limit=99_999)
                print(
                    f"{symbol} {timeframe}: {len(merged)} candles no acervo "
                    f"({merged.index.min()} → {merged.index.max()})"
                )
            except Exception as exc:  # noqa: BLE001 — seguir para o próximo símbolo
                print(f"{symbol} {timeframe}: FALHOU — {exc}")

    broker.disconnect()


if __name__ == "__main__":
    main()
