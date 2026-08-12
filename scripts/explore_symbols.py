"""Descobre o que a corretora oferece: instrumentos e profundidade de histórico.

Mais instrumentos = mais testes de replicação (o que reprovou nossa
carteira no WDO). Mais histórico = mais janelas de validação.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MetaTrader5 as mt5
import pandas as pd

CANDIDATES = [
    # Futuros
    "WIN$N", "WDO$N", "IND$N", "DOL$N", "BIT$N",
    # ETFs e ações líquidas — replicação em outro tipo de ativo
    "BOVA11", "SMAL11", "IVVB11",
    "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "B3SA3", "WEGE3", "PRIO3",
]


def main() -> None:
    if not mt5.initialize():
        raise SystemExit(f"MT5 não conectou: {mt5.last_error()}")

    print(f"{'símbolo':<10} {'candles 1d':>11} {'início':>12} {'fim':>12}")
    print("-" * 50)
    available = []
    for symbol in CANDIDATES:
        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"{symbol:<10} {'não existe':>11}")
            continue
        if not info.visible:
            mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_range(
            symbol, mt5.TIMEFRAME_D1, datetime(2000, 1, 1), datetime.now()
        )
        if rates is None or len(rates) == 0:
            print(f"{symbol:<10} {'sem dados':>11}")
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        print(
            f"{symbol:<10} {len(df):>11} {df.time.min().date()!s:>12} {df.time.max().date()!s:>12}"
        )
        available.append(symbol)

    print(f"\nDisponíveis: {available}")
    mt5.shutdown()


if __name__ == "__main__":
    main()
