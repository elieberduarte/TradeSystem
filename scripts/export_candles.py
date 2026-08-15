"""Exporta o histórico diário das ações do painel de fundamentos.

Um JSON por ticker em web/candles/ (carregado sob demanda pelo
painel — nunca tudo de uma vez). Semanal e mensal são recompostos
do diário no navegador, então um arquivo serve os três períodos.

Uso: python scripts/export_candles.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MetaTrader5 as mt5
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "candles"


def main() -> None:
    data = json.loads((ROOT / "web" / "fundamentals.json").read_text(encoding="utf-8"))
    tickers = [p["ticker"] for p in data["papeis"]]

    if not mt5.initialize():
        raise SystemExit(f"MT5 indisponível: {mt5.last_error()}")

    OUT.mkdir(parents=True, exist_ok=True)
    exported = 0
    for ticker in tickers:
        mt5.symbol_select(ticker, True)
        rates = mt5.copy_rates_from_pos(ticker, mt5.TIMEFRAME_D1, 0, 99_999)
        if rates is None or len(rates) < 30:
            print(f"  {ticker}: sem histórico")
            continue
        frame = pd.DataFrame(rates)
        frame["d"] = pd.to_datetime(frame["time"], unit="s").dt.strftime("%Y-%m-%d")
        candles = [
            {"d": r["d"], "o": round(float(r["open"]), 2),
             "h": round(float(r["high"]), 2), "l": round(float(r["low"]), 2),
             "c": round(float(r["close"]), 2)}
            for _, r in frame.iterrows()
        ]
        (OUT / f"{ticker}.json").write_text(
            json.dumps(candles, separators=(",", ":")), encoding="utf-8"
        )
        exported += 1
    mt5.shutdown()
    print(f"{exported}/{len(tickers)} tickers exportados para {OUT}")


if __name__ == "__main__":
    main()
