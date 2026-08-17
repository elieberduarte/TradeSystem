"""Mapa de interesse ao vivo: topos/fundos, redondos e 50% do WIN e WDO.

Lê o acervo intraday (5m) e o diário, detecta os pivôs pelo zigzag
causal com limiar em fração do range típico, e imprime os níveis
mais próximos do preço nos dois horizontes.

Uso: python scripts/interest_map.py [WIN$N|WDO$N] [--json]
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.analysis.interest_map import build_map, render
from src.bot.data.history import HistoryStore

ROOT = Path(__file__).resolve().parents[1]


def typical_range(daily, days: int = 20) -> float:
    return float((daily["high"] - daily["low"]).tail(days).mean())


def main() -> None:
    symbols = [a for a in sys.argv[1:] if not a.startswith("--")] or ["WIN$N", "WDO$N"]
    as_json = "--json" in sys.argv
    store = HistoryStore()
    payload = {}

    for symbol in symbols:
        daily = store.load(symbol, "1d")
        intraday = store.load(symbol, "5m")
        rng = typical_range(daily)
        price = float(intraday["close"].iloc[-1])

        # Intraday: pivôs da sessão corrente + últimas, limiar 15% do range típico
        session_days = 3
        last_days = intraday[intraday.index.normalize() >= intraday.index.normalize().unique()[-session_days]]
        intra_levels, intra_pivots = build_map(last_days, symbol, threshold=0.15 * rng,
                                               lookback_pivots=8, round_span=1.5 * rng)
        # Diário: pivôs de swing, limiar 1,5× o range típico
        daily_levels, daily_pivots = build_map(daily, symbol, threshold=1.5 * rng,
                                               lookback_pivots=6, round_span=4 * rng)

        if as_json:
            payload[symbol] = {
                "preco": price, "range_tipico": rng,
                "intraday": [asdict(l) for l in intra_levels],
                "diario": [asdict(l) for l in daily_levels],
            }
            continue
        print(f"\n╔══ {symbol} · range típico {rng:,.0f} · {len(intra_pivots)} pivôs intraday "
              f"({session_days} sessões) · {len(daily_pivots)} pivôs diários ══")
        print("INTRADAY (5m, limiar 15% do range)")
        print(render(symbol, price, intra_levels))
        print("\nDIÁRIO (swing, limiar 1,5× range)")
        print(render(symbol, price, daily_levels))

    if as_json:
        out = ROOT / "web" / "interest_map.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Exportado para {out}")


if __name__ == "__main__":
    main()
