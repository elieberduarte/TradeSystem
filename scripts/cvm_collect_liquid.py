"""Amplia a coleta de eventos CVM para todos os papéis líquidos com preço.

Reaproveita o cache dos zips (FCA/IPE já baixados); só o cruzamento
CNPJ↔ticker cresce. Salva data/cvm/eventos_liquid.parquet.

Uso: python scripts/cvm_collect_liquid.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.data.cvm import events_for_tickers

ROOT = Path(__file__).resolve().parents[1]
YEARS = range(2021, 2027)


def main() -> None:
    liquid = json.loads((ROOT / "data" / "liquid_universe.json").read_text(encoding="utf-8"))["symbols"]
    # ações e units só (BDR/ETF não têm IPE de companhia aberta brasileira útil aqui)
    tickers = [s for s in liquid if not s[-2:] in ("32", "33", "34", "35", "39")]
    print(f"Cruzando IPE com {len(tickers)} tickers…")
    events = events_for_tickers(tickers, YEARS)
    out = ROOT / "data" / "cvm" / "eventos_liquid.parquet"
    events.to_parquet(out)
    print(f"{len(events):,} eventos · {events['ticker'].nunique()} tickers com CNPJ no FCA")
    for cat in ("Fato Relevante", "Dados Econômico-Financeiros", "Documentos de Oferta"):
        n = int(events["categoria"].str.contains(cat, case=False, na=False).sum())
        print(f"  {cat:<32} {n:>7,}")


if __name__ == "__main__":
    main()
