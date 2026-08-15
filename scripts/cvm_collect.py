"""Primeira coleta CVM: eventos das nossas 13 ações, 2021-2026.

Baixa FCA (mapa ticker↔CNPJ) e IPE (fatos relevantes/comunicados),
salva o acervo em Parquet e mostra o resumo por categoria — o
inventário de matéria-prima do estudo de eventos.

Uso: python scripts/cvm_collect.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.data.cvm import events_for_tickers
from src.bot.universe import ACOES_BR

ROOT = Path(__file__).resolve().parents[1]
YEARS = range(2021, 2027)


def main() -> None:
    print(f"Coletando FCA + IPE {YEARS.start}-{YEARS.stop - 1} para {len(ACOES_BR)} ações…")
    events = events_for_tickers(ACOES_BR, YEARS)

    out = ROOT / "data" / "cvm" / "eventos.parquet"
    events.to_parquet(out)
    print(f"{len(events):,} eventos salvos em {out}\n")

    print("── Eventos por categoria ──")
    counts = events["categoria"].value_counts()
    for category, count in counts.head(12).items():
        print(f"  {category:<55} {count:>6,}")

    fatos = events[events["categoria"].str.contains("Fato Relevante", case=False, na=False)]
    print(f"\n── Fatos Relevantes por ação ──")
    for ticker, count in fatos["ticker"].value_counts().items():
        print(f"  {ticker:<8} {count:>4}")

    print("\nÚltimos 5 fatos relevantes do acervo:")
    for _, row in fatos.tail(5).iterrows():
        print(f"  {row['data_entrega'].date()} {row['ticker']:<7} {str(row['assunto'])[:70]}")


if __name__ == "__main__":
    main()
