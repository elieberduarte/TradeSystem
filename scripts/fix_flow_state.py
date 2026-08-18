"""Reprocessa o acervo de fluxo já gravado: classifica o estado do book.

Os arquivos anteriores à correção não têm a coluna `estado` e calculam
imbalance também em leilão (book cruzado por construção). Este script
deriva o estado de bid1/ask1 — que já estão gravados — e anula o
imbalance onde ele não se aplica. Idempotente: rodar duas vezes não
muda nada.

Uso: python scripts/fix_flow_state.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.bot.data.flow_recorder import classify_book

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    total = {"continuo": 0, "leilao": 0, "cruzado": 0}
    for path in sorted((ROOT / "data" / "flow").glob("*/*_book.parquet")):
        frame = pd.read_parquet(path)
        frame["estado"] = [classify_book(b, a) for b, a in zip(frame.bid1, frame.ask1)]
        fora = frame["estado"] != "continuo"
        frame.loc[fora, ["imb_l1", "imb_lk"]] = float("nan")
        frame.to_parquet(path)
        counts = frame["estado"].value_counts().to_dict()
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
        print(f"{path.parent.name}/{path.name}: " +
              " · ".join(f"{k} {v:,}" for k, v in counts.items()))
    print("\nacervo inteiro: " + " · ".join(f"{k} {v:,}" for k, v in total.items()))
    descartado = total["leilao"] + total["cruzado"]
    print(f"snapshots com imbalance válido: {total['continuo']:,} "
          f"({total['continuo']/(total['continuo']+descartado):.1%})")


if __name__ == "__main__":
    main()
