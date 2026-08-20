"""Reprocessa o acervo de fluxo: estado do book/ticks e base de tempo.

Três correções, todas derivadas dos dados brutos já gravados (nada se
perde):

  1. `estado` em cada snapshot de book (continuo/leilao/cruzado) e o
     imbalance anulado fora do contínuo.
  2. `estado` em cada tick — em leilão o `volume_real` é o ACUMULADO do
     leilão, repetido a cada leitura, e não pode entrar no CVD.
  3. Base de tempo dos ticks: o servidor carimba o horário de Brasília
     COMO SE fosse UTC, deixando a série 3h fora da do book. Aqui o
     carimbo original vira `ts_server_ms` e `ts_ms` passa a ser o epoch
     UTC real — a mesma régua do book.

Idempotente: rodar duas vezes não muda nada.

Uso: python scripts/fix_flow_state.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.bot.data.flow_recorder import classify_book

ROOT = Path(__file__).resolve().parents[1]
FUSO = "America/Sao_Paulo"


def corrigir_book(path: Path) -> dict:
    frame = pd.read_parquet(path)
    frame["estado"] = [classify_book(b, a) for b, a in zip(frame.bid1, frame.ask1)]
    frame.loc[frame["estado"] != "continuo", ["imb_l1", "imb_lk"]] = float("nan")
    frame.to_parquet(path)
    return frame["estado"].value_counts().to_dict()


def corrigir_ticks(path: Path) -> dict:
    frame = pd.read_parquet(path)
    frame["estado"] = [classify_book(b, a) for b, a in zip(frame.bid, frame.ask)]
    if "ts_server_ms" not in frame.columns:
        frame["ts_server_ms"] = frame["ts_ms"]
    # O carimbo do servidor é horário de Brasília rotulado como UTC:
    # localiza no fuso certo e converte para epoch UTC de verdade. A
    # subtração da época em Timedelta é independente da unidade interna
    # do datetime (ns/us/ms) — `astype("int64")` não é, e foi o que
    # corrompeu a primeira versão desta correção.
    local = pd.to_datetime(frame["ts_server_ms"], unit="ms").dt.tz_localize(FUSO)
    utc = local.dt.tz_convert("UTC").dt.tz_localize(None)
    frame["ts_ms"] = ((utc - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")).astype("int64")
    frame.to_parquet(path)
    return frame["estado"].value_counts().to_dict()


def main() -> None:
    total = {}
    for path in sorted((ROOT / "data" / "flow").glob("*/*.parquet")):
        kind = "book" if path.name.endswith("_book.parquet") else "ticks"
        counts = corrigir_book(path) if kind == "book" else corrigir_ticks(path)
        for k, v in counts.items():
            total[f"{kind}/{k}"] = total.get(f"{kind}/{k}", 0) + v
        print(f"{path.parent.name}/{path.name}: " +
              " · ".join(f"{k} {v:,}" for k, v in counts.items()))
    print("\nacervo inteiro:")
    for k, v in sorted(total.items()):
        print(f"  {k:<18} {v:>8,}")


if __name__ == "__main__":
    main()
