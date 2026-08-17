"""Acumula o fluxo oficial dos players (BDI da B3) no acervo local.

Baixa tudo que a API retém (~21 pregões) e mescla ao Parquet — cada
execução diária estende o histórico. Mostra o saldo do estrangeiro
no à vista e os contratos em aberto dos nossos futuros.

Uso: python scripts/b3_flow_collect.py
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.data.b3_bdi import (
    collect_open_interest,
    collect_participacao,
    daily_flow,
    workdays,
)


def main() -> None:
    days = sorted(workdays(str(date.today())))
    print(f"API retém {len(days)} pregões: {days[0]} → {days[-1]}")

    participacao = collect_participacao(days)
    total_days = participacao["data"].nunique()
    print(f"\nParticipação por investidor: {len(participacao)} linhas, "
          f"{total_days} pregões no acervo")

    flow = daily_flow(participacao)
    gringo = flow[flow["categoria"] == "Investidor Estrangeiro"].copy()
    gringo["saldo_mi"] = gringo["saldo_dia_mil"] / 1_000
    print("\nSaldo DIÁRIO do estrangeiro no à vista (R$ mi) — derivado do acumulado mensal da B3:")
    for _, row in gringo.tail(12).iterrows():
        if row["saldo_mi"] != row["saldo_mi"]:      # NaN = buraco no acervo
            print(f"  {row['data']}  (buraco no acervo)")
            continue
        bar = "█" * min(int(abs(row["saldo_mi"]) / 200), 30)
        sign = "+" if row["saldo_mi"] >= 0 else "−"
        print(f"  {row['data']}  {sign}{abs(row['saldo_mi']):>8,.0f}  {bar}")

    oi = collect_open_interest(days)
    print(f"\nContratos em aberto (futuros da carteira): {len(oi)} linhas")
    latest = oi[oi["data"] == oi["data"].max()]
    for _, row in latest.iterrows():
        print(f"  {row['data']}  {row['mercado'][:44]:<44} {row['contratos_abertos']:>12,}")

    print("\nAcervo em data/b3/*.parquet — rode diariamente (junto com o bot) "
          "para o histórico crescer além dos 21 dias da B3.")


if __name__ == "__main__":
    main()
