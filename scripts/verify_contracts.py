"""Confere os valores de ponto e margem contra o MetaTrader 5.

Todos os backtests de futuros usam FUT_POINT_VALUE e FUT_MARGIN de
`universe.py`, que até aqui eram estimativas. O sizing depende
diretamente deles: valor de ponto errado dimensiona errado, margem
errada distribui as vagas errado.

A fonte da verdade é a especificação do contrato no terminal:
  valor do ponto = trade_tick_value / trade_tick_size
  margem         = order_calc_margin(compra de 1 contrato ao preço atual)

Uso: python scripts/verify_contracts.py   (terminal MT5 aberto e logado)
"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MetaTrader5 as mt5

from src.bot.data.contracts import resolve_symbol
from src.bot.universe import FUT_MARGIN, FUT_POINT_VALUE, FUTUROS

ROOT = Path(__file__).resolve().parents[1]


def inspect(symbol: str) -> dict | None:
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    if not info.visible:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)

    tick = mt5.symbol_info_tick(symbol)
    price = tick.last or tick.bid or tick.ask if tick else 0.0

    point_value = (
        info.trade_tick_value / info.trade_tick_size if info.trade_tick_size else None
    )

    margin = None
    if price:
        margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, 1.0, price)
    if not margin:
        margin = info.margin_initial or None

    return {
        "symbol": symbol,
        "descricao": info.description,
        "preco": price,
        "tick_size": info.trade_tick_size,
        "tick_value": info.trade_tick_value,
        "point_value": point_value,
        "margin": margin,
        "contract_size": info.trade_contract_size,
    }


def main() -> None:
    if not mt5.initialize():
        print(f"MT5 indisponível: {mt5.last_error()}")
        print("Abra o terminal MetaTrader 5 (XP demo) e rode de novo.")
        sys.exit(1)

    account = mt5.account_info()
    print(f"Conectado: conta {account.login} · {account.server} · "
          f"saldo {account.balance:,.0f} {account.currency}\n")

    print(f"{'símbolo':<8} {'preço':>10} {'pt estimado':>12} {'pt REAL':>10} {'razão':>7} "
          f"{'mg estimada':>12} {'mg REAL':>10} {'razão':>7}")
    print("-" * 84)

    report = []
    for symbol in FUTUROS:
        spec = inspect(symbol)
        if spec is None:
            # $N pode não ter spec completa; tenta o contrato vigente
            front = resolve_symbol(symbol.replace("$N", ""), date.today())
            spec = inspect(front)
            if spec is None:
                print(f"{symbol:<8} — símbolo não encontrado")
                continue
            spec["symbol"] = f"{symbol} ({front})"

        est_pt = FUT_POINT_VALUE.get(symbol)
        est_mg = FUT_MARGIN.get(symbol)
        real_pt, real_mg = spec["point_value"], spec["margin"]

        ratio_pt = real_pt / est_pt if real_pt and est_pt else None
        ratio_mg = real_mg / est_mg if real_mg and est_mg else None
        flag = " ⚠️" if (ratio_pt and abs(ratio_pt - 1) > 0.2) or \
                        (ratio_mg and abs(ratio_mg - 1) > 0.5) else ""

        report.append({**spec, "estimado_pt": est_pt, "estimado_mg": est_mg})
        print(f"{spec['symbol']:<8} {spec['preco']:>10,.2f} {est_pt:>12} "
              f"{real_pt if real_pt else '—':>10} "
              f"{f'{ratio_pt:.2f}x' if ratio_pt else '—':>7} "
              f"{est_mg:>12,.0f} "
              f"{f'{real_mg:,.0f}' if real_mg else '—':>10} "
              f"{f'{ratio_mg:.2f}x' if ratio_mg else '—':>7}{flag}")

    mt5.shutdown()

    out = ROOT / "web" / "contract_specs.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    print(f"\nExportado para {out}")
    print("⚠️ = divergência relevante (ponto >20% ou margem >50%) — corrigir universe.py")


if __name__ == "__main__":
    main()
