"""Censo do mercado à vista: quantos papéis existem e quantos são operáveis?

Varre todos os símbolos do MT5 (XP), classifica por tipo de ticker e
mede a liquidez real — mediana do volume financeiro diário dos
últimos 60 pregões. Responde três perguntas:

  1. Quantos papéis são negociados na B3 (pela janela da XP)?
  2. Quantos já passaram pelo nosso teste de replicação?
  3. Quantos têm liquidez para a nossa carteira?

Critério de liquidez pré-declarado: a posição não pode passar de
0,5% do volume mediano diário (entrar e sair sem mover o preço).
Com vagas de ~R$ 10 mil (capital 100k / 10 vagas), isso exige
mediana ≥ R$ 2 milhões/dia; adotamos R$ 5 mi como corte confortável.

Uso: python scripts/universe_scan.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MetaTrader5 as mt5
import numpy as np

from src.bot.universe import EXPANDED

ROOT = Path(__file__).resolve().parents[1]

ACAO = re.compile(r"^[A-Z]{4}(3|4|5|6)$")          # ON, PN, PNA, PNB
UNIT11 = re.compile(r"^[A-Z]{4}11$")                # units, ETFs e FIIs
BDR = re.compile(r"^[A-Z]{4}3[2-9]$")               # recibos de exterior


def classify(name: str) -> str | None:
    if ACAO.match(name):
        return "ação"
    if UNIT11.match(name):
        return "unit/ETF/FII"
    if BDR.match(name):
        return "BDR"
    return None


def median_daily_volume(symbol: str) -> float | None:
    """Mediana do volume financeiro diário (R$) nos últimos 60 pregões."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 60)
    if rates is None or len(rates) < 20:
        return None
    shares = np.where(rates["real_volume"] > 0, rates["real_volume"], rates["tick_volume"])
    financial = shares * rates["close"]
    return float(np.median(financial))


def main() -> None:
    if not mt5.initialize():
        raise SystemExit(f"MT5 indisponível: {mt5.last_error()}")

    rows = []
    skipped = 0
    for info in mt5.symbols_get() or []:
        kind = classify(info.name)
        if kind is None:
            continue
        volume = median_daily_volume(info.name)
        if volume is None:
            skipped += 1
            continue
        rows.append({"symbol": info.name, "tipo": kind,
                     "volume_mediano": round(volume, 0),
                     "descricao": info.description})
    mt5.shutdown()

    tested = {s for s in EXPANDED if classify(s)}
    print(f"═══ Censo do mercado à vista · janela da XP · {len(rows)} papéis com histórico ═══")
    print(f"(+{skipped} tickers sem candles suficientes — recém-listados ou sem negócio)\n")

    print(f"{'tipo':<14} {'papéis':>7} {'≥R$50M/dia':>11} {'10–50M':>8} {'1–10M':>7} {'<1M':>6}")
    print("-" * 60)
    for kind in ("ação", "unit/ETF/FII", "BDR"):
        group = [r for r in rows if r["tipo"] == kind]
        buckets = [
            sum(1 for r in group if r["volume_mediano"] >= 50e6),
            sum(1 for r in group if 10e6 <= r["volume_mediano"] < 50e6),
            sum(1 for r in group if 1e6 <= r["volume_mediano"] < 10e6),
            sum(1 for r in group if r["volume_mediano"] < 1e6),
        ]
        print(f"{kind:<14} {len(group):>7} {buckets[0]:>11} {buckets[1]:>8} "
              f"{buckets[2]:>7} {buckets[3]:>6}")

    liquid = [r for r in rows if r["volume_mediano"] >= 5e6]
    liquid_stocks = [r for r in liquid if r["tipo"] == "ação"]
    print(f"\nCorte da carteira (mediana ≥ R$ 5 mi/dia): {len(liquid)} papéis, "
          f"{len(liquid_stocks)} ações puras")
    print(f"Já analisados pelo nosso teste de replicação: {len(tested)} papéis à vista")
    print(f"Território inexplorado e líquido: {len(liquid) - len([r for r in liquid if r['symbol'] in tested])} papéis\n")

    top = sorted(rows, key=lambda r: -r["volume_mediano"])[:15]
    print("Top 15 por liquidez:")
    for r in top:
        flag = " ◀ já testado" if r["symbol"] in tested else ""
        print(f"  {r['symbol']:<8} {r['tipo']:<13} R$ {r['volume_mediano'] / 1e6:>8,.1f} mi/dia{flag}")

    out = ROOT / "web" / "universe_scan.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
