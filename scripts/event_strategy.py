"""A célula sobrevivente do estudo de eventos vira estratégia com custos.

Regra (pré-declarada no commit anterior):
  ENTRADA  fechamento de D+1 após comunicado de RESULTADO cuja reação
           D→D+1 foi >= +1 ATR (só o lado comprador — o negativo é moeda)
  SAÍDA    fechamento de D+15 (14 pregões) OU stop em 2×ATR abaixo
           da entrada, o que vier primeiro (stop checado no low)
  CUSTO    slippage R$ 0,01/ação na entrada e na saída + 0,01/ação
           de emolumentos — as mesmas frições da esteira de ações
  TAMANHO  1% do capital em risco (stop 2×ATR) — só para o PnL em R$
           ser comparável; a métrica que decide é a replicação e o
           Sharpe por trade.

Régua: replicação por papel (>= 60% dos papéis com PnL > 0),
comparação com placebo (mesma regra em dias fortes SEM evento) e
Sharpe por trade > 0 depois de custos.

Uso: python scripts/event_strategy.py
"""

import json
import sys
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.data.history import HistoryStore
from src.bot.strategies.swing_reversion import atr

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 100_000.0
RISK_PCT = 1.0
HOLD = 14
STOP_ATR = 2.0
SLIP = 0.01
FEE = 0.01
RNG = np.random.default_rng(7)


def simulate(daily: pd.DataFrame, entry_positions: list[int]) -> list[dict]:
    closes = daily["close"].to_numpy(float)
    lows = daily["low"].to_numpy(float)
    vol = atr(daily, 14).shift(1).to_numpy(float)
    trades = []
    for i in entry_positions:                       # i = índice de D+1 (barra da entrada)
        if i + HOLD >= len(closes) or np.isnan(vol[i]) or vol[i] <= 0:
            continue
        entry = closes[i] + SLIP
        stop = entry - STOP_ATR * vol[i]
        qty = int((CAPITAL * RISK_PCT / 100) / (entry - stop))
        if qty <= 0:
            continue
        exit_price, reason = None, "tempo"
        for j in range(i + 1, i + 1 + HOLD):
            if lows[j] <= stop:
                exit_price, reason = stop - SLIP, "stop"
                break
        if exit_price is None:
            exit_price = closes[i + HOLD] - SLIP
        pnl = (exit_price - entry) * qty - FEE * qty * 2
        trades.append({"pnl": pnl, "reason": reason, "ret_atr": (exit_price - entry) / vol[i]})
    return trades


def summarize(label: str, per_ticker: dict[str, list[dict]]) -> dict:
    all_trades = [t for ts in per_ticker.values() for t in ts]
    if not all_trades:
        return {}
    pnls = np.array([t["pnl"] for t in all_trades])
    by_ticker = {k: sum(t["pnl"] for t in v) for k, v in per_ticker.items() if len(v) >= 3}
    positive = sum(1 for v in by_ticker.values() if v > 0)
    sharpe = float(pnls.mean() / pnls.std(ddof=1)) if len(pnls) > 1 else 0.0
    row = {
        "label": label, "trades": len(pnls),
        "acerto": round(float((pnls > 0).mean()), 3),
        "pnl_total": round(float(pnls.sum()), 0),
        "pnl_medio": round(float(pnls.mean()), 1),
        "sharpe_trade": round(sharpe, 3),
        "t": round(sharpe * sqrt(len(pnls)), 2),
        "stops": round(float(np.mean([t["reason"] == "stop" for t in all_trades])), 3),
        "papeis": len(by_ticker), "positivos": positive,
        "replica": round(positive / len(by_ticker), 3) if by_ticker else None,
    }
    print(f"{label:<34} {row['trades']:>6} {row['acerto']:>7.1%} {row['pnl_medio']:>+9.1f} "
          f"{row['sharpe_trade']:>+7.3f} {row['t']:>6} {row['stops']:>6.1%} "
          f"{positive:>4}/{len(by_ticker):<4} {row['replica'] if row['replica'] is not None else 0:>5.0%}")
    return row


def main() -> None:
    store = HistoryStore(ROOT / "data")
    events = pd.read_parquet(ROOT / "data" / "event_study.parquet")
    signals = events[(events["tipo"] == "resultado") & (events["reacao"] >= 1.0)]

    real, placebo = {}, {}
    for ticker, group in signals.groupby("ticker"):
        daily = store.load(ticker, "1d")
        if daily is None:
            continue
        idx = daily.index
        # 'data' no estudo é D (alinhado ao calendário); a entrada é D+1
        entry_pos = [p + 1 for p in idx.get_indexer(pd.DatetimeIndex(group["data"])) if p >= 0]
        real[ticker] = simulate(daily, entry_pos)

        # placebo: dias fortes (ret >= 1 ATR) sem evento, mesma quantidade
        closes = daily["close"]
        vol = atr(daily, 14).shift(1)
        strong = ((closes - closes.shift(1)) / vol >= 1.0)
        event_days = set(idx.get_indexer(pd.DatetimeIndex(events[events["ticker"] == ticker]["data"])))
        candidates = [i for i in np.where(strong.to_numpy())[0]
                      if 20 <= i < len(idx) - HOLD - 2 and (i - 1) not in event_days]
        k = min(len(entry_pos), len(candidates))
        if k:
            placebo[ticker] = simulate(daily, list(RNG.choice(candidates, size=k, replace=False)))

    print("═══ Estratégia PEAD comprador · resultado + reação ≥ +1 ATR · entra D+1, sai D+15 ou stop 2×ATR ═══")
    print(f"{'variante':<34} {'trades':>6} {'acerto':>7} {'R$/trade':>9} {'Sharpe':>7} {'t':>6} {'stops':>6} {'papéis+':>9} {'repl':>5}")
    print("-" * 100)
    report = [summarize("PEAD (evento real)", real), summarize("placebo (dia forte sem evento)", placebo)]

    out = ROOT / "web" / "event_strategy.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nCustos: slippage R$ 0,01/ação (2 pontas) + emolumentos 0,01/ação · risco 1% de R$ 100k")
    print(f"Exportado para {out}")


if __name__ == "__main__":
    main()
