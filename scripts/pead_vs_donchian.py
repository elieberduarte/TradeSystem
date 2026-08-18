"""PEAD × Donchian no à vista: fontes independentes ou a mesma aposta?

O PEAD (gatilho de INFORMAÇÃO: resultado + reação) e o Donchian
(gatilho de TENDÊNCIA: rompimento de 20 dias) podem estar comprando
os mesmos papéis nos mesmos dias — um resultado forte costuma romper
o canal. Se a correlação dos PnLs diários for baixa, a soma agrega
Calmar (a lição da carteira de 3 estratégias, 3,37 → 4,42); se for
alta, o PEAD é o Donchian com outro nome.

Universo: os papéis líquidos onde AMBAS geram trades. Métrica:
correlação dos PnLs diários agregados + Calmar isolado e combinado
50/50 (mesmo capital, metade do risco cada — a comparação honesta).

Uso: python scripts/pead_vs_donchian.py
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.backtest.engine import BacktestEngine, Trade
from src.bot.backtest.portfolio import combine, correlation_matrix
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.strategies.swing_reversion import atr

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 100_000.0
HOLD, STOP_ATR, SLIP, FEE = 14, 2.0, 0.01, 0.01


def risk() -> RiskManager:
    return RiskManager(RiskConfig(
        capital=CAPITAL, max_risk_per_trade_pct=1.0, max_daily_loss_pct=100.0,
        max_weekly_loss_pct=6.0, max_open_positions=1, mode="swing_trade",
        trading_start=time(0, 0), trading_end=time(23, 59),
        max_consecutive_losses=0, risk_slots=1, cash_slots=10, enforce_cash=True,
    ))


def pead_trades(daily: pd.DataFrame, entry_positions: list[int], symbol: str) -> list[Trade]:
    """Reproduz a regra do event_strategy como objetos Trade (para o portfolio)."""
    closes = daily["close"].to_numpy(float)
    lows = daily["low"].to_numpy(float)
    vol = atr(daily, 14).shift(1).to_numpy(float)
    idx = daily.index
    trades = []
    for i in entry_positions:
        if i + HOLD >= len(closes) or np.isnan(vol[i]) or vol[i] <= 0:
            continue
        entry = closes[i] + SLIP
        stop = entry - STOP_ATR * vol[i]
        qty = int((CAPITAL * 0.01) / (entry - stop))
        if qty <= 0:
            continue
        exit_i, exit_price, reason = i + HOLD, closes[i + HOLD] - SLIP, "tempo"
        for j in range(i + 1, i + 1 + HOLD):
            if lows[j] <= stop:
                exit_i, exit_price, reason = j, stop - SLIP, "stop"
                break
        trade = Trade(symbol=symbol, side="buy", entry_time=idx[i], entry_price=entry,
                      quantity=qty, stop_loss=stop, take_profit=entry + 100 * vol[i])
        trade.exit_time, trade.exit_price, trade.exit_reason = idx[exit_i], exit_price, reason
        trade.pnl = (exit_price - entry) * qty - FEE * qty * 2
        trades.append(trade)
    return trades


def main() -> None:
    store = HistoryStore(ROOT / "data")
    events = pd.read_parquet(ROOT / "data" / "event_study.parquet")
    signals = events[(events["tipo"] == "resultado") & (events["reacao"] >= 1.0)]

    pead_all, donch_all = [], []
    both = 0
    for ticker, group in signals.groupby("ticker"):
        daily = store.load(ticker, "1d")
        if daily is None or len(daily) < 700:
            continue
        pos = [p + 1 for p in daily.index.get_indexer(pd.DatetimeIndex(group["data"])) if p >= 0]
        p_trades = pead_trades(daily, pos, ticker)
        engine = BacktestEngine(
            DonchianStrategy({"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}),
            risk(), point_value=1.0, warmup=210, slippage_points=SLIP, cost_per_contract=FEE,
        )
        d_trades = engine.run(ticker, daily).trades
        if len(p_trades) >= 3 and len(d_trades) >= 3:
            both += 1
            pead_all.extend(p_trades)
            donch_all.extend(d_trades)

    results = {"pead": pead_all, "donchian": donch_all}
    corr = correlation_matrix(results)
    rho = float(corr.loc["pead", "donchian"])

    print(f"═══ PEAD × Donchian · {both} papéis com as duas · à vista, mesmos custos ═══")
    print(f"PEAD: {len(pead_all)} trades · Donchian: {len(donch_all)} trades")
    print(f"Correlação dos PnLs diários: {rho:+.3f}\n")

    # Sobreposição real: quantos trades PEAD entram em dia em que o Donchian também comprou o mesmo papel
    d_keys = {(t.symbol, pd.Timestamp(t.entry_time).normalize()) for t in donch_all}
    overlap = sum(1 for t in pead_all if (t.symbol, pd.Timestamp(t.entry_time).normalize()) in d_keys)
    print(f"Entradas do PEAD no MESMO dia e papel que o Donchian: {overlap}/{len(pead_all)} "
          f"({overlap / len(pead_all):.1%})\n")

    print(f"{'carteira':<24} {'PnL':>10} {'retorno':>8} {'DD':>8} {'Calmar':>7} {'±dia':>8}")
    print("-" * 70)
    report = {"correlacao": round(rho, 3), "sobreposicao": round(overlap / len(pead_all), 3), "carteiras": []}
    for label, subset, weights in [
        ("PEAD isolado", {"pead": pead_all}, None),
        ("Donchian isolado", {"donchian": donch_all}, None),
        ("50/50", results, {"pead": 0.5, "donchian": 0.5}),
    ]:
        r = combine(subset, weights)
        print(f"{label:<24} {r.total_pnl:>10,.0f} {r.total_pnl / CAPITAL:>7.1%} "
              f"{r.max_drawdown / CAPITAL:>7.1%} {r.calmar:>7.2f} {r.daily_std:>8,.0f}")
        report["carteiras"].append({"label": label, "pnl": round(r.total_pnl, 0),
                                    "calmar": round(r.calmar, 2),
                                    "dd_pct": round(r.max_drawdown / CAPITAL * 100, 1)})

    out = ROOT / "web" / "pead_vs_donchian.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
