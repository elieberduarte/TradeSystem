"""A replicação definitiva: Donchian e squeeze nos 185 papéis líquidos.

O veredito da carteira repousava em 28 instrumentos (~14 apostas
independentes, p=0,032). O censo achou 185 líquidos (mediana ≥ R$ 5
mi/dia); só 19 tinham sido testados. Aqui a esteira roda no universo
inteiro, com a fricção por classe (ação, unit/ETF/FII, BDR) e a
mesma régua de sempre: replicação = fração de papéis com PnL > 0.

Pré-registro (escrito antes de rodar):
  Donchian só-compra replica em 65-75% (menos que os 84% dos 28 —
  o universo original tinha viés de qualidade); squeeze 65-80% com
  Calmar realista de 3-6. Small caps (terço inferior de liquidez)
  devem replicar PIOR que large (fricção relativa maior, mais ruído
  idiossincrático). Se Donchian < 55%, a hipótese de edge estrutural
  em ações perde força e a carteira fica só nos futuros.

Uso: python scripts/replication_liquid.py
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bot.backtest.engine import BacktestEngine
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.strategies.squeeze import SqueezeBreakoutStrategy
from src.bot.universe import EXPANDED

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 100_000.0
SLOTS = 10


def kind_of(symbol: str) -> str:
    if symbol.endswith("11"):
        return "unit/ETF/FII"
    if symbol[-2:] in ("32", "33", "34", "35", "39"):
        return "BDR"
    return "ação"


def friction(symbol: str) -> tuple[float, float]:
    """Slippage (R$/ação) + custo por unidade — à vista, por classe."""
    kind = kind_of(symbol)
    if kind == "BDR":
        return 0.02, 0.02          # spread mais largo
    return 0.01, 0.01


def risk() -> RiskManager:
    return RiskManager(RiskConfig(
        capital=CAPITAL, max_risk_per_trade_pct=1.0, max_daily_loss_pct=100.0,
        max_weekly_loss_pct=6.0, max_open_positions=1, mode="swing_trade",
        trading_start=time(0, 0), trading_end=time(23, 59),
        max_consecutive_losses=0, risk_slots=1, cash_slots=SLOTS, enforce_cash=True,
    ))


STRATEGIES = {
    "donchian": lambda: DonchianStrategy({"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}),
    "squeeze": SqueezeBreakoutStrategy,
}


def run_all(store: HistoryStore, symbols: list[str], volumes: dict[str, float]) -> pd.DataFrame:
    rows = []
    for name, factory in STRATEGIES.items():
        for symbol in symbols:
            candles = store.load(symbol, "1d")
            if candles is None or len(candles) < 700:
                continue
            slippage, cost = friction(symbol)
            engine = BacktestEngine(
                factory(), risk(), point_value=1.0, warmup=210,
                slippage_points=slippage, cost_per_contract=cost, unit_cost=None,
            )
            result = engine.run(symbol, candles)
            if len(result.trades) < 5:
                continue
            pnls = [t.pnl for t in result.trades]
            rows.append({
                "estrategia": name, "symbol": symbol, "tipo": kind_of(symbol),
                "volume_mi": volumes.get(symbol, 0.0) / 1e6,
                "trades": len(result.trades), "pnl": result.total_pnl,
                "acerto": np.mean([p > 0 for p in pnls]),
                "sharpe": float(np.mean(pnls) / np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0,
                "no_universo_28": symbol in EXPANDED,
            })
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, label: str) -> dict:
    if frame.empty:
        return {}
    positive = int((frame["pnl"] > 0).sum())
    return {
        "grupo": label, "papeis": len(frame), "positivos": positive,
        "replica": round(positive / len(frame), 3),
        "pnl_mediano": round(float(frame["pnl"].median()), 0),
        "sharpe_medio": round(float(frame["sharpe"].mean()), 3),
        "trades": int(frame["trades"].sum()),
    }


def main() -> None:
    store = HistoryStore(ROOT / "data")
    liquid = json.loads((ROOT / "data" / "liquid_universe.json").read_text(encoding="utf-8"))["symbols"]
    scan = json.loads((ROOT / "web" / "universe_scan.json").read_text(encoding="utf-8"))
    volumes = {r["symbol"]: r["volume_mediano"] for r in scan}

    frame = run_all(store, liquid, volumes)
    frame.to_parquet(ROOT / "data" / "replication_liquid.parquet")

    print(f"═══ Replicação no universo líquido · {frame['symbol'].nunique()} papéis "
          f"· critério: papéis com PnL > 0 ═══\n")
    report = []
    for name in STRATEGIES:
        part = frame[frame["estrategia"] == name]
        tercis = part["volume_mi"].quantile([1 / 3, 2 / 3]).to_list()
        groups = [
            ("TODOS", part),
            ("· já testados (28)", part[part["no_universo_28"]]),
            ("· INÉDITOS", part[~part["no_universo_28"]]),
            ("· ações", part[part["tipo"] == "ação"]),
            ("· units/ETFs/FIIs", part[part["tipo"] == "unit/ETF/FII"]),
            ("· BDRs", part[part["tipo"] == "BDR"]),
            ("· small (terço menos líquido)", part[part["volume_mi"] <= tercis[0]]),
            ("· mid", part[(part["volume_mi"] > tercis[0]) & (part["volume_mi"] <= tercis[1])]),
            ("· large (terço mais líquido)", part[part["volume_mi"] > tercis[1]]),
        ]
        print(f"── {name} ──")
        print(f"{'grupo':<32} {'papéis':>7} {'replica':>9} {'PnL med':>9} {'Sharpe':>7} {'trades':>7}")
        for label, sub in groups:
            s = summarize(sub, label)
            if not s:
                continue
            s["estrategia"] = name
            report.append(s)
            print(f"{label:<32} {s['papeis']:>7} {s['positivos']:>3}/{s['papeis']:<3} {s['replica']:>4.0%} "
                  f"{s['pnl_mediano']:>9,.0f} {s['sharpe_medio']:>7.3f} {s['trades']:>7,}")
        print()

    out = ROOT / "web" / "replication_liquid.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Régua: ≥70% replica · 50-70% inconclusivo · <50% pior que moeda")
    print(f"Exportado para {out}")


if __name__ == "__main__":
    main()
