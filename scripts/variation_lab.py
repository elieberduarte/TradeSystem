"""Laboratório de variações sobre uma estratégia aprovada.

Pega a base que sobreviveu à replicação e testa sistematicamente as
variações de gestão — relação alvo/risco, largura de stop, sem alvo,
breakeven, stop móvel e saída parcial.

O critério NÃO é o lucro no melhor ativo: é em quantos dos 13
instrumentos a variação fica positiva. Foi assim que descobrimos que
o band_fade era sorte e o donchian era edge.

Uso: python scripts/variation_lab.py
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.backtest.engine import BacktestEngine
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.strategies.overlays import (
    NoTargetOverlay,
    RewardRatioOverlay,
    StopWidthOverlay,
)

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 150_000.0
POINT_VALUE = {"WIN$N": 0.20, "WDO$N": 10.00}
UNIVERSE = [
    "WIN$N", "WDO$N", "BOVA11", "SMAL11", "IVVB11",
    "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "B3SA3", "WEGE3", "PRIO3",
]
BASE = {"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}


def risk() -> RiskManager:
    return RiskManager(
        RiskConfig(
            capital=CAPITAL, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=100.0, max_weekly_loss_pct=6.0,
            max_open_positions=1, mode="swing_trade",
            trading_start=time(0, 0), trading_end=time(23, 59),
            max_consecutive_losses=0, risk_slots=len(UNIVERSE),
        )
    )


def friction(symbol: str) -> tuple[float, float]:
    if symbol in POINT_VALUE:
        return (10.0, 1.0) if symbol == "WIN$N" else (0.5, 2.0)
    return (0.01, 0.01)


def evaluate(label: str, factory, engine_kwargs: dict, store: HistoryStore) -> dict:
    """Roda a variação em todo o universo e devolve o placar."""
    pnls, trades, wins_total, total = [], 0, 0, 0
    for symbol in UNIVERSE:
        candles = store.load(symbol, "1d")
        if candles is None:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            factory(), risk(), point_value=POINT_VALUE.get(symbol, 1.0),
            warmup=210, slippage_points=slippage, cost_per_contract=cost,
            **engine_kwargs,
        )
        result = engine.run(symbol, candles)
        if not result.trades:
            continue
        pnls.append(result.total_pnl)
        trades += len(result.trades)
        wins_total += sum(1 for t in result.trades if t.pnl > 0)
        total += len(result.trades)

    if not pnls:
        return {"label": label, "tested": 0}
    positive = sum(1 for p in pnls if p > 0)
    return {
        "label": label,
        "tested": len(pnls),
        "positive": positive,
        "rate": round(positive / len(pnls), 3),
        "median_pnl": round(sorted(pnls)[len(pnls) // 2], 2),
        "total_pnl": round(sum(pnls), 2),
        "trades": trades,
        "win_rate": round(wins_total / total, 3) if total else 0,
    }


def main() -> None:
    store = HistoryStore()

    base = lambda: DonchianStrategy(BASE)  # noqa: E731
    variations = [
        ("BASE  alvo 3x risco", base, {}),
        # ── Relação alvo/risco ──
        ("R:R  1:1", lambda: RewardRatioOverlay(base(), 1.0), {}),
        ("R:R  2:1", lambda: RewardRatioOverlay(base(), 2.0), {}),
        ("R:R  5:1", lambda: RewardRatioOverlay(base(), 5.0), {}),
        ("R:R  0,5:1 (invertido)", lambda: RewardRatioOverlay(base(), 0.5), {}),
        ("R:R  0,33:1 (invertido)", lambda: RewardRatioOverlay(base(), 0.33), {}),
        ("sem alvo (deixa correr)", lambda: NoTargetOverlay(base()), {}),
        # ── Largura do stop ──
        ("stop 0,5x (apertado)", lambda: StopWidthOverlay(base(), 0.5), {}),
        ("stop 1,5x (largo)", lambda: StopWidthOverlay(base(), 1.5), {}),
        # ── Gestão da posição ──
        ("breakeven a 1x", base, {"breakeven_at": 1.0}),
        ("breakeven a 2x", base, {"breakeven_at": 2.0}),
        ("stop móvel a 1x", base, {"trailing_atr": 1.0}),
        ("stop móvel a 2x", base, {"trailing_atr": 2.0}),
        ("parcial a 1x", base, {"partial_at": 1.0}),
        ("parcial a 2x", base, {"partial_at": 2.0}),
        # ── Combinações ──
        ("sem alvo + móvel 2x", lambda: NoTargetOverlay(base()), {"trailing_atr": 2.0}),
        ("parcial 1x + breakeven", base, {"partial_at": 1.0, "breakeven_at": 1.0}),
    ]

    print("═══ Laboratório de variações · donchian somente-compra ═══")
    print(f"Base: canal {BASE['channel']}, stop {BASE['stop_atr']}xATR, alvo {BASE['rr']}x")
    print(f"{len(UNIVERSE)} instrumentos · 5 anos diários · risco dividido por instrumento\n")
    print(f"{'variação':<26} {'positivos':>10} {'taxa':>6} {'mediana':>10} {'total':>11} {'acerto':>7}")
    print("-" * 76)

    report = []
    for label, factory, kwargs in variations:
        row = evaluate(label, factory, kwargs, store)
        report.append(row)
        if not row.get("tested"):
            print(f"{label:<26} {'sem trades':>10}")
            continue
        print(
            f"{label:<26} {row['positive']:>4}/{row['tested']:<5} {row['rate']:>5.0%} "
            f"{row['median_pnl']:>10,.0f} {row['total_pnl']:>11,.0f} {row['win_rate']:>6.1%}"
        )

    print("-" * 76)
    ranked = sorted(
        (r for r in report if r.get("tested")),
        key=lambda r: (r["rate"], r["median_pnl"]),
        reverse=True,
    )
    print("\nMelhores por REPLICAÇÃO (não por lucro):")
    for row in ranked[:5]:
        print(f"  {row['rate']:>4.0%}  {row['label']:<26} mediana {row['median_pnl']:>9,.0f}")

    out = ROOT / "web" / "variations.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
