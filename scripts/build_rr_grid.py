"""Pré-calcula a grade de alvo/stop para o explorador interativo.

O backtest é pesado demais para rodar no navegador a cada ajuste. A
solução é inverter: calcular todas as combinações aqui, uma vez, e o
painel apenas alterna entre resultados prontos — resposta instantânea,
sem servidor de cálculo.

Uso: python scripts/build_rr_grid.py
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.bot.backtest.engine import BacktestEngine
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.strategies.overlays import RewardRatioOverlay, StopWidthOverlay

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 150_000.0
POINT_VALUE = {"WIN$N": 0.20, "WDO$N": 10.00}
UNIVERSE = [
    "WIN$N", "WDO$N", "BOVA11", "SMAL11", "IVVB11",
    "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "B3SA3", "WEGE3", "PRIO3",
]
BASE = {"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}

# Eixos da grade: relação alvo/risco e largura do stop
RATIOS = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]
STOP_FACTORS = [0.5, 0.75, 1.0, 1.5, 2.0]


def risk() -> RiskManager:
    # risk_slots=1: cada instrumento avaliado isoladamente, para o
    # explorador mostrar o efeito do R:R e não o do rateio de capital
    return RiskManager(
        RiskConfig(
            capital=CAPITAL, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=100.0, max_weekly_loss_pct=6.0,
            max_open_positions=1, mode="swing_trade",
            trading_start=time(0, 0), trading_end=time(23, 59),
            max_consecutive_losses=0,
        )
    )


def friction(symbol: str) -> tuple[float, float]:
    if symbol in POINT_VALUE:
        return (10.0, 1.0) if symbol == "WIN$N" else (0.5, 2.0)
    return (0.01, 0.01)


def evaluate(store: HistoryStore, ratio: float, stop_factor: float) -> dict:
    per_symbol, pnls = {}, []
    trades = wins = 0
    for symbol in UNIVERSE:
        candles = store.load(symbol, "1d")
        if candles is None:
            continue
        slippage, cost = friction(symbol)

        def factory():
            base = DonchianStrategy(BASE)
            widened = StopWidthOverlay(base, stop_factor) if stop_factor != 1.0 else base
            return RewardRatioOverlay(widened, ratio)

        engine = BacktestEngine(
            factory(), risk(), point_value=POINT_VALUE.get(symbol, 1.0),
            warmup=210, slippage_points=slippage, cost_per_contract=cost,
        )
        result = engine.run(symbol, candles)
        if not result.trades:
            continue
        pnls.append(result.total_pnl)
        trades += len(result.trades)
        wins += sum(1 for t in result.trades if t.pnl > 0)
        per_symbol[symbol] = {
            "pnl": round(result.total_pnl, 0),
            "trades": len(result.trades),
            "win_rate": round(sum(1 for t in result.trades if t.pnl > 0) / len(result.trades), 3),
            "max_drawdown": round(result.max_drawdown, 0),
        }

    if not pnls:
        return {}
    positive = sum(1 for p in pnls if p > 0)
    return {
        "ratio": ratio,
        "stop_factor": stop_factor,
        "tested": len(pnls),
        "positive": positive,
        "rate": round(positive / len(pnls), 3),
        "median_pnl": round(float(np.median(pnls)), 0),
        "total_pnl": round(float(sum(pnls)), 0),
        "trades": trades,
        "win_rate": round(wins / trades, 3) if trades else 0,
        "per_symbol": per_symbol,
    }


def main() -> None:
    store = HistoryStore()
    total = len(RATIOS) * len(STOP_FACTORS)
    print(f"═══ Grade alvo/stop · {total} combinações × {len(UNIVERSE)} instrumentos ═══\n")

    cells = []
    for i, stop_factor in enumerate(STOP_FACTORS):
        for j, ratio in enumerate(RATIOS):
            cell = evaluate(store, ratio, stop_factor)
            if not cell:
                continue
            cells.append(cell)
            done = i * len(RATIOS) + j + 1
            print(
                f"[{done:>2}/{total}] stop {stop_factor:.2f}x · alvo {ratio:.2f}x → "
                f"{cell['positive']}/{cell['tested']} positivos · "
                f"mediana {cell['median_pnl']:>8,.0f} · acerto {cell['win_rate']:.1%}"
            )

    payload = {
        "base": BASE,
        "capital": CAPITAL,
        "universe": UNIVERSE,
        "ratios": RATIOS,
        "stop_factors": STOP_FACTORS,
        "cells": cells,
    }
    out = ROOT / "web" / "rr_grid.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nExportado para {out} ({len(cells)} combinações)")


if __name__ == "__main__":
    main()
