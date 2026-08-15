"""Walk-forward das estratégias de swing em candles diários.

Uso: python scripts/run_swing_backtest.py [SIMBOLO]

Diferenças para o intradiário: sem zeragem de fim de pregão, janela de
horário aberta (candles diários carimbam 00:00) e custo proporcional
muito menor, porque a operação dura dias em vez de minutos.
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.analysis.regime import Regime
from src.bot.backtest.walkforward import WalkForward
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.band_fade import BandFadeStrategy
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.strategies.ema_cross import EmaCrossStrategy
from src.bot.strategies.ibs import IbsStrategy
from src.bot.strategies.regime_filter import RegimeFilteredStrategy
from src.bot.strategies.swing_reversion import SwingReversionStrategy

ROOT = Path(__file__).resolve().parents[1]
POINT_VALUE = {"WIN$N": 0.20, "WDO$N": 10.00}
# O ATR diário do WIN é ~3.200 pts: um stop de 2xATR arrisca R$ 1.274 por
# contrato. Com 1% de risco, é preciso ~R$ 130 mil para caber 1 contrato.
# Este capital existe para responder "o edge existe?", separado de
# "quanto custa operá-lo" — com R$ 20 mil o sizing trava em zero contrato.
CAPITAL = 150_000.0
SLIPPAGE_POINTS = 10.0
COST_PER_CONTRACT = 1.0


def swing_risk() -> RiskManager:
    return RiskManager(
        RiskConfig(
            capital=CAPITAL,
            max_risk_per_trade_pct=1.0,
            # Swing acumula perda em dias: o limite útil é o semanal
            max_daily_loss_pct=100.0,
            max_weekly_loss_pct=6.0,
            max_open_positions=1,
            mode="swing_trade",
            # Candles diários carimbam 00:00 — janela precisa ser aberta
            trading_start=time(0, 0),
            trading_end=time(23, 59),
            max_consecutive_losses=0,
        )
    )


def trade_window(trade, candles, positions, before: int = 12, after: int = 8):
    """Candles ao redor do trade, para o gráfico clicável do painel."""
    entry_idx = positions.get(trade.entry_time)
    exit_idx = positions.get(trade.exit_time)
    if entry_idx is None or exit_idx is None:
        return None
    start = max(0, entry_idx - before)
    end = min(len(candles) - 1, exit_idx + after)
    return [
        {
            "date": str(ts.date()),
            "o": round(float(row["open"]), 2), "h": round(float(row["high"]), 2),
            "l": round(float(row["low"]), 2), "c": round(float(row["close"]), 2),
        }
        for ts, row in candles.iloc[start : end + 1].iterrows()
    ]


def export(report, symbol: str, strategy: str, candles) -> dict:
    trades = report.oos_trades
    positions = {ts: i for i, ts in enumerate(candles.index)}
    daily: dict[str, float] = {}
    for t in trades:
        key = str(t.exit_time.date())
        daily[key] = daily.get(key, 0.0) + t.pnl
    days = sorted(daily)
    cum, equity = 0.0, []
    for d in days:
        cum += daily[d]
        equity.append({"date": d, "value": round(cum, 2)})
    wins = sum(1 for t in trades if t.pnl > 0)
    gains = sum(t.pnl for t in trades if t.pnl > 0)
    losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
    return {
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": "1d",
        "oos_pnl": round(report.oos_pnl, 2),
        "trades": len(trades),
        "win_rate": round(wins / len(trades), 4) if trades else 0,
        "profit_factor": round(gains / losses, 2) if losses else None,
        "risk": report.risk_metrics(),
        "equity": equity,
        "daily_pnl": [{"date": d, "value": round(daily[d], 2)} for d in days],
        "windows": [
            {
                "test_start": str(w.test_start.date()),
                "test_end": str(w.test_end.date()),
                "params": w.best_params,
                "train_pnl": round(w.train_result.total_pnl, 2),
                "test_pnl": round(w.test_result.total_pnl, 2),
                "test_trades": len(w.test_result.trades),
            }
            for w in report.windows
        ],
        "last_trades": [
            {
                "date": str(t.exit_time.date()), "side": t.side,
                "entry": t.entry_price, "exit": t.exit_price,
                "reason": t.exit_reason, "pnl": round(t.pnl, 2),
                "entry_date": str(t.entry_time.date()),
                "exit_date": str(t.exit_time.date()),
                "trigger": t.entry_reason,
                "stop": round(t.stop_loss, 2),
                "target": round(t.take_profit, 2),
                "qty": t.quantity,
                "candles": trade_window(t, candles, positions),
            }
            for t in trades[-10:]
        ],
    }


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "WIN$N"
    candles = HistoryStore().load(symbol, "1d")
    if candles is None:
        raise SystemExit(f"Sem acervo diário para {symbol}")

    point_value = POINT_VALUE.get(symbol, 1.0)
    print(f"═══ SWING · {symbol} diário · capital R$ {CAPITAL:,.0f} ═══")
    print(f"{len(candles)} pregões ({candles.index.min().date()} → {candles.index.max().date()})")
    print(f"Custos: {SLIPPAGE_POINTS} pts/execução + R$ {COST_PER_CONTRACT}/contrato\n")

    trend_only = {Regime.TREND_UP, Regime.TREND_DOWN}
    runs = [
        ("donchian", lambda p: DonchianStrategy(p),
         {"channel": [20, 40], "stop_atr": [2.0], "rr": [2.0, 3.0]}),
        # Candidatas a curva mais suave: reversão com alvo curto acerta
        # mais vezes e sofre mergulhos menores que seguir tendência
        ("ibs", lambda p: IbsStrategy(p),
         {"entry_low": [0.1, 0.2], "target_atr": [0.5, 1.0], "stop_atr": [2.0]}),
        ("ibs_trend", lambda p: IbsStrategy(p),
         {"entry_low": [0.1, 0.2], "target_atr": [0.5, 1.0], "trend_filter": [200]}),
        ("band_fade_daily", lambda p: BandFadeStrategy(p),
         {"period": [20], "mult": [2.0, 2.5], "band": ["keltner"], "target": ["mid", 1.5]}),
        ("swing_reversion", lambda p: SwingReversionStrategy(p),
         {"lookback": [40, 60], "threshold_std": [1.0, 1.5], "rr": [1.5, 2.0]}),
        ("donchian_regime", lambda p: RegimeFilteredStrategy(DonchianStrategy(p), allowed=trend_only),
         {"channel": [20, 40], "stop_atr": [2.0], "rr": [2.0, 3.0]}),
        ("ema_cross_swing", lambda p: EmaCrossStrategy(p),
         {"fast": [9, 20], "slow": [21, 50], "trend": [0, 200], "rr": [2.0]}),
    ]

    out = ROOT / "web" / "results.json"
    payload = {
        "updated": str(candles.index.max()),
        "capital": CAPITAL,
        "costs": {"slippage_points": SLIPPAGE_POINTS, "cost_per_contract": COST_PER_CONTRACT},
        "strategies": {},
    }
    if out.exists():
        try:
            payload["strategies"] = json.loads(out.read_text(encoding="utf-8")).get("strategies", {})
        except (json.JSONDecodeError, OSError):
            pass

    for name, factory, grid in runs:
        print(f"── {name} ──")
        wf = WalkForward(
            strategy_factory=factory,
            risk_factory=swing_risk,
            point_value=point_value,
            warmup=210,
            slippage_points=SLIPPAGE_POINTS,
            cost_per_contract=COST_PER_CONTRACT,
        )
        # ~2 anos de treino, ~6 meses de teste
        report = wf.run(symbol, candles, grid, train_bars=500, test_bars=125)
        print(report.summary())
        trades = report.oos_trades
        if trades:
            wins = sum(1 for t in trades if t.pnl > 0)
            print(f"  Win rate OOS: {wins / len(trades):.1%}")
        payload["strategies"][f"{name} · {symbol} 1d"] = export(report, symbol, name, candles)
        print()

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Exportado para {out}")


if __name__ == "__main__":
    main()
