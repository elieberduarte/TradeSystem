"""Exporta a anatomia das operações para visualização.

Para cada trade real do backtest: os candles ao redor, o canal de
Donchian que gerou o gatilho, o preço de entrada, o stop, o alvo e o
ponto de saída. É o que permite ver a mecânica em vez de só o
resultado agregado.

Uso: python scripts/export_trade_anatomy.py [SIMBOLO] [quantidade]
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.bot.backtest.engine import BacktestEngine
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.strategies.swing_reversion import atr

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 150_000.0
POINT_VALUE = {"WIN$N": 0.20, "IND$N": 1.00, "WDO$N": 10.00, "DOL$N": 50.00}
CHANNEL = 20
PADDING = 25  # candles mostrados antes da entrada e depois da saída


def risk_manager() -> RiskManager:
    return RiskManager(
        RiskConfig(
            capital=CAPITAL, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=100.0, max_weekly_loss_pct=6.0,
            max_open_positions=1, mode="swing_trade",
            trading_start=time(0, 0), trading_end=time(23, 59),
            max_consecutive_losses=0,
        )
    )


def channel_series(candles: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    """Canal formado pelos candles anteriores ao atual (como a estratégia vê)."""
    upper = candles["high"].rolling(window).max().shift(1)
    lower = candles["low"].rolling(window).min().shift(1)
    return upper, lower


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ABEV3"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    candles = HistoryStore().load(symbol, "1d")
    if candles is None:
        raise SystemExit(f"Sem acervo diário para {symbol}")

    point_value = POINT_VALUE.get(symbol, 1.0)
    slippage = 10.0 if symbol in ("WIN$N", "IND$N") else (0.5 if symbol in POINT_VALUE else 0.01)
    cost = 1.0 if symbol in POINT_VALUE else 0.01

    engine = BacktestEngine(
        DonchianStrategy({"channel": CHANNEL, "stop_atr": 2.0, "rr": 3.0}),
        risk_manager(), point_value=point_value, warmup=210,
        slippage_points=slippage, cost_per_contract=cost,
    )
    result = engine.run(symbol, candles)
    if not result.trades:
        raise SystemExit(f"{symbol}: nenhum trade gerado")

    upper, lower = channel_series(candles, CHANNEL)
    volatility = atr(candles, 14)
    positions = {ts: i for i, ts in enumerate(candles.index)}

    # Amostra representativa: alterna ganhos e perdas, do mais recente
    winners = [t for t in result.trades if t.pnl > 0][::-1]
    losers = [t for t in result.trades if t.pnl <= 0][::-1]
    sample, i = [], 0
    while len(sample) < limit and (i < len(winners) or i < len(losers)):
        if i < len(winners):
            sample.append(winners[i])
        if len(sample) < limit and i < len(losers):
            sample.append(losers[i])
        i += 1
    sample.sort(key=lambda t: t.entry_time)

    trades = []
    for trade in sample:
        start = max(0, positions[trade.entry_time] - PADDING)
        end = min(len(candles) - 1, positions[trade.exit_time] + PADDING)
        window = candles.iloc[start : end + 1]
        trades.append(
            {
                "symbol": symbol,
                "side": trade.side,
                "entry_time": str(trade.entry_time.date()),
                "entry_price": round(trade.entry_price, 2),
                "exit_time": str(trade.exit_time.date()),
                "exit_price": round(trade.exit_price, 2),
                "exit_reason": trade.exit_reason,
                "stop_loss": round(trade.stop_loss, 2),
                "take_profit": round(trade.take_profit, 2),
                "quantity": trade.quantity,
                "pnl": round(trade.pnl, 2),
                "bars_held": (trade.exit_time - trade.entry_time).days,
                "atr_at_entry": round(float(volatility.loc[trade.entry_time]), 2),
                "candles": [
                    {
                        "date": str(ts.date()),
                        "o": round(float(row["open"]), 2),
                        "h": round(float(row["high"]), 2),
                        "l": round(float(row["low"]), 2),
                        "c": round(float(row["close"]), 2),
                        "up": None if pd.isna(upper.loc[ts]) else round(float(upper.loc[ts]), 2),
                        "lo": None if pd.isna(lower.loc[ts]) else round(float(lower.loc[ts]), 2),
                    }
                    for ts, row in window.iterrows()
                ],
            }
        )

    wins = sum(1 for t in result.trades if t.pnl > 0)
    payload = {
        "symbol": symbol,
        "strategy": "donchian",
        "params": {"channel": CHANNEL, "stop_atr": 2.0, "rr": 3.0},
        "point_value": point_value,
        "capital": CAPITAL,
        "summary": {
            "trades": len(result.trades),
            "win_rate": round(wins / len(result.trades), 3),
            "total_pnl": round(result.total_pnl, 2),
            "expectancy": round(result.expectancy, 2),
            "max_drawdown": round(result.max_drawdown, 2),
            "longest_losing_streak": result.longest_losing_streak,
        },
        "trades_sample": trades,
    }
    out = ROOT / "web" / "anatomy.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{symbol}: {len(result.trades)} trades, {wins / len(result.trades):.1%} de acerto")
    print(f"Amostra de {len(trades)} operações exportada para {out}")
    for t in trades:
        print(
            f"  {t['entry_time']} {t['side']:<4} entrada {t['entry_price']:>9,.2f} "
            f"stop {t['stop_loss']:>9,.2f} alvo {t['take_profit']:>9,.2f} → "
            f"{t['exit_reason']:<16} {t['pnl']:>10,.2f}"
        )


if __name__ == "__main__":
    main()
