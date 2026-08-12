"""Walk-forward das estratégias com custos realistas + export para o painel.

Uso:
    python scripts/run_backtest.py                 # WINQ26 5m (contrato atual)
    python scripts/run_backtest.py WIN$N 5m        # série contínua (anos)

Resultado honesto = janelas out-of-sample. Salva web/results.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import time

from src.bot.analysis.regime import Regime
from src.bot.backtest.walkforward import WalkForward
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.band_fade import BandFadeStrategy
from src.bot.strategies.ema_cross import EmaCrossStrategy
from src.bot.strategies.intraday_momentum import IntradayMomentumStrategy
from src.bot.strategies.opening_range import OpeningRangeStrategy
from src.bot.strategies.pfr import PfrStrategy
from src.bot.strategies.regime_filter import RegimeFilteredStrategy

ROOT = Path(__file__).resolve().parents[1]
WIN_POINT_VALUE = 0.20
CAPITAL = 20_000.0
SLIPPAGE_POINTS = 10.0   # 2 ticks do WIN por execução a mercado
COST_PER_CONTRACT = 1.0  # R$ por contrato, ida e volta (emolumentos B3)


def risk_factory() -> RiskManager:
    return RiskManager(
        RiskConfig(
            capital=CAPITAL,
            max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=3.0,
            max_open_positions=1,
            max_trades_per_day=5,
            max_consecutive_losses=3,
        )
    )


def risk_late() -> RiskManager:
    """Risco para setups que operam no fim do pregão (momentum de horário)."""
    return RiskManager(
        RiskConfig(
            capital=CAPITAL,
            max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=3.0,
            max_open_positions=1,
            trading_end=time(17, 30),
            flat_time=time(17, 55),
        )
    )


def export(report, symbol: str, timeframe: str, strategy: str) -> dict:
    trades = report.oos_trades
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
        "timeframe": timeframe,
        "oos_pnl": round(report.oos_pnl, 2),
        "trades": len(trades),
        "win_rate": round(wins / len(trades), 4) if trades else 0,
        "profit_factor": round(gains / losses, 2) if losses else None,
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
                "date": str(t.exit_time.date()),
                "side": t.side,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "reason": t.exit_reason,
                "pnl": round(t.pnl, 2),
            }
            for t in trades[-10:]
        ],
    }


def main() -> None:
    args = sys.argv[1:]
    symbol = args[0] if args else "WINQ26"
    timeframe = args[1] if len(args) > 1 else "5m"
    only = args[2] if len(args) > 2 else None

    store = HistoryStore()
    candles = store.load(symbol, timeframe)
    if candles is None:
        raise SystemExit(f"Sem acervo para {symbol} {timeframe}: rode scripts/download_history.py")
    print(f"{symbol} {timeframe}: {len(candles)} candles ({candles.index.min()} → {candles.index.max()})")
    print(f"Custos: slippage {SLIPPAGE_POINTS} pts/execução a mercado + R$ {COST_PER_CONTRACT}/contrato\n")

    # Janelas maiores para históricos longos
    long_history = len(candles) > 20_000
    train_bars, test_bars = (6000, 3000) if long_history else (2500, 1250)

    trend_only = {Regime.TREND_UP, Regime.TREND_DOWN}
    # (nome, fábrica, grade, risco, barras de holding máximo)
    runs = [
        ("opening_range", lambda p: OpeningRangeStrategy(p),
         {"range_bars": [3, 6], "rr": [1.5, 2.0]}, risk_factory, 0),
        ("ema_cross", lambda p: EmaCrossStrategy(p),
         {"fast": [9], "slow": [21], "trend": [0, 80], "rr": [1.5, 2.5]}, risk_factory, 0),
        (
            # Só opera rompimento quando o dia já mostra direção (ADX > 25)
            "opening_range_regime",
            lambda p: RegimeFilteredStrategy(OpeningRangeStrategy(p), allowed=trend_only),
            {"range_bars": [3, 6], "rr": [1.5, 2.0]}, risk_factory, 0,
        ),
        (
            # Reversão em regime lateral com holding longo — a arquitetura
            # que sobreviveu à validação rigorosa em futuros de índice
            "band_fade_regime",
            lambda p: RegimeFilteredStrategy(BandFadeStrategy(p), allowed={Regime.RANGE}),
            {"period": [20], "mult": [2.0, 2.5], "band": ["bollinger", "keltner"], "target": ["mid"]},
            risk_factory, 13,
        ),
        ("pfr", lambda p: PfrStrategy(p), {"rr": [1.0, 2.0]}, risk_factory, 13),
        (
            # Hipótese prioritária: gap + 1ª meia hora prevê a última meia
            # hora. 1 trade/dia. Testa também a variante contrária.
            "intraday_momentum",
            lambda p: IntradayMomentumStrategy(p),
            {"contrarian": [False, True], "min_move_mult": [0.0, 0.15]},
            risk_late, 0,
        ),
    ]
    if only:
        runs = [r for r in runs if r[0] == only]

    out = ROOT / "web" / "results.json"
    # Mescla com o que já existe: rodadas parciais não apagam as anteriores,
    # e o painel acumula estratégias para comparação.
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

    for name, factory, grid, risk, holding in runs:
        print(f"═══ {name} · {symbol} {timeframe} · capital R$ {CAPITAL:,.0f} ═══")
        wf = WalkForward(
            strategy_factory=factory,
            risk_factory=risk,
            point_value=WIN_POINT_VALUE,
            warmup=110,
            slippage_points=SLIPPAGE_POINTS,
            cost_per_contract=COST_PER_CONTRACT,
            max_holding_bars=holding,
        )
        report = wf.run("WIN", candles, grid, train_bars=train_bars, test_bars=test_bars)
        print(report.summary())
        # Chave inclui símbolo/timeframe: rodadas em bases diferentes coexistem
        payload["strategies"][f"{name} · {symbol} {timeframe}"] = export(
            report, symbol, timeframe, name
        )
        print()

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Resultados exportados para {out}")


if __name__ == "__main__":
    main()
