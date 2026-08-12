"""Auditoria com o universo ampliado.

A pergunta central: ao incluir mercados dirigidos por forças
diferentes (juros, safra, tecnologia americana, China, ouro, cripto),
o número de apostas efetivamente independentes sobe o suficiente para
o resultado do Donchian passar a ser significativo?

Compara diretamente o universo original com o ampliado, e quebra o
resultado por bloco econômico.

Uso: python scripts/expanded_audit.py
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
from src.bot.universe import EXPANDED, ORIGINAL, POINT_VALUE, block_of

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 150_000.0
BASE = {"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}


def risk() -> RiskManager:
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
    if symbol == "WIN$N":
        return 10.0, 1.0
    if symbol == "WDO$N":
        return 0.5, 2.0
    if symbol.startswith("DI1"):
        return 0.005, 0.01     # DI é cotado em taxa: fricção em pontos-base
    if symbol.endswith("$N"):
        return 0.5, 1.0        # commodities cotadas em R$/unidade
    return 0.01, 0.01          # ações e ETFs


def run_universe(store: HistoryStore, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, daily = [], {}
    for symbol in symbols:
        candles = store.load(symbol, "1d")
        if candles is None or len(candles) < 700:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            DonchianStrategy(BASE), risk(), point_value=POINT_VALUE.get(symbol, 1.0),
            warmup=210, slippage_points=slippage, cost_per_contract=cost,
        )
        result = engine.run(symbol, candles)
        if len(result.trades) < 5:
            continue
        pnls = [t.pnl for t in result.trades]
        sharpe = float(np.mean(pnls) / np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
        rows.append(
            {
                "symbol": symbol,
                "bloco": block_of(symbol),
                "trades": len(result.trades),
                "sharpe_trade": round(sharpe, 3),
                "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 3),
                "positivo": result.total_pnl > 0,
            }
        )
        daily[symbol] = candles["close"].pct_change().dropna()
    return pd.DataFrame(rows), pd.DataFrame(daily)


def effective_bets(returns: pd.DataFrame) -> float:
    matrix = returns.dropna().corr().to_numpy()
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    weights = eigenvalues / eigenvalues.sum()
    return float(np.exp(-(weights * np.log(weights)).sum()))


def null_p_value(returns: pd.DataFrame, observed: int, rounds: int = 4000, seed: int = 11) -> tuple[float, np.ndarray]:
    clean = returns.dropna()
    corr = clean.corr().to_numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    transform = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 1e-10, None)))
    rng = np.random.default_rng(seed)
    counts = np.empty(rounds, dtype=int)
    for i in range(rounds):
        noise = rng.standard_normal((len(clean), corr.shape[0]))
        counts[i] = int(((noise @ transform.T).sum(axis=0) > 0).sum())
    return float((counts >= observed).mean()), counts


def summarize(label: str, results: pd.DataFrame, returns: pd.DataFrame) -> dict:
    positives = int(results["positivo"].sum())
    total = len(results)
    effective = effective_bets(returns)
    p_value, counts = null_p_value(returns, positives)
    upper = returns.corr().values[np.triu_indices(len(returns.columns), 1)]

    print(f"\n═══ {label} ═══")
    print(f"Instrumentos: {total} · positivos: {positives} ({positives / total:.0%})")
    print(f"Correlação média: {upper.mean():.3f}")
    print(f"Apostas efetivamente independentes: {effective:.1f}")
    print(f"Acaso produz: mediana {np.median(counts):.0f} · p95 {np.percentile(counts, 95):.0f}")
    print(f"p-valor: {p_value:.4f} → {'SIGNIFICATIVO' if p_value < 0.05 else 'não significativo'} a 5%")
    return {
        "label": label, "instruments": total, "positive": positives,
        "mean_correlation": round(float(upper.mean()), 3),
        "effective_bets": round(effective, 1),
        "p_value": round(p_value, 4),
        "significant": bool(p_value < 0.05),
    }


def main() -> None:
    store = HistoryStore()

    original, ret_original = run_universe(store, ORIGINAL)
    expanded, ret_expanded = run_universe(store, EXPANDED)

    print("═══ Donchian somente-compra · universo ampliado ═══")
    print(expanded.sort_values(["bloco", "sharpe_trade"], ascending=[True, False]).to_string(index=False))

    print("\n── Por bloco econômico ──")
    grouped = expanded.groupby("bloco")
    by_block = pd.DataFrame(
        {
            "instrumentos": grouped.size(),
            "positivos": grouped["positivo"].sum(),
            "sharpe_medio": grouped["sharpe_trade"].mean().round(3),
            "acerto_medio": (grouped["win_rate"].mean() * 100).round(1),
        }
    ).sort_values("sharpe_medio", ascending=False)
    print(by_block.to_string())

    summary_original = summarize("UNIVERSO ORIGINAL", original, ret_original)
    summary_expanded = summarize("UNIVERSO AMPLIADO", expanded, ret_expanded)

    print("\n── Veredito ──")
    if summary_expanded["p_value"] < summary_original["p_value"]:
        print(f"Ampliar o universo FORTALECEU a evidência: "
              f"p {summary_original['p_value']:.4f} → {summary_expanded['p_value']:.4f}")
    else:
        print(f"Ampliar o universo NÃO fortaleceu: "
              f"p {summary_original['p_value']:.4f} → {summary_expanded['p_value']:.4f}")
    print(f"Apostas independentes: {summary_original['effective_bets']} → "
          f"{summary_expanded['effective_bets']}")

    out = ROOT / "web" / "expanded_audit.json"
    out.write_text(
        json.dumps(
            {
                "original": summary_original,
                "expanded": summary_expanded,
                "by_block": by_block.reset_index().to_dict("records"),
                "per_symbol": expanded.to_dict("records"),
            },
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
