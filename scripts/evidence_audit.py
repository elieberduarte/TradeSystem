"""Auditoria da nossa principal evidência.

Duas correções que a pesquisa apontou e que atacam diretamente a
conclusão "donchian replica em 11 de 13 instrumentos":

1. INDEPENDÊNCIA EFETIVA — nossos 13 instrumentos são altamente
   correlacionados (WIN ≈ IND ≈ BOVA11 ≈ ações do IBOV; WDO ≈ DOL).
   Sob correlação alta, "11 de 13 positivos" é MUITO menos evidência
   do que sob independência: a distribuição nula é bem mais larga que
   a binomial. Aqui ela é simulada preservando a correlação real.

2. TAMANHO DE TICK — Kurth, Eisler, Rej & Bouchaud (CFM, 2026)
   mostram que o trend following de curto prazo morreu após 2009
   exatamente nos instrumentos de tick pequeno relativo à
   volatilidade, e que futuros de índice e câmbio caem nesse grupo.
   Se o achado valer aqui, nosso Donchian deve ir PIOR no WIN, IND,
   WDO e DOL do que nas ações e ETFs.

Uso: python scripts/evidence_audit.py
"""

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

CAPITAL = 150_000.0
POINT_VALUE = {"WIN$N": 0.20, "WDO$N": 10.00}
# Tick mínimo em unidades de preço do próprio instrumento
TICK_SIZE = {"WIN$N": 5.0, "WDO$N": 0.5}
FUTURES = {"WIN$N", "WDO$N"}
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


def run_all(store: HistoryStore) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda o Donchian em cada instrumento e devolve resultados e retornos diários."""
    rows, daily = [], {}
    for symbol in UNIVERSE:
        candles = store.load(symbol, "1d")
        if candles is None:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            DonchianStrategy(BASE), risk(), point_value=POINT_VALUE.get(symbol, 1.0),
            warmup=210, slippage_points=slippage, cost_per_contract=cost,
        )
        result = engine.run(symbol, candles)
        if not result.trades:
            continue

        returns = candles["close"].pct_change().dropna()
        tick = TICK_SIZE.get(symbol, 0.01)
        # ρ do paper: tick dividido pela volatilidade diária em unidades de preço
        daily_vol_price = float((returns.std() * candles["close"]).median())
        rho = tick / daily_vol_price if daily_vol_price > 0 else np.nan

        pnls = [t.pnl for t in result.trades]
        sharpe = float(np.mean(pnls) / np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
        rows.append(
            {
                "symbol": symbol,
                "tipo": "futuro" if symbol in FUTURES else "ação/ETF",
                "rho_x1000": round(rho * 1000, 3),
                "trades": len(result.trades),
                "pnl": round(result.total_pnl, 0),
                "sharpe_trade": round(sharpe, 3),
                "positivo": result.total_pnl > 0,
            }
        )
        daily[symbol] = returns
    return pd.DataFrame(rows), pd.DataFrame(daily)


def effective_bets(returns: pd.DataFrame) -> float:
    """Número efetivo de apostas independentes, via autovalores da correlação.

    Se todos os ativos fossem independentes, seria igual ao número de
    ativos. Quanto mais correlacionados, menor — e menos evidência
    "11 de 13" representa.
    """
    matrix = returns.dropna().corr().to_numpy()
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]
    # Entropia dos autovalores normalizados (número efetivo de fatores)
    weights = eigenvalues / eigenvalues.sum()
    return float(np.exp(-(weights * np.log(weights)).sum()))


def null_distribution(returns: pd.DataFrame, rounds: int = 2000, seed: int = 11) -> np.ndarray:
    """Quantos instrumentos ficariam positivos por acaso, dada a correlação real.

    Simula retornos com a mesma estrutura de correlação dos ativos, sem
    nenhuma vantagem embutida, e conta quantos terminam positivos.
    """
    clean = returns.dropna()
    corr = clean.corr().to_numpy()
    n_assets = corr.shape[0]
    n_days = len(clean)
    # Fatoração para gerar séries correlacionadas
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)
    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    rng = np.random.default_rng(seed)
    counts = np.empty(rounds, dtype=int)
    for i in range(rounds):
        noise = rng.standard_normal((n_days, n_assets))
        correlated = noise @ transform.T
        # Soma acumulada: quantos terminam acima de zero
        counts[i] = int((correlated.sum(axis=0) > 0).sum())
    return counts


def main() -> None:
    store = HistoryStore()
    results, returns = run_all(store)
    if results.empty:
        raise SystemExit("Sem resultados")

    print("═══ Auditoria da evidência · donchian somente-compra ═══\n")
    print(results.to_string(index=False))

    positives = int(results["positivo"].sum())
    total = len(results)
    print(f"\nPlacar bruto: {positives}/{total} positivos")

    # ── 1. Quanto vale esse placar sob a correlação real ──
    print("\n── Quanto vale '11 de 13' sob a correlação real? ──")
    effective = effective_bets(returns)
    print(f"Correlação média entre os ativos: {returns.corr().values[np.triu_indices(len(returns.columns), 1)].mean():.3f}")
    print(f"Apostas efetivamente independentes: {effective:.1f} (de {total} instrumentos)")

    null = null_distribution(returns)
    p_value = float((null >= positives).mean())
    print(f"Sob a hipótese nula (sem vantagem, mesma correlação):")
    print(f"  positivos por acaso: mediana {np.median(null):.0f} · "
          f"percentil 95 = {np.percentile(null, 95):.0f} · máximo {null.max()}")
    print(f"  p-valor de observar {positives} ou mais: {p_value:.3f}")
    veredito = "significativo" if p_value < 0.05 else "NÃO significativo"
    print(f"  → {veredito} a 5%")

    # ── 2. O achado do tamanho de tick ──
    print("\n── Tick pequeno mata a tendência? (Kurth et al., CFM 2026) ──")
    futures = results[results["tipo"] == "futuro"]
    stocks = results[results["tipo"] == "ação/ETF"]
    print(f"Futuros de índice/câmbio: {len(futures)} instrumentos · "
          f"Sharpe médio {futures['sharpe_trade'].mean():+.3f} · "
          f"{int(futures['positivo'].sum())}/{len(futures)} positivos")
    print(f"Ações e ETFs:              {len(stocks)} instrumentos · "
          f"Sharpe médio {stocks['sharpe_trade'].mean():+.3f} · "
          f"{int(stocks['positivo'].sum())}/{len(stocks)} positivos")

    if len(results) > 3:
        correlation = float(results["rho_x1000"].corr(results["sharpe_trade"]))
        print(f"\nCorrelação entre ρ (tick/volatilidade) e Sharpe: {correlation:+.3f}")
        if correlation > 0.3:
            print("  → mais tick relativo, mais desempenho: CONSISTENTE com o paper")
        elif correlation < -0.3:
            print("  → contradiz o paper")
        else:
            print("  → sem relação clara nesta amostra")


if __name__ == "__main__":
    main()
