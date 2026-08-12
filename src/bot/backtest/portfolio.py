"""Combinação de estratégias em carteira.

A pergunta que este módulo responde: vale mais escolher a melhor
estratégia ou operar várias ao mesmo tempo?

A resposta depende da correlação entre elas. Estratégias pouco
correlacionadas somam retorno e cancelam parte da oscilação — é o
único "almoço grátis" reconhecido em finanças (Markowitz). Duas
estratégias com o mesmo retorno e correlação zero, combinadas meio a
meio, mantêm o retorno e reduzem a volatilidade em ~30%.

Alternativa comum e pior: alternar entre estratégias conforme o
regime. Isso exige prever o regime — cada troca é uma aposta que pode
errar — enquanto operar em paralelo captura o benefício sem prever
nada.
"""

from dataclasses import dataclass

import pandas as pd

from src.bot.backtest.engine import Trade


def daily_pnl(trades: list[Trade]) -> pd.Series:
    """Resultado por dia de saída, indexado por data."""
    if not trades:
        return pd.Series(dtype=float)
    rows: dict[pd.Timestamp, float] = {}
    for trade in trades:
        if trade.exit_time is None:
            continue
        day = pd.Timestamp(trade.exit_time).normalize()
        rows[day] = rows.get(day, 0.0) + trade.pnl
    return pd.Series(rows).sort_index()


def correlation_matrix(results: dict[str, list[Trade]]) -> pd.DataFrame:
    """Correlação entre os resultados diários das estratégias.

    Valores próximos de 0 (ou negativos) indicam que elas ganham em
    momentos diferentes — exatamente o que faz a carteira valer a pena.
    """
    series = {name: daily_pnl(trades) for name, trades in results.items() if trades}
    if len(series) < 2:
        return pd.DataFrame()
    # União das datas: dia sem trade de uma estratégia conta como zero
    frame = pd.DataFrame(series).fillna(0.0)
    return frame.corr().round(3)


@dataclass
class PortfolioResult:
    equity: pd.Series
    total_pnl: float
    max_drawdown: float
    daily_std: float
    calmar: float
    days: int

    def summary(self) -> str:
        return (
            f"PnL {self.total_pnl:,.0f} | Drawdown máx {self.max_drawdown:,.0f} | "
            f"Calmar {self.calmar:.2f} | Oscilação diária ±{self.daily_std:,.0f} | "
            f"{self.days} dias com resultado"
        )


def combine(results: dict[str, list[Trade]], weights: dict[str, float] | None = None) -> PortfolioResult:
    """Soma as estratégias em uma carteira, com pesos opcionais.

    Peso 0,5 significa metade do risco alocado àquela estratégia — o
    equivalente a dividir o capital entre elas.
    """
    series = {name: daily_pnl(trades) for name, trades in results.items() if trades}
    if not series:
        return PortfolioResult(pd.Series(dtype=float), 0.0, 0.0, 0.0, 0.0, 0)

    frame = pd.DataFrame(series).fillna(0.0)
    if weights:
        for name in frame.columns:
            frame[name] *= weights.get(name, 1.0)
    combined = frame.sum(axis=1).sort_index()
    equity = combined.cumsum()

    peak, max_dd = float("-inf"), 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)

    total = float(equity.iloc[-1])
    return PortfolioResult(
        equity=equity,
        total_pnl=total,
        max_drawdown=max_dd,
        daily_std=float(combined.std()),
        calmar=total / max_dd if max_dd > 0 else float("inf"),
        days=len(combined),
    )


def equal_risk_weights(results: dict[str, list[Trade]]) -> dict[str, float]:
    """Pesos que igualam a contribuição de risco de cada estratégia.

    Divide pela volatilidade de cada uma: a mais errática entra com
    peso menor, para nenhuma dominar o resultado da carteira.
    """
    volatility = {}
    for name, trades in results.items():
        series = daily_pnl(trades)
        std = float(series.std()) if len(series) > 1 else 0.0
        if std > 0:
            volatility[name] = std
    if not volatility:
        return {}
    inverse = {name: 1.0 / std for name, std in volatility.items()}
    total = sum(inverse.values())
    return {name: round(value / total * len(inverse), 3) for name, value in inverse.items()}
