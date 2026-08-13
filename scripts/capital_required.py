"""Quanto capital a estratégia realmente exige?

Nosso dimensionamento calcula a quantidade pelo RISCO (1% do capital
dividido pela distância do stop). Isso responde "quanto posso perder",
não "quanto preciso ter na conta".

No mercado à vista a diferença é decisiva: comprar 1.000 ações de
R$41 custa R$41.000, ainda que o risco seja de apenas R$1.500. Sem
alavancagem para swing, é o VALOR FINANCEIRO da posição que precisa
caber no caixa.

Este script mede a exposição financeira real da carteira ao longo do
tempo e diz qual capital seria necessário para operá-la de verdade.
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
from src.bot.universe import EXPANDED, POINT_VALUE, block_of

CAPITAL = 150_000.0
BASE = {"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}

# Futuros exigem apenas margem, não o valor cheio. Margens de day trade
# são menores, mas swing exige margem cheia — valores aproximados.
MARGEM_FUTURO = {"WIN$N": 2_000.0, "WDO$N": 5_000.0}
FUTUROS = set(MARGEM_FUTURO) | {"DI1F27", "DI1F28", "DI1F29", "CCM$N", "BGI$N", "ICF$N"}


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
        return 0.005, 0.01
    if symbol.endswith("$N"):
        return 0.5, 1.0
    return 0.01, 0.01


def main() -> None:
    store = HistoryStore()
    positions = []

    for symbol in EXPANDED:
        candles = store.load(symbol, "1d")
        if candles is None or len(candles) < 700:
            continue
        slippage, cost = friction(symbol)
        engine = BacktestEngine(
            DonchianStrategy(BASE), risk(), point_value=POINT_VALUE.get(symbol, 1.0),
            warmup=210, slippage_points=slippage, cost_per_contract=cost,
        )
        for trade in engine.run(symbol, candles).trades:
            # Ações e ETFs: valor financeiro cheio. Futuros: margem.
            if symbol in FUTUROS:
                exposure = MARGEM_FUTURO.get(symbol, 2_000.0) * trade.quantity
                kind = "margem"
            else:
                exposure = trade.entry_price * trade.quantity
                kind = "à vista"
            positions.append(
                {
                    "symbol": symbol,
                    "bloco": block_of(symbol),
                    "tipo": kind,
                    "entrada": pd.Timestamp(trade.entry_time).normalize(),
                    "saida": pd.Timestamp(trade.exit_time).normalize(),
                    "quantidade": trade.quantity,
                    "preco": trade.entry_price,
                    "exposicao": exposure,
                    "risco": abs(trade.entry_price - trade.stop_loss) * trade.quantity
                    * POINT_VALUE.get(symbol, 1.0),
                }
            )

    frame = pd.DataFrame(positions)
    if frame.empty:
        raise SystemExit("Sem posições")

    print("═══ Capital realmente necessário ═══")
    print(f"Backtest feito com capital nominal de R$ {CAPITAL:,.0f} e risco de 1% por operação\n")

    spot = frame[frame["tipo"] == "à vista"]
    print("── Exposição por operação (mercado à vista) ──")
    print(f"Valor financeiro médio por posição: R$ {spot['exposicao'].mean():,.0f}")
    print(f"  mediana R$ {spot['exposicao'].median():,.0f} · "
          f"máximo R$ {spot['exposicao'].max():,.0f}")
    print(f"Risco médio por posição: R$ {spot['risco'].mean():,.0f} "
          f"({spot['risco'].mean() / spot['exposicao'].mean():.1%} da exposição)")

    # ── Exposição simultânea, dia a dia ──
    days = pd.date_range(frame["entrada"].min(), frame["saida"].max(), freq="B")
    exposure = pd.Series(0.0, index=days)
    for _, row in frame.iterrows():
        exposure.loc[row["entrada"] : row["saida"]] += row["exposicao"]

    print("\n── Capital imobilizado ao mesmo tempo ──")
    for pct in (50, 75, 90, 95, 99, 100):
        value = np.percentile(exposure, pct)
        print(f"  percentil {pct:>3}: R$ {value:>12,.0f} "
              f"({value / CAPITAL:>6.0%} do capital do backtest)")

    peak = exposure.max()
    print(f"\nPico de capital exigido: R$ {peak:,.0f}")
    print(f"O backtest assumiu R$ {CAPITAL:,.0f} — ou seja, ele operou como se houvesse")
    print(f"{peak / CAPITAL:.1f}x de alavancagem, que NÃO existe para swing no à vista.")

    # ── Qual capital tornaria a carteira executável ──
    print("\n── Capital necessário para operar de verdade ──")
    for target_pct in (100, 90, 75):
        needed = np.percentile(exposure, target_pct)
        print(f"  cobrir {target_pct}% dos dias: R$ {needed:,.0f}")

    print("\n── Versão enxuta: só os blocos de melhor desempenho ──")
    best_blocks = {"alternativos", "exterior", "índices BR", "juros"}
    lean = frame[frame["bloco"].isin(best_blocks)]
    lean_exposure = pd.Series(0.0, index=days)
    for _, row in lean.iterrows():
        lean_exposure.loc[row["entrada"] : row["saida"]] += row["exposicao"]
    print(f"Instrumentos: {lean['symbol'].nunique()} · operações: {len(lean)}")
    print(f"Capital no percentil 95: R$ {np.percentile(lean_exposure, 95):,.0f}")
    print(f"Pico: R$ {lean_exposure.max():,.0f}")

    print("\n── Por instrumento: exposição típica de UMA posição ──")
    per_symbol = frame.groupby(["bloco", "symbol"]).agg(
        preco_medio=("preco", "mean"),
        qtd_mediana=("quantidade", "median"),
        exposicao_mediana=("exposicao", "median"),
    ).round(0).sort_values("exposicao_mediana", ascending=False)
    print(per_symbol.head(15).to_string())

    # ── A conta invertida: dado o caixa, quanto se pode operar ──
    print("\n" + "=" * 66)
    print("A CONTA QUE IMPORTA: partindo do caixa disponível")
    print("=" * 66)
    print("\nNo à vista, o risco de cada posição é uma fração do valor investido.")
    ratio = float((spot["risco"] / spot["exposicao"]).median())
    print(f"Nesta estratégia, a mediana é {ratio:.1%} — um stop de 2×ATR fica a")
    print(f"{ratio:.1%} do preço de entrada.\n")
    print("Então, dividindo o caixa entre N posições simultâneas:\n")
    print(f"{'caixa':>12} {'posições':>9} {'R$/posição':>12} {'risco/posição':>14} "
          f"{'% do caixa':>11} {'risco total':>12}")
    print("-" * 76)
    for cash in (20_000, 50_000, 100_000, 200_000, 500_000):
        for slots in (5, 10):
            per_position = cash / slots
            risk_each = per_position * ratio
            print(
                f"{cash:>12,.0f} {slots:>9} {per_position:>12,.0f} "
                f"{risk_each:>14,.0f} {risk_each / cash:>10.2%} "
                f"{risk_each * slots:>12,.0f}"
            )
    print("\nLeitura: com o caixa dividido em 10 posições, cada trade arrisca")
    print(f"~{ratio / 10:.2%} do capital — bem abaixo do 1% que o backtest assumiu.")
    print("A estratégia é operável em conta pequena; o que muda é a escala do")
    print("resultado, não a validade do sinal. Os percentuais de retorno do")
    print("backtest permanecem válidos; os valores em reais, não.")


if __name__ == "__main__":
    main()
