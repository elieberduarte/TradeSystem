"""Operar com dinheiro emprestado: a aritmética.

A ideia é coerente em abstrato — se o retorno esperado supera o juro,
o spread é lucro. Mas há duas assimetrias que a conta simples esconde:

1. O retorno é ESPERADO, INCERTO e VOLÁTIL. O juro é CERTO, FIXO e
   devido todo mês, inclusive durante o drawdown.
2. A ruína é absorvente. Uma perda que a conta própria suporta vira
   liquidação forçada quando o capital é emprestado — e liquidação
   forçada acontece no pior momento por construção.

Este script simula a estratégia real (retorno e volatilidade medidos
no backtest de futuros) contra as taxas de crédito praticadas no
Brasil, e conta em que fração dos cenários o operador termina devendo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

# Medidos no backtest de futuros, melhor configuração
ANNUAL_RETURN = 0.109
MAX_DRAWDOWN = 0.129
# Volatilidade anual implícita: drawdown máximo costuma rodar entre 1,5x
# e 2,5x a volatilidade anual em estratégias de tendência
ANNUAL_VOL = 0.129 / 2.0

# Custo de crédito no Brasil (taxas mensais típicas, ago/2026)
CREDIT = {
    "consignado (o mais barato)": 0.020,
    "crédito pessoal com garantia": 0.035,
    "crédito pessoal sem garantia": 0.065,
    "cheque especial": 0.080,
}

MONTHS = 48
ROUNDS = 20_000


def simulate(monthly_rate: float, seed: int = 7) -> dict:
    """Simula 4 anos: retorno mensal aleatório contra parcela fixa."""
    rng = np.random.default_rng(seed)
    monthly_return = (1 + ANNUAL_RETURN) ** (1 / 12) - 1
    monthly_vol = ANNUAL_VOL / np.sqrt(12)

    # Parcela do empréstimo (sistema Price) sobre 1 unidade tomada
    if monthly_rate > 0:
        payment = monthly_rate / (1 - (1 + monthly_rate) ** -MONTHS)
    else:
        payment = 1 / MONTHS

    final, ruined, worst = [], 0, []
    for _ in range(ROUNDS):
        equity = 1.0          # capital operado = valor tomado
        owed_months = MONTHS
        low = 1.0
        for _ in range(MONTHS):
            equity *= 1 + rng.normal(monthly_return, monthly_vol)
            equity -= payment  # a parcela vence independente do resultado
            low = min(low, equity)
            owed_months -= 1
            if equity <= 0:
                ruined += 1
                break
        final.append(max(equity, 0.0))
        worst.append(low)

    final = np.array(final)
    total_paid = payment * MONTHS
    return {
        "parcela_mensal": payment,
        "total_pago": total_paid,
        "custo_efetivo_aa": (1 + monthly_rate) ** 12 - 1,
        "patrimonio_mediano": float(np.median(final)),
        "prob_ruina": ruined / ROUNDS,
        "prob_prejuizo": float((final < 1e-9).mean() + (final[final > 0] < 0).mean()),
        "prob_terminar_pior": float((final <= 0).mean()),
        "percentil_5": float(np.percentile(final, 5)),
        "percentil_95": float(np.percentile(final, 95)),
    }


def main() -> None:
    print("═══ Operar com crédito bancário: a aritmética ═══\n")
    print(f"Estratégia (medida no backtest de futuros):")
    print(f"  retorno esperado {ANNUAL_RETURN:.1%} ao ano")
    print(f"  volatilidade estimada {ANNUAL_VOL:.1%} ao ano")
    print(f"  drawdown máximo observado {MAX_DRAWDOWN:.1%}")
    print(f"\nHorizonte: {MONTHS} meses · {ROUNDS:,} cenários simulados\n")

    print(f"{'linha de crédito':<30} {'a.m.':>6} {'a.a.':>8} {'parcela':>9} "
          f"{'pago total':>11} {'ruína':>7} {'mediana':>9}")
    print("-" * 86)

    for label, rate in CREDIT.items():
        result = simulate(rate)
        print(
            f"{label:<30} {rate:>5.1%} {result['custo_efetivo_aa']:>7.1%} "
            f"{result['parcela_mensal']:>8.2%} {result['total_pago']:>10.2f}x "
            f"{result['prob_ruina']:>6.1%} {result['patrimonio_mediano']:>8.2f}x"
        )

    print("-" * 86)
    print("\nparcela = % do valor tomado, por mês · pago total = múltiplo do que foi tomado")
    print("mediana = patrimônio final, em múltiplos do valor tomado (1,00x = empatou)")

    print("\n── O ponto central ──")
    cheapest = min(CREDIT.values())
    cost_aa = (1 + cheapest) ** 12 - 1
    print(f"A linha MAIS BARATA custa {cost_aa:.1%} ao ano.")
    print(f"A estratégia rende {ANNUAL_RETURN:.1%} ao ano no backtest.")
    print(f"Diferença: {ANNUAL_RETURN - cost_aa:+.1%} ao ano — o crédito custa "
          f"{cost_aa / ANNUAL_RETURN:.1f}x o retorno.")

    print("\n── E se a estratégia rendesse muito mais? ──")
    print("Retorno anual necessário para apenas EMPATAR com cada linha:")
    for label, rate in CREDIT.items():
        needed = (1 + rate) ** 12 - 1
        print(f"  {label:<30} {needed:>6.1%} ao ano")
    print("\nPara comparação: o melhor track record público de robô de B3")
    print("(48.879 trades, 5,9 anos, verificado) rende ~32% ao ano com")
    print("drawdown de 35% — e nem ele cobriria crédito pessoal.")


if __name__ == "__main__":
    main()
