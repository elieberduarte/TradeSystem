"""Resolução dos contratos futuros vigentes da B3 (WIN e WDO).

Os símbolos de futuros mudam a cada vencimento (ex.: WINQ26 → WINV26).
Este módulo calcula o contrato "front" (mais líquido) para uma data,
permitindo configurar apenas "WIN" ou "WDO" no config.yaml.

Regras de vencimento da B3:
- WIN (mini-índice): vence na quarta-feira mais próxima do dia 15 dos
  meses pares (G=fev, J=abr, M=jun, Q=ago, V=out, Z=dez).
- WDO (mini-dólar): vence no primeiro dia útil do mês de referência,
  em todos os meses — o contrato negociado em agosto é o de setembro (U).
"""

from datetime import date, timedelta

# Códigos de mês padrão dos futuros (janeiro a dezembro)
MONTH_CODES = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]

# Meses de vencimento do WIN (pares)
WIN_MONTHS = [2, 4, 6, 8, 10, 12]


def win_expiry(year: int, month: int) -> date:
    """Quarta-feira mais próxima do dia 15 do mês de vencimento."""
    base = date(year, month, 15)
    for offset in [0, 1, -1, 2, -2, 3, -3]:
        candidate = base + timedelta(days=offset)
        if candidate.weekday() == 2:  # quarta-feira
            return candidate
    raise RuntimeError("unreachable")


def win_contract(ref: date) -> str:
    """Contrato vigente do mini-índice na data `ref` (ex.: 'WINQ26')."""
    for months_ahead in range(0, 14):
        month = ref.month + months_ahead
        year = ref.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        if month in WIN_MONTHS and ref < win_expiry(year, month):
            return f"WIN{MONTH_CODES[month - 1]}{year % 100:02d}"
    raise RuntimeError("unreachable")


def wdo_contract(ref: date) -> str:
    """Contrato vigente do mini-dólar na data `ref` (ex.: 'WDOU26').

    O contrato do mês M vence no 1º dia útil de M, então o negociado
    durante o mês corrente é sempre o do mês seguinte.
    """
    month = ref.month + 1
    year = ref.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return f"WDO{MONTH_CODES[month - 1]}{year % 100:02d}"


def wdo_expiry(year: int, month: int) -> date:
    """Primeiro dia útil do mês de vencimento (sem considerar feriados)."""
    day = date(year, month, 1)
    while day.weekday() >= 5:  # sábado/domingo
        day += timedelta(days=1)
    return day


def days_to_expiry(symbol: str, ref: date | None = None) -> int:
    """Dias corridos até o vencimento do contrato vigente de WIN ou WDO.

    Essencial para swing: posição perto do vencimento precisa ser rolada
    para o contrato seguinte antes que a liquidez migre.
    """
    ref = ref or date.today()
    base = symbol.upper()
    if base == "WIN":
        contract = win_contract(ref)
        month = MONTH_CODES.index(contract[3]) + 1
        year = 2000 + int(contract[4:6])
        return (win_expiry(year, month) - ref).days
    if base == "WDO":
        contract = wdo_contract(ref)
        month = MONTH_CODES.index(contract[3]) + 1
        year = 2000 + int(contract[4:6])
        return (wdo_expiry(year, month) - ref).days
    raise ValueError(f"days_to_expiry suporta WIN e WDO, não '{symbol}'")


def resolve_symbol(symbol: str, ref: date | None = None) -> str:
    """Converte 'WIN'/'WDO' no contrato vigente; outros símbolos passam direto."""
    ref = ref or date.today()
    if symbol.upper() == "WIN":
        return win_contract(ref)
    if symbol.upper() == "WDO":
        return wdo_contract(ref)
    return symbol
