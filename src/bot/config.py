"""Validações da configuração do bot."""


def validate_symbol_modes(assignments: list[tuple[str, str]]) -> None:
    """Garante que nenhum símbolo é operado em day trade E swing trade.

    A conta MT5 da B3 mantém uma única posição líquida por símbolo
    (netting): day e swing no mesmo ativo se anulariam mutuamente.
    `assignments` é uma lista de pares (símbolo, modo).
    """
    modes_by_symbol: dict[str, set[str]] = {}
    for symbol, mode in assignments:
        modes_by_symbol.setdefault(symbol.upper(), set()).add(mode)

    conflicting = [s for s, modes in modes_by_symbol.items() if len(modes) > 1]
    if conflicting:
        raise ValueError(
            f"Símbolo(s) {conflicting} configurado(s) em day trade e swing trade "
            "ao mesmo tempo — a conta B3 é netting (uma posição líquida por "
            "símbolo) e os modos se anulariam. Use símbolos diferentes."
        )
