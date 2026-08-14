"""Testes do ciclo de decisão do bot (plan_cycle é função pura)."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.bot.execution.broker import Position
from src.bot.runner import Decision, RunnerConfig, plan_cycle, position_root, root_of

NOW = datetime(2026, 8, 14, 17, 40)


def breakout_frame(
    n: int = 400, base: float = 100_000.0, amplitude: float = 500.0
) -> pd.DataFrame:
    """Histórico lateral que rompe o canal no último candle."""
    rows = []
    for i in range(n - 1):
        rows.append((base, base + amplitude, base - amplitude, base))
    rows.append((base, base + 4 * amplitude, base - amplitude / 5,
                 base + 3.6 * amplitude))                        # rompimento
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1_000.0] * n,
        },
        index=pd.date_range(end="2026-08-14", periods=n, freq="B"),
    )


def flat_frame(n: int = 400, base: float = 100_000.0) -> pd.DataFrame:
    frame = breakout_frame(n, base)
    frame.iloc[-1] = frame.iloc[0]  # sem rompimento
    return frame


def config(**overrides) -> RunnerConfig:
    values = {"symbols": ["WIN$N", "WDO$N"], "slots": 10}
    values.update(overrides)
    return RunnerConfig(**values)


def test_rompimento_gera_abertura_com_tamanho():
    decisions = plan_cycle(
        {"WIN$N": breakout_frame()}, [], {}, 0.0, config(), NOW
    )
    opens = [d for d in decisions if d.action == "abrir"]
    assert len(opens) == 1
    d = opens[0]
    assert d.symbol == "WIN$N"
    assert d.quantity >= 1
    assert d.stop_distance > 0
    assert d.target_distance == 3.0 * d.stop_distance   # rr da config


def test_sem_rompimento_nao_ha_decisao():
    decisions = plan_cycle({"WIN$N": flat_frame()}, [], {}, 0.0, config(), NOW)
    assert decisions == []


def test_posicao_existente_silencia_o_sinal():
    position = Position(symbol="WINV26", side="buy", quantity=1,
                        entry_price=100_000, unrealized_pnl=0)
    decisions = plan_cycle(
        {"WIN$N": breakout_frame()}, [position], {"WINV26": None}, 0.0, config(), NOW
    )
    assert [d for d in decisions if d.action == "abrir"] == []


def test_perda_semanal_veta_tudo():
    decisions = plan_cycle(
        {"WIN$N": breakout_frame()}, [], {},
        weekly_pnl=-7_000.0,        # limite: 6% de 100k = 6.000
        config=config(), now=NOW,
    )
    assert len(decisions) == 1
    assert decisions[0].action == "vetar"
    assert "semanal" in decisions[0].reason


def test_vencimento_proximo_gera_rolagem():
    position = Position(symbol="WINQ26", side="buy", quantity=2,
                        entry_price=100_000, unrealized_pnl=0)
    expiry = NOW.astimezone(timezone.utc) + timedelta(days=2)
    decisions = plan_cycle(
        {"WIN$N": flat_frame()}, [position], {"WINQ26": expiry}, 0.0, config(), NOW
    )
    rolls = [d for d in decisions if d.action == "rolar"]
    assert len(rolls) == 1
    assert rolls[0].symbol == "WINQ26"
    assert rolls[0].quantity == 2


def test_vencimento_distante_nao_rola():
    position = Position(symbol="WINV26", side="buy", quantity=1,
                        entry_price=100_000, unrealized_pnl=0)
    expiry = NOW.astimezone(timezone.utc) + timedelta(days=40)
    decisions = plan_cycle(
        {"WIN$N": flat_frame()}, [position], {"WINV26": expiry}, 0.0, config(), NOW
    )
    assert [d for d in decisions if d.action == "rolar"] == []


def test_disputa_de_vagas_respeita_a_regra_de_bloco():
    # Uma vaga livre; sinais em WIN (índice BR) e WDO (câmbio), com
    # posição já aberta em IND (índice BR) → WDO diversifica e vence
    position = Position(symbol="INDV26", side="buy", quantity=1,
                        entry_price=100_000, unrealized_pnl=0)
    decisions = plan_cycle(
        {"WIN$N": breakout_frame(),
         "WDO$N": breakout_frame(base=5_000.0, amplitude=20.0)},
        [position], {"INDV26": None}, 0.0,
        config(symbols=["WIN$N", "WDO$N", "IND$N"], slots=2), NOW,
    )
    opens = [d for d in decisions if d.action == "abrir"]
    watching = [d for d in decisions if d.action == "observar"]
    assert [d.symbol for d in opens] == ["WDO$N"]
    assert [d.symbol for d in watching] == ["WIN$N"]
    assert "sem vaga" in watching[0].reason


def test_historico_curto_e_ignorado():
    decisions = plan_cycle(
        {"WIN$N": breakout_frame(n=100)}, [], {}, 0.0, config(), NOW
    )
    assert decisions == []


def test_mapeamento_de_raizes():
    assert root_of("WIN$N") == "WIN"
    assert root_of("DI1F27") == "DI1F27"
    roots = ["WIN", "WDO", "DI1F27", "T10"]
    assert position_root("WINV26", roots) == "WIN"
    assert position_root("WDOU26", roots) == "WDO"
    assert position_root("DI1F27", roots) == "DI1F27"
    assert position_root("T10U26", roots) == "T10"
    assert position_root("PETR4", roots) is None
