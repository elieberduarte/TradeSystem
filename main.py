"""Ponto de entrada do bot — a carteira Donchian em futuros da B3.

Comandos:
  python main.py status   mostra conta, posições e a última decisão
  python main.py plan     ciclo completo SEM enviar ordens (ensaio)
  python main.py run      ciclo completo COM ordens (17h40)
  python main.py morning  repesca sinais de ontem em mercados que já
                          estavam fechados (rodar ~9h05; descarta o
                          candle parcial de hoje)
  python main.py loop     agenda run às 17h40 e morning às 9h05, para
                          sempre (deixe numa janela de terminal)

Segurança: o bot verifica se a conta é DEMO antes de qualquer ordem.
Conta real exige `allow_real: true` no config.yaml — e não recomendo
ligar isso antes de meses de paper trading limpo.
"""

import json
import sys
import time as time_module
from datetime import datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.bot.data.history import HistoryStore
from src.bot.execution.mt5_broker import MT5Broker
from src.bot.runner import Runner, RunnerConfig

CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config() -> RunnerConfig:
    if not CONFIG_PATH.exists():
        return RunnerConfig()
    import yaml

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    section = raw.get("carteira", {})
    known = set(RunnerConfig.__dataclass_fields__)
    return RunnerConfig(**{k: v for k, v in section.items() if k in known})


def build_runner() -> tuple[Runner, MT5Broker]:
    broker = MT5Broker()
    broker.connect()
    runner = Runner(
        broker=broker,
        store=HistoryStore(ROOT / "data"),
        config=load_config(),
        journal_path=ROOT / "data" / "journal.jsonl",
        status_path=ROOT / "web" / "live_status.json",
    )
    return runner, broker


def print_decisions(decisions) -> None:
    if not decisions:
        print("Nenhuma decisão hoje: sem sinais, sem rolagens, sem vetos.")
        return
    for d in decisions:
        line = f"  [{d.action:>9}] {d.symbol:<8} {d.reason}"
        if d.quantity:
            line += f" · {d.quantity} contrato(s)"
        print(line)


def cmd_status() -> None:
    runner, broker = build_runner()
    account_kind = "DEMO" if broker.is_demo() else "⚠️ REAL"
    print(f"Conta {account_kind} · saldo R$ {broker.get_balance():,.2f}")
    positions = broker.get_open_positions()
    if positions:
        print(f"\n{len(positions)} posição(ões) aberta(s):")
        for p in positions:
            print(f"  {p.symbol:<8} {p.side} {p.quantity:g} @ {p.entry_price:,.2f} "
                  f"· stop {p.stop_loss:,.2f} · alvo {p.take_profit:,.2f} "
                  f"· aberto {p.unrealized_pnl:+,.2f}")
    else:
        print("\nSem posições abertas.")
    print(f"\nPnL realizado na semana (ordens do bot): "
          f"R$ {broker.realized_pnl(since_days=7):+,.2f}")
    broker.disconnect()


def cmd_cycle(execute: bool, include_today: bool) -> None:
    runner, broker = build_runner()
    label = "EXECUTANDO" if execute else "ENSAIO (nenhuma ordem será enviada)"
    print(f"Ciclo de {datetime.now():%d/%m/%Y %H:%M} — {label}\n")
    decisions = runner.cycle(execute=execute, include_today=include_today)
    print_decisions(decisions)
    print(f"\nRegistrado em data/journal.jsonl e web/live_status.json")
    broker.disconnect()


def cmd_loop() -> None:
    """Agenda simples: morning às 9h05, run às 17h40, dias úteis."""
    schedule = [(time(9, 5), False), (time(17, 40), True)]
    print("Loop agendado: 9h05 (repesca) e 17h40 (ciclo principal). Ctrl+C para sair.")
    while True:
        now = datetime.now()
        candidates = []
        for at, include_today in schedule:
            target = datetime.combine(now.date(), at)
            if target <= now:
                target += timedelta(days=1)
            candidates.append((target, include_today))
        target, include_today = min(candidates)
        while target.weekday() >= 5:                   # pula fim de semana
            target += timedelta(days=1)
        wait = (target - datetime.now()).total_seconds()
        print(f"Próximo ciclo: {target:%d/%m %H:%M} "
              f"({'principal' if include_today else 'repesca'}) — dormindo…")
        time_module.sleep(max(wait, 1))
        try:
            cmd_cycle(execute=True, include_today=include_today)
        except Exception as exc:  # noqa: BLE001 — o loop não morre por um ciclo ruim
            print(f"Ciclo falhou: {exc}")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "status":
        cmd_status()
    elif command == "plan":
        cmd_cycle(execute=False, include_today=True)
    elif command == "run":
        cmd_cycle(execute=True, include_today=True)
    elif command == "morning":
        cmd_cycle(execute=True, include_today=False)
    elif command == "loop":
        cmd_loop()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
