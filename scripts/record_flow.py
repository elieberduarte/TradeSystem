"""Grava o fluxo (book + ticks) dos contratos vigentes até o fim do pregão.

Resolve WIN/WDO para o contrato vigente pelo MT5, grava a cada
segundo e encerra sozinho no horário indicado (padrão 18:20, depois
do after-market do índice). Rode uma vez por pregão — junto com o
bot — e o acervo em data/flow/ cresce um dia por dia.

Uso: python scripts/record_flow.py [HH:MM de término]
"""

import sys
from datetime import datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.data.flow_recorder import FlowRecorder
from src.bot.execution.mt5_broker import MT5Broker


def main() -> None:
    end_text = sys.argv[1] if len(sys.argv) > 1 else "18:20"
    hh, mm = (int(x) for x in end_text.split(":"))
    until = datetime.combine(datetime.now().date(), time(hh, mm))

    broker = MT5Broker()
    broker.connect()
    symbols = [broker.front_contract(root) for root in ("WIN", "WDO")]
    broker.disconnect()

    print(f"{datetime.now():%H:%M:%S} gravando fluxo de {symbols} até {until:%H:%M} "
          f"(book a cada 1s + ticks com Lee-Ready) → data/flow/", flush=True)
    recorder = FlowRecorder(symbols=symbols, levels=5, book_interval=1.0, flush_every=120)
    recorder.run(until=until)
    print(f"{datetime.now():%H:%M:%S} encerrado; acervo gravado.", flush=True)


if __name__ == "__main__":
    main()
