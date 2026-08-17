"""Baixa o diário de TODOS os papéis líquidos do censo (universe_scan).

Corte: mediana ≥ R$ 5 mi/dia (o mesmo da carteira). Papéis já no
acervo são só atualizados (merge incremental). Grava a lista final em
data/liquid_universe.json para a bateria de replicação.

Uso: python scripts/download_liquid.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.data.history import HistoryStore
from src.bot.execution.mt5_broker import MT5Broker

ROOT = Path(__file__).resolve().parents[1]
MIN_VOLUME = 5e6


def main() -> None:
    scan = json.loads((ROOT / "web" / "universe_scan.json").read_text(encoding="utf-8"))
    liquid = sorted(r["symbol"] for r in scan if r["volume_mediano"] >= MIN_VOLUME)
    print(f"{len(liquid)} papéis líquidos no censo")

    store = HistoryStore(ROOT / "data")
    broker = MT5Broker()
    broker.connect()
    ok, failed = [], []
    for i, symbol in enumerate(liquid, 1):
        try:
            frame = store.update_from_broker(broker, symbol, "1d", limit=5_000)
            ok.append(symbol)
            if i % 20 == 0:
                print(f"  {i}/{len(liquid)} · {symbol}: {len(frame)} pregões", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append((symbol, str(exc)[:60]))
    broker.disconnect()

    (ROOT / "data" / "liquid_universe.json").write_text(
        json.dumps({"symbols": ok, "min_volume": MIN_VOLUME}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\n{len(ok)} baixados · {len(failed)} falharam")
    for symbol, why in failed[:10]:
        print(f"  {symbol}: {why}")


if __name__ == "__main__":
    main()
