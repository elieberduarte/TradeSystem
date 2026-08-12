"""Mescla arquivos de resultados no web/results.json do painel.

Uso: python scripts/merge_results.py <arquivo.json> [outro.json ...]

Normaliza as chaves para "estrategia · SIMBOLO timeframe", de modo que
rodadas em bases diferentes coexistam no comparativo.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "web" / "results.json"


def normalize(payload: dict) -> dict:
    out = {}
    for key, s in payload.get("strategies", {}).items():
        label = key if "·" in key else f"{s['strategy']} · {s['symbol']} {s['timeframe']}"
        out[label] = s
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    merged = {}
    base = None
    for path in [TARGET, *[Path(a) for a in sys.argv[1:]]]:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        base = base or payload
        merged.update(normalize(payload))

    if base is None:
        raise SystemExit("Nenhum arquivo válido encontrado")

    base["strategies"] = merged
    TARGET.write_text(json.dumps(base, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(merged)} backtests no painel:")
    for label, s in merged.items():
        print(f"  {label}: R$ {s['oos_pnl']:+,.2f} em {s['trades']} trades")


if __name__ == "__main__":
    main()
