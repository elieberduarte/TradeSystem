"""Buscador de notícias ao vivo → web/news_watch.json.

Eventos CVM dos últimos N dias nos papéis líquidos, com a reação de
preço já calculada e os sinais PEAD armados; manchetes RSS só para os
papéis com evento recente. Roda em segundos além do cache da CVM.

Uso: python scripts/news_watch.py [dias=10]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.data.history import HistoryStore
from src.bot.data.news_watch import build_feed, headlines, to_payload

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    liquid = json.loads((ROOT / "data" / "liquid_universe.json").read_text(encoding="utf-8"))["symbols"]
    tickers = [s for s in liquid if s[-2:] not in ("32", "33", "34", "35", "39")]

    store = HistoryStore(ROOT / "data")
    feed = build_feed(store, tickers, days, ROOT / "data" / "cvm")
    signals = [r for r in feed if r.sinal_pead]

    print(f"Últimos {days} dias · {len(feed)} eventos em {len({r.ticker for r in feed})} papéis "
          f"· {len(signals)} sinal(is) PEAD armado(s)\n")
    for r in signals:
        print(f"  🟢 {r.ticker:<7} {r.data_entrega}  {r.motivo}")
    print("\nResultados recentes (todos):")
    for r in [x for x in feed if x.tipo == "resultado"][:12]:
        mark = "🟢" if r.sinal_pead else "  "
        reac = f"{r.reacao_atr:+.2f} ATR" if r.reacao_atr is not None else "aguardando"
        print(f"  {mark} {r.ticker:<7} {r.data_entrega}  {reac:<12} {r.assunto[:60]}")

    watch = sorted({r.ticker for r in feed if r.tipo in ("resultado", "fato")})[:25]
    news = {t: headlines(t) for t in watch}
    payload = to_payload(feed, news)
    out = ROOT / "web" / "news_watch.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nManchetes coletadas para {sum(1 for v in news.values() if v)} papéis · exportado para {out}")


if __name__ == "__main__":
    main()
