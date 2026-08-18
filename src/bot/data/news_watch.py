"""Buscador de notícias ao vivo: CVM (a fonte com edge) + RSS (contexto).

Duas camadas, com pesos diferentes de propósito:

  CVM / IPE   os eventos que o estudo mediu: comunicado de RESULTADO
              seguido de reação >= +1 ATR gera sinal PEAD comprador
              (deriva +0,556 ATR em 14 pregões, t=4,38 com custos,
              66% de replicação). Fatos relevantes entram no feed
              como informação — o estudo não achou deriva neles.
  RSS         manchetes por ticker (Google News). Só contexto: nunca
              vira sinal por si — não há edge medido em manchete.

Roda no ciclo diário: baixa a IPE do ano corrente (cache), filtra
os últimos N dias, casa com o acervo de preços para calcular a
reação, e escreve web/news_watch.json para o painel.
"""

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.bot.data.cvm import load_events, ticker_map
from src.bot.strategies.swing_reversion import atr

REACTION_MIN = 1.0
HOLD_DAYS = 14


@dataclass
class EventRow:
    ticker: str
    data_entrega: str
    tipo: str            # "resultado" | "fato" | "provento" | "oferta" | "outro"
    assunto: str
    link: str
    reacao_atr: float | None
    dias_apos: int | None
    sinal_pead: bool
    motivo: str


def classify(categoria: str) -> str:
    c = (categoria or "").lower()
    if "econômico-financeiros" in c or "economico-financeiros" in c:
        return "resultado"
    if "fato relevante" in c:
        return "fato"
    if "provento" in c or "dividendo" in c or "juros sobre" in c:
        return "provento"
    if "oferta" in c:
        return "oferta"
    return "outro"


def recent_events(tickers: list[str], days: int, cache_dir: Path) -> pd.DataFrame:
    """Eventos IPE dos últimos `days` dias para os tickers (ano corrente + anterior)."""
    year = datetime.now().year
    # O zip IPE do ano corrente cresce todo dia: apaga o cache dele
    # para a CVM ser lida fresca (os anos passados ficam em cache).
    current = cache_dir / f"ipe_{year}.zip"
    if current.exists() and datetime.now().timestamp() - current.stat().st_mtime > 6 * 3600:
        current.unlink()
    mapping = ticker_map(range(year - 1, year + 1), cache_dir)
    wanted = mapping[mapping["ticker"].isin([t.upper() for t in tickers])]
    events = load_events(range(year - 1, year + 1), cache_dir)
    since = pd.Timestamp(datetime.now().date() - timedelta(days=days))
    events = events[events["data_entrega"] >= since]
    merged = events.merge(wanted[["cnpj", "ticker"]], on="cnpj")
    merged["tipo"] = merged["categoria"].map(classify)
    return merged.sort_values("data_entrega", ascending=False)


def reaction_for(daily: pd.DataFrame, delivered: pd.Timestamp) -> tuple[float | None, int | None]:
    """Reação D→D+1 em ATR e pregões decorridos desde D+1 (None se cedo demais)."""
    if daily is None or len(daily) < 30:
        return None, None
    idx = daily.index
    pos = idx.get_indexer([delivered.normalize()], method="bfill")[0]
    if pos < 15 or pos + 1 >= len(idx):
        return None, None
    vol = float(atr(daily, 14).shift(1).iloc[pos + 1])
    if not vol or np.isnan(vol):
        return None, None
    reaction = float((daily["close"].iloc[pos + 1] - daily["close"].iloc[pos]) / vol)
    return reaction, int(len(idx) - 1 - (pos + 1))


def build_feed(store, tickers: list[str], days: int, cache_dir: Path) -> list[EventRow]:
    events = recent_events(tickers, days, cache_dir)
    seen: set[tuple[str, str, str]] = set()
    rows: list[EventRow] = []
    for _, ev in events.iterrows():
        key = (ev["ticker"], str(ev["data_entrega"].date()), ev["tipo"])
        if key in seen:
            continue
        seen.add(key)
        daily = store.load(ev["ticker"], "1d")
        reaction, elapsed = reaction_for(daily, ev["data_entrega"])
        signal, why = False, ""
        if ev["tipo"] == "resultado":
            if reaction is None:
                why = "resultado entregue; reação ainda não fechou (aguardar D+1)"
            elif reaction >= REACTION_MIN and elapsed is not None and elapsed <= HOLD_DAYS:
                signal, why = True, (f"resultado + reação {reaction:+.2f} ATR ≥ +1 → sinal PEAD "
                                     f"comprador (dia {elapsed} de {HOLD_DAYS})")
            elif reaction >= REACTION_MIN:
                why = f"reação {reaction:+.2f} ATR mas janela de {HOLD_DAYS} pregões já passou"
            else:
                why = f"reação {reaction:+.2f} ATR < +1 — sem sinal (estudo: só a reação forte deriva)"
        elif ev["tipo"] == "fato":
            why = "fato relevante — informativo (estudo: sem deriva mensurável)"
        elif ev["tipo"] == "oferta":
            why = "documento de oferta pública — vigia de IPO/follow-on"
        rows.append(EventRow(
            ticker=ev["ticker"], data_entrega=str(ev["data_entrega"].date()),
            tipo=ev["tipo"], assunto=str(ev.get("assunto") or "")[:140],
            link=str(ev.get("link") or ""), reacao_atr=None if reaction is None else round(reaction, 2),
            dias_apos=elapsed, sinal_pead=signal, motivo=why,
        ))
    return rows


def headlines(ticker: str, limit: int = 6, timeout: int = 15) -> list[dict]:
    """Manchetes recentes por ticker via RSS do Google News (só contexto)."""
    query = urllib.parse.quote(f"{ticker} ações")
    url = f"https://news.google.com/rss/search?q={query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "trade-bot/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            root = ET.fromstring(response.read())
    except Exception:  # noqa: BLE001 — sem rede/RSS não derruba o ciclo
        return []
    items = []
    for item in root.iter("item"):
        title = html.unescape(item.findtext("title") or "")
        title = re.sub(r"\s+-\s+[^-]+$", "", title)      # remove " - Fonte" do fim
        items.append({"titulo": title, "link": item.findtext("link") or "",
                      "data": item.findtext("pubDate") or ""})
        if len(items) >= limit:
            break
    return items


def to_payload(feed: list[EventRow], news: dict[str, list[dict]]) -> dict:
    return {
        "atualizado": datetime.now().isoformat(timespec="seconds"),
        "sinais_pead": [asdict(r) for r in feed if r.sinal_pead],
        "eventos": [asdict(r) for r in feed],
        "manchetes": news,
    }
