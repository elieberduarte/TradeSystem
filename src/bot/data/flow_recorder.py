"""Gravador de fluxo: book de ofertas e ticks do MT5, com desequilíbrio.

O MT5 não guarda histórico de book, e a XP demo não entrega ticks
históricos em tempo real (só o último negócio, ao vivo) — e sem o
lado agressor. Este módulo constrói o acervo que
não existe: a cada intervalo, fotografa o book (N níveis) e drena os
ticks novos, calculando duas medidas de desequilíbrio no ato:

  book_imbalance  (Σ qtde compra − Σ qtde venda) / (Σ total) nos K
                  melhores níveis — quem tem mais peso empilhado.
  delta           volume agredido na compra − na venda, classificado
                  pela regra de Lee-Ready (1991): negócio no ask (ou
                  acima do meio) = agressão de compra; no bid (ou
                  abaixo do meio) = de venda; no meio, tick test.
                  ~85-90% de acurácia contra o lado verdadeiro.

Tudo é gravado em Parquet por dia (data/flow/{símbolo}/{data}_book.
parquet e _ticks.parquet) para os estudos OFI / CVD / book-imbalance
em horizonte operável (dia, não segundos).

Convenções do MT5: BOOK_TYPE_SELL = ofertas de venda (asks),
BOOK_TYPE_BUY = ofertas de compra (bids). copy_ticks devolve
bid/ask/last/volume_real e time_msc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None

STORE = Path("data/flow")


def book_imbalance(bids: np.ndarray, asks: np.ndarray, levels: int) -> float:
    """(compra − venda)/(total) nos K melhores níveis; +1 = só compra."""
    b = float(bids[:levels].sum()) if len(bids) else 0.0
    a = float(asks[:levels].sum()) if len(asks) else 0.0
    return (b - a) / (b + a) if (b + a) > 0 else 0.0


def classify_book(bid1: float, ask1: float, cross_limit: float = 0.005) -> str:
    """O estado do book naquele instante — três realidades, não uma.

    'continuo'  bid < ask: o pregão normal, onde imbalance faz sentido.
    'leilao'    book CRUZADO por muito (ordens nos limites do túnel de
                ±10%, quantidades agregadas gigantes). Em leilão as
                ordens se cruzam de propósito: é assim que o preço
                teórico de abertura é formado. Não é erro de leitura —
                é outro mercado, e a métrica de desequilíbrio contínuo
                não se aplica.
    'cruzado'   bid ligeiramente acima do ask (< `cross_limit` do preço):
                a leitura pegou o book no meio de uma atualização.
                Descartável.
    """
    if ask1 > bid1:
        return "continuo"
    mid = (bid1 + ask1) / 2.0
    if mid > 0 and (bid1 - ask1) / mid > cross_limit:
        return "leilao"
    return "cruzado"


def lee_ready(last: np.ndarray, bid: np.ndarray, ask: np.ndarray,
              prev_side: int = 0) -> np.ndarray:
    """Lado agressor por negócio: +1 compra, −1 venda, 0 indefinido.

    Quote rule primeiro (acima/abaixo do meio); tick test nos empates
    (comparação com o último preço diferente); herda o lado anterior
    quando nem isso resolve.
    """
    mid = (bid + ask) / 2.0
    side = np.zeros(len(last), dtype=np.int8)
    side[last > mid] = 1
    side[last < mid] = -1
    last_side = prev_side
    last_price = None
    for i in range(len(last)):
        if side[i] == 0:                          # no meio: tick test
            if last_price is not None and last[i] != last_price:
                side[i] = 1 if last[i] > last_price else -1
            else:
                side[i] = last_side
        if side[i] != 0:
            last_side = side[i]
        if last_price is None or last[i] != last_price:
            last_price = last[i]
    return side


@dataclass
class FlowRecorder:
    symbols: list[str]
    levels: int = 5                    # níveis usados no imbalance
    book_interval: float = 1.0         # segundos entre fotos do book
    flush_every: int = 300             # fotos acumuladas antes de gravar
    store: Path = STORE
    _book_rows: dict[str, list] = field(default_factory=dict, init=False)
    _tick_rows: dict[str, list] = field(default_factory=dict, init=False)
    _last_tick_msc: dict[str, int] = field(default_factory=dict, init=False)
    _last_side: dict[str, int] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------ ciclo

    def start(self) -> None:
        if mt5 is None:
            raise ImportError("MetaTrader5 não instalado")
        if not mt5.initialize():
            raise ConnectionError(f"MT5 indisponível: {mt5.last_error()}")
        for symbol in self.symbols:
            mt5.symbol_select(symbol, True)
            mt5.market_book_add(symbol)
            self._book_rows[symbol] = []
            self._tick_rows[symbol] = []
            # 0, não o relógio local: o servidor carimba em UTC (3h atrás de
            # Brasília) e um marcador local nunca deixaria um tick passar
            self._last_tick_msc[symbol] = 0
            self._last_side[symbol] = 0

    def stop(self) -> None:
        self.flush()
        for symbol in self.symbols:
            try:
                mt5.market_book_release(symbol)
            except Exception:  # noqa: BLE001
                pass
        mt5.shutdown()

    def snapshot(self, symbol: str, now_ms: int | None = None) -> dict | None:
        """Uma foto do book, já com o imbalance calculado."""
        book = mt5.market_book_get(symbol)
        if not book:
            return None
        now_ms = now_ms or int(time.time() * 1000)
        asks = sorted((b for b in book if b.type == mt5.BOOK_TYPE_SELL), key=lambda b: b.price)
        bids = sorted((b for b in book if b.type == mt5.BOOK_TYPE_BUY), key=lambda b: -b.price)
        if not asks or not bids:
            return None
        bid_qty = np.array([b.volume for b in bids], dtype=float)
        ask_qty = np.array([a.volume for a in asks], dtype=float)
        estado = classify_book(bids[0].price, asks[0].price)
        # Imbalance só tem sentido no mercado contínuo: em leilão o book
        # é cruzado por construção e as quantidades são agregados do
        # leilão, não profundidade comparável.
        contínuo = estado == "continuo"
        row = {
            "ts_ms": now_ms,
            "bid1": bids[0].price, "ask1": asks[0].price,
            "bid1_qty": bids[0].volume, "ask1_qty": asks[0].volume,
            "spread": asks[0].price - bids[0].price,
            "estado": estado,
            "imb_l1": book_imbalance(bid_qty, ask_qty, 1) if contínuo else float("nan"),
            "imb_lk": book_imbalance(bid_qty, ask_qty, self.levels) if contínuo else float("nan"),
            "bid_depth_lk": float(bid_qty[:self.levels].sum()),
            "ask_depth_lk": float(ask_qty[:self.levels].sum()),
            "levels": min(len(bids), len(asks)),
        }
        self._book_rows[symbol].append(row)
        return row

    def drain_ticks(self, symbol: str) -> int:
        """Captura o último negócio via symbol_info_tick e classifica o lado.

        A XP demo não entrega histórico de ticks em tempo real
        (copy_ticks_* só devolve o cache já sincronizado — na prática,
        nada intradiário). symbol_info_tick, porém, é ao vivo. Então o
        gravador AMOSTRA: a cada iteração lê o último negócio; se o
        time_msc mudou, é um tick novo. Resolução ≈ o intervalo do laço
        (1 s) — perde granularidade sub-segundo, ganha um acervo que
        de fato existe. Registra também o volume acumulado do último
        tick (volume_real do último negócio).
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or tick.time_msc <= self._last_tick_msc[symbol]:
            return 0
        if tick.last <= 0:
            return 0
        side = lee_ready(np.array([tick.last]), np.array([tick.bid]), np.array([tick.ask]),
                         self._last_side[symbol])
        s = int(side[0])
        if s != 0:
            self._last_side[symbol] = s
        # O estado importa tanto quanto no book: em LEILÃO o `last` é o
        # preço teórico e `volume_real` é o ACUMULADO do leilão, repetido
        # a cada leitura. Contá-lo como negócio inflou o CVD do WIN em
        # 1,3 milhão de contratos no dia 18/08 — o volume do dia inteiro.
        estado = classify_book(tick.bid, tick.ask)
        self._tick_rows[symbol].append({
            # Duas bases de tempo, explícitas: o servidor carimba o
            # horário de Brasília COMO SE fosse UTC, o que deixava a
            # série de ticks 3h fora da do book. `ts_ms` é sempre o
            # epoch UTC real do momento da captura — o mesmo relógio do
            # book, para que as duas séries sejam cruzáveis.
            "ts_ms": int(time.time() * 1000),
            "ts_server_ms": int(tick.time_msc),
            "last": float(tick.last),
            "bid": float(tick.bid), "ask": float(tick.ask),
            "volume": float(tick.volume_real or tick.volume or 0.0),
            "side": s, "estado": estado,
        })
        self._last_tick_msc[symbol] = int(tick.time_msc)
        return 1

    def flush(self) -> None:
        for symbol in self.symbols:
            for kind, rows in (("book", self._book_rows), ("ticks", self._tick_rows)):
                if not rows.get(symbol):
                    continue
                self._append(symbol, kind, pd.DataFrame(rows[symbol]))
                rows[symbol] = []

    def _append(self, symbol: str, kind: str, frame: pd.DataFrame) -> None:
        day = datetime.now().strftime("%Y-%m-%d")
        path = self.store / symbol / f"{day}_{kind}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
        frame.to_parquet(path)

    def run(self, until: datetime | None = None) -> None:
        """Laço principal: foto do book + drenagem de ticks a cada intervalo."""
        self.start()
        try:
            n = 0
            while until is None or datetime.now() < until:
                for symbol in self.symbols:
                    self.snapshot(symbol)
                    self.drain_ticks(symbol)
                n += 1
                if n % self.flush_every == 0:
                    self.flush()
                time.sleep(self.book_interval)
        finally:
            self.stop()


def cumulative_delta(ticks: pd.DataFrame) -> pd.Series:
    """CVD: soma acumulada de (volume × lado), só no mercado contínuo.

    Ticks de leilão carregam o volume ACUMULADO do leilão, não o de um
    negócio — somá-los conta o mesmo volume dezenas de vezes.
    """
    frame = ticks
    if "estado" in frame.columns:
        frame = frame[frame["estado"] == "continuo"]
    return (frame["volume"] * frame["side"]).cumsum()
