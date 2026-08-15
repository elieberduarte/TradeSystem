"""Estudo de níveis objetivos: eles existem ou são pareidolia?

A literatura de price action inteira se apoia em uma premissa nunca
demonstrada: a de que certos preços (máxima de ontem, pivô, R1...)
são especiais — o mercado os "respeita". Este módulo mede isso com
um experimento controlado:

  1. Detecta cada TOQUE em um nível objetivo e verifica se o preço
     repica (afasta-se uma distância R antes de atravessar R além) ou
     rompe.
  2. Repete a medição em níveis PLACEBO: o mesmo preço deslocado por
     uma fração do range de ontem. O placebo herda a vizinhança, o
     horário e a volatilidade — só perde a "objetividade" do nível.

Se a taxa de repique no nível real não superar a do placebo, o nível
não existe, e toda estratégia ancorada nele (filtro de geometria 3:1
do Eykyn incluído) morre pela premissa.

De quebra, o registro do índice do toque (1º, 2º, 3º...) testa a
afirmação do Madang de que "níveis rompem após 2-3 testes".

Regras fixadas ANTES de rodar, sem otimização:
  banda de toque  = 5% do range de ontem
  distância R     = 25% do range de ontem (corrida repique × rompimento)
  placebo         = nível real ± {13%, 21%, 34%} do range de ontem,
                    excluído se cair a menos de 2 bandas de um nível
                    real (senão mediria o próprio nível)
  ambiguidade     = se o mesmo candle atinge os dois lados da corrida,
                    decide o fechamento (mesma regra para real e placebo,
                    então qualquer viés se cancela na comparação)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Touch:
    date: pd.Timestamp
    level_name: str
    level_price: float
    side: str          # "below" = testado como resistência; "above" = suporte
    index: int         # k-ésimo toque deste nível/lado no dia (1-based)
    broke: bool
    placebo: bool


def session_frame(intraday: pd.DataFrame) -> pd.DataFrame:
    """Agrega candles intraday em sessões diárias (OHLC + contagem)."""
    groups = intraday.groupby(intraday.index.normalize())
    return pd.DataFrame(
        {
            "open": groups["open"].first(),
            "high": groups["high"].max(),
            "low": groups["low"].min(),
            "close": groups["close"].last(),
            "bars": groups["close"].size(),
        }
    )


def levels_for(
    prev: pd.Series, prev2: pd.Series, round_step: float = 0.0
) -> list[tuple[str, float]]:
    """Níveis objetivos da sessão, todos conhecidos antes da abertura."""
    high, low, close = float(prev["high"]), float(prev["low"]), float(prev["close"])
    pivot = (high + low + close) / 3
    levels = [
        ("YH", high), ("YL", low), ("YC", close),
        ("DBYH", float(prev2["high"])), ("DBYL", float(prev2["low"])),
        ("PP", pivot),
        ("R1", 2 * pivot - low), ("S1", 2 * pivot - high),
        ("R2", pivot + (high - low)), ("S2", pivot - (high - low)),
    ]
    if round_step > 0:
        span = high - low
        mark = np.ceil((low - span) / round_step) * round_step
        while mark <= high + span:
            levels.append(("RND", float(mark)))
            mark += round_step
    return levels


def _race(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    start: int, level: float, race: float, side: str,
) -> tuple[bool | None, int]:
    """A partir do toque, o que vem primeiro: repique ou rompimento?

    Retorna (rompeu, barra_da_resolução); rompeu=None se o dia acaba
    sem resolução.

    A barra do toque exige tratamento especial: ela veio de R de
    distância, então o extremo do lado da aproximação é o caminho de
    ida, não um repique. Nela só se avalia o rompimento — e, se o
    preço atravessa R além mas fecha de volta do lado de origem, o
    fechamento decide (pavio de rejeição conta como repique).
    """
    if side == "below":
        if highs[start] >= level + race:
            return closes[start] > level, start
    else:
        if lows[start] <= level - race:
            return closes[start] < level, start

    for i in range(start + 1, len(highs)):
        if side == "below":                      # nível testado como resistência
            hit_break = highs[i] >= level + race
            hit_bounce = lows[i] <= level - race
        else:                                    # nível testado como suporte
            hit_break = lows[i] <= level - race
            hit_bounce = highs[i] >= level + race
        if hit_break and hit_bounce:             # ambíguo: decide o fechamento
            broke = closes[i] > level if side == "below" else closes[i] < level
            return broke, i
        if hit_break:
            return True, i
        if hit_bounce:
            return False, i
    return None, len(highs) - 1


def walk_level(
    bars: pd.DataFrame, level: float, band: float, race: float
) -> list[tuple[str, bool]]:
    """Sequência de toques (lado, rompeu) de um nível em uma sessão.

    Um toque só conta com o preço "armado": vindo de pelo menos R de
    distância do nível. Depois de um repique o preço já está a R de
    distância por construção, então o rearme é automático. O
    acompanhamento termina no primeiro rompimento — a pergunta do
    estudo é quantos testes o nível aguenta até romper.
    """
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    n = len(highs)

    armed: str | None = None
    start = n
    for i in range(n):                            # posição inicial do preço
        if highs[i] <= level - race:
            armed, start = "below", i + 1
            break
        if lows[i] >= level + race:
            armed, start = "above", i + 1
            break

    events: list[tuple[str, bool]] = []
    i = start
    while i < n and armed is not None:
        touched = (
            highs[i] >= level - band if armed == "below" else lows[i] <= level + band
        )
        if not touched:
            i += 1
            continue
        broke, resolved_at = _race(highs, lows, closes, i, level, race, armed)
        if broke is None:                         # dia acabou no meio da corrida
            break
        events.append((armed, broke))
        if broke:                                 # nível consumido
            break
        i = resolved_at + 1                       # repicou: já rearmado a R do nível
    return events


def walk_zone(
    bars: pd.DataFrame,
    level: float,
    band: float,
    race: float,
    start: int,
    side: str,
) -> list[tuple[int, bool]]:
    """Toques em uma zona nascida DURANTE a sessão (nível dinâmico).

    Igual a `walk_level`, com três diferenças: começa em `start` (a
    barra em que a zona ficou conhecível — antes disso ela não
    existia para ninguém), testa um único lado (`side`: "above" =
    suporte tocado por cima, "below" = resistência por baixo) e
    termina no primeiro rompimento.

    Retorna [(barra_do_toque, rompeu), ...].
    """
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    n = len(highs)

    events: list[tuple[int, bool]] = []
    armed = False
    i = start
    while i < n:
        if not armed:
            far_enough = (
                lows[i] >= level + race if side == "above" else highs[i] <= level - race
            )
            if far_enough:
                armed = True
            i += 1
            continue

        touched = (
            lows[i] <= level + band if side == "above" else highs[i] >= level - band
        )
        if not touched:
            i += 1
            continue

        broke, resolved_at = _race(highs, lows, closes, i, level, race, side)
        if broke is None:                     # a sessão acabou no meio da corrida
            break
        events.append((i, broke))
        if broke:
            break
        i = resolved_at + 1                   # repicou: já está a R da zona (rearmado)
    return events


def study_levels(
    intraday: pd.DataFrame,
    round_step: float = 0.0,
    band_frac: float = 0.05,
    race_frac: float = 0.25,
    shift_fracs: tuple[float, ...] = (0.13, 0.21, 0.34),
    min_bars: int = 30,
) -> pd.DataFrame:
    """Roda o estudo completo e devolve um DataFrame de toques."""
    daily = session_frame(intraday)
    by_day = dict(tuple(intraday.groupby(intraday.index.normalize())))

    touches: list[Touch] = []
    for t in range(2, len(daily)):
        day = daily.index[t]
        prev, prev2 = daily.iloc[t - 1], daily.iloc[t - 2]
        prev_range = float(prev["high"] - prev["low"])
        if prev_range <= 0 or daily["bars"].iloc[t] < min_bars:
            continue

        band = band_frac * prev_range
        race = race_frac * prev_range

        reals = levels_for(prev, prev2, round_step)
        real_prices = [price for _, price in reals]

        candidates: list[tuple[str, float, bool]] = [
            (name, price, False) for name, price in reals
        ]
        placed: list[float] = []
        for _, price in reals:
            for frac in shift_fracs:
                for placebo in (price - frac * prev_range, price + frac * prev_range):
                    # placebo em cima de um nível real mediria o próprio
                    # nível; em cima de outro placebo contaria em dobro
                    if min(abs(placebo - rp) for rp in real_prices) < 2 * band:
                        continue
                    if placed and min(abs(placebo - pp) for pp in placed) < 2 * band:
                        continue
                    placed.append(placebo)
                    candidates.append(("PBO", placebo, True))

        bars = by_day[day]
        for name, price, is_placebo in candidates:
            events = walk_level(bars, price, band, race)
            for k, (side, broke) in enumerate(events, start=1):
                touches.append(
                    Touch(day, name, price, side, k, broke, is_placebo)
                )

    return pd.DataFrame([t.__dict__ for t in touches])


def two_proportion_z(hits_a: int, n_a: int, hits_b: int, n_b: int) -> tuple[float, float]:
    """z e p (bicaudal) para a diferença entre duas proporções."""
    from math import erf, sqrt

    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    p_a, p_b = hits_a / n_a, hits_b / n_b
    pooled = (hits_a + hits_b) / (n_a + n_b)
    se = sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_a - p_b) / se
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return z, p_value
