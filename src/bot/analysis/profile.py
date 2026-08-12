"""Estudo estatístico do comportamento do ativo.

Antes de desenhar um setup, descobrir ONDE existe assimetria: em que
horários o mercado anda, quantos dias são de tendência, se rompimentos
têm continuação ou reversão. Cada função devolve um DataFrame pronto
para leitura (e para o painel).
"""

from datetime import time

import numpy as np
import pandas as pd

from src.bot.analysis.regime import Regime, adx


def by_hour(candles: pd.DataFrame) -> pd.DataFrame:
    """Movimento e direção por hora do pregão.

    `move` = amplitude média do candle (high-low) — quanto o mercado anda.
    `net` = retorno médio (close-open) — se há viés direcional na hora.
    `follow` = % de candles cujo fechamento vai na direção da abertura.
    """
    df = candles.copy()
    df["hour"] = df.index.hour
    df["range"] = df["high"] - df["low"]
    df["net"] = df["close"] - df["open"]
    grouped = df.groupby("hour")
    out = pd.DataFrame(
        {
            "candles": grouped.size(),
            "amplitude_media": grouped["range"].mean().round(1),
            "retorno_medio": grouped["net"].mean().round(1),
            "volatilidade": grouped["net"].std().round(1),
        }
    )
    return out[out["candles"] > 50]


def daily_regimes(candles: pd.DataFrame, threshold: float = 25.0) -> pd.DataFrame:
    """Classifica cada pregão pelo ADX do fim do dia.

    Responde: quantos dias do ano são realmente de tendência? É o dado
    que decide se vale a pena ter setup de rompimento no arsenal.
    """
    indicators = adx(candles, 14)
    df = candles.assign(adx=indicators["adx"], plus=indicators["plus_di"], minus=indicators["minus_di"])
    df["day"] = df.index.normalize()

    rows = []
    for day, group in df.groupby("day"):
        if len(group) < 20:
            continue
        last = group.iloc[-1]
        if pd.isna(last["adx"]) or last["adx"] < threshold:
            regime = Regime.RANGE
        elif last["plus"] >= last["minus"]:
            regime = Regime.TREND_UP
        else:
            regime = Regime.TREND_DOWN
        rows.append(
            {
                "day": day,
                "regime": regime.value,
                "adx": round(float(last["adx"]), 1),
                "amplitude": float(group["high"].max() - group["low"].min()),
                "net": float(group["close"].iloc[-1] - group["open"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def opening_range_study(
    candles: pd.DataFrame, range_bars: int = 3, horizon_bars: int = 12
) -> pd.DataFrame:
    """O rompimento da abertura tem continuação ou reverte?

    Para cada dia: forma o range dos primeiros candles, encontra o
    primeiro rompimento e mede o que aconteceu nos candles seguintes —
    quanto andou a favor (MFE) e quanto andou contra (MAE).
    """
    df = candles.copy()
    df["day"] = df.index.normalize()
    rows = []
    for day, group in df.groupby("day"):
        if len(group) < range_bars + horizon_bars + 1:
            continue
        opening = group.iloc[:range_bars]
        hi, lo = float(opening["high"].max()), float(opening["low"].min())
        rest = group.iloc[range_bars:]

        for i in range(1, len(rest)):
            close_now = float(rest["close"].iloc[i])
            close_prev = float(rest["close"].iloc[i - 1])
            side = None
            if close_prev <= hi and close_now > hi:
                side = "alta"
            elif close_prev >= lo and close_now < lo:
                side = "baixa"
            if side is None:
                continue

            future = rest.iloc[i + 1 : i + 1 + horizon_bars]
            if future.empty:
                break
            if side == "alta":
                mfe = float(future["high"].max()) - close_now
                mae = close_now - float(future["low"].min())
            else:
                mfe = close_now - float(future["low"].min())
                mae = float(future["high"].max()) - close_now
            rows.append(
                {
                    "day": day,
                    "lado": side,
                    "hora": rest.index[i].hour,
                    "range_pts": hi - lo,
                    "mfe": round(mfe, 1),   # máximo a favor
                    "mae": round(mae, 1),   # máximo contra
                }
            )
            break  # só o primeiro rompimento do dia
    return pd.DataFrame(rows)


def early_regimes(
    candles: pd.DataFrame, until: time = time(10, 0), threshold: float = 25.0
) -> pd.DataFrame:
    """Classifica o pregão usando SÓ os candles até `until`.

    Diferença crítica para `daily_regimes`: aquela usa o ADX do último
    candle do dia — informação que só existe depois que o pregão acabou.
    Condicionar uma estratégia àquele rótulo é look-ahead. Esta versão
    responde a pergunta operacional: dá para saber o regime cedo, com a
    informação que o bot realmente teria na hora de decidir?
    """
    # O ADX é calculado sobre a série contínua e depois lido no candle de
    # corte de cada dia — é o que o bot ao vivo enxerga, já que às 10h
    # existem poucos candles do dia mas o histórico anterior está lá.
    indicators = adx(candles, 14)
    days = candles.index.normalize()

    rows = []
    for day in days.unique():
        mask = (days == day) & (candles.index.time <= until)
        if not mask.any():
            continue
        row = indicators[mask].iloc[-1]
        if pd.isna(row["adx"]) or row["adx"] < threshold:
            regime = Regime.RANGE
        elif row["plus_di"] >= row["minus_di"]:
            regime = Regime.TREND_UP
        else:
            regime = Regime.TREND_DOWN
        rows.append(
            {"day": day, "regime_cedo": regime.value, "adx_cedo": round(float(row["adx"]), 1)}
        )
    return pd.DataFrame(rows)


def regime_persistence(
    candles: pd.DataFrame, until: time = time(10, 0), threshold: float = 25.0
) -> pd.DataFrame:
    """O regime detectado de manhã se confirma no fim do dia?

    Matriz de transição entre `early_regimes` e `daily_regimes`. Se o
    regime da manhã não prevê o do fim do dia, filtrar por ele não
    ajuda — seria adivinhar, não classificar.
    """
    early = early_regimes(candles, until, threshold).set_index("day")["regime_cedo"]
    late = daily_regimes(candles, threshold).set_index("day")["regime"]
    joined = pd.concat([early, late], axis=1).dropna()
    if joined.empty:
        return pd.DataFrame()
    return pd.crosstab(joined["regime_cedo"], joined["regime"], normalize="index").round(3)


def breakout_by_early_regime(
    candles: pd.DataFrame,
    range_bars: int = 3,
    horizon_bars: int = 12,
    until: time = time(10, 0),
) -> pd.DataFrame:
    """MFE/MAE dos rompimentos por regime CONHECIDO NA HORA (sem look-ahead)."""
    breakouts = opening_range_study(candles, range_bars, horizon_bars)
    if breakouts.empty:
        return pd.DataFrame()
    early = early_regimes(candles, until).set_index("day")["regime_cedo"]
    breakouts = breakouts.assign(regime=breakouts["day"].map(early)).dropna(subset=["regime"])
    if breakouts.empty:
        return pd.DataFrame()

    grouped = breakouts.groupby("regime")
    out = pd.DataFrame(
        {
            "rompimentos": grouped.size(),
            "mfe_medio": grouped["mfe"].mean().round(1),
            "mae_medio": grouped["mae"].mean().round(1),
        }
    )
    out["razao"] = (out["mfe_medio"] / out["mae_medio"]).round(3)
    return out


def swing_regime_persistence(
    candles: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 10, 20), threshold: float = 25.0
) -> pd.DataFrame:
    """O regime persiste no horizonte de swing?

    Para candles diários: classifica cada barra pelo ADX e mede se o
    regime de hoje continua o mesmo daqui a N barras. É o teste direto
    da premissa "identificar o ciclo antes de escolher o setup" — que
    falhou no intradiário porque o regime da manhã não previa o do dia.
    """
    indicators = adx(candles, 14)
    regime = pd.Series(Regime.RANGE.value, index=candles.index, dtype=object)
    trending = indicators["adx"] >= threshold
    regime[trending & (indicators["plus_di"] >= indicators["minus_di"])] = Regime.TREND_UP.value
    regime[trending & (indicators["plus_di"] < indicators["minus_di"])] = Regime.TREND_DOWN.value
    regime = regime[indicators["adx"].notna()]

    rows = []
    for horizon in horizons:
        future = regime.shift(-horizon)
        pairs = pd.concat([regime, future], axis=1).dropna()
        pairs.columns = ["agora", "depois"]
        if pairs.empty:
            continue
        same = (pairs["agora"] == pairs["depois"]).mean()
        # Base de comparação: acertar por acaso, dada a frequência de cada regime
        chance = (regime.value_counts(normalize=True) ** 2).sum()
        rows.append(
            {
                "horizonte_barras": horizon,
                "mesmo_regime": round(float(same), 3),
                "acaso": round(float(chance), 3),
                "ganho_sobre_acaso": round(float(same - chance), 3),
                "amostras": len(pairs),
            }
        )
    return pd.DataFrame(rows)


def autocorrelation_by_regime(
    candles: pd.DataFrame, lags: tuple[int, ...] = (1, 3, 6, 12)
) -> pd.DataFrame:
    """Autocorrelação dos retornos, separada por regime do pregão.

    A pergunta central: o WIN se comporta de forma diferente em dias de
    tendência e em dias laterais? Autocorrelação positiva = continuação
    (favorece seguir movimento); negativa = reversão (favorece contra);
    zero = passeio aleatório, sem estrutura explorável.
    """
    regimes = daily_regimes(candles).set_index("day")["regime"]
    df = candles.copy()
    df["day"] = df.index.normalize()
    df["regime"] = df["day"].map(regimes)
    df["ret"] = df["close"].diff()

    rows = []
    for regime, group in df.dropna(subset=["regime"]).groupby("regime"):
        row = {"regime": regime, "candles": len(group)}
        for lag in lags:
            # Correlação entre o retorno e o retorno `lag` barras à frente,
            # calculada dentro de cada dia (não atravessa a virada)
            pairs = group.groupby("day")["ret"].apply(
                lambda s: s.corr(s.shift(-lag)) if len(s) > lag + 5 else float("nan")
            )
            row[f"lag_{lag}"] = round(float(pairs.mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def breakout_by_regime(
    candles: pd.DataFrame, range_bars: int = 3, horizon_bars: int = 12
) -> pd.DataFrame:
    """MFE/MAE do rompimento da abertura, separado por regime do dia.

    Se a razão for ~1,0 em todos os regimes, o rompimento não tem
    assimetria escondida pela agregação — a ausência de edge é real,
    não um artefato de misturar mercados diferentes.
    """
    breakouts = opening_range_study(candles, range_bars, horizon_bars)
    if breakouts.empty:
        return pd.DataFrame()
    regimes = daily_regimes(candles).set_index("day")["regime"]
    breakouts = breakouts.assign(regime=breakouts["day"].map(regimes)).dropna(subset=["regime"])

    grouped = breakouts.groupby("regime")
    out = pd.DataFrame(
        {
            "rompimentos": grouped.size(),
            "mfe_medio": grouped["mfe"].mean().round(1),
            "mae_medio": grouped["mae"].mean().round(1),
        }
    )
    out["razao"] = (out["mfe_medio"] / out["mae_medio"]).round(3)
    return out


def gap_study(candles: pd.DataFrame) -> pd.DataFrame:
    """Gap de abertura vs. o que o dia fez depois.

    Responde: gap tende a ser fechado (reversão) ou continuado?
    """
    df = candles.copy()
    df["day"] = df.index.normalize()
    daily = df.groupby("day").agg(
        open=("open", "first"), close=("close", "last"),
        high=("high", "max"), low=("low", "min"),
    )
    daily["prev_close"] = daily["close"].shift(1)
    daily = daily.dropna()
    daily["gap"] = daily["open"] - daily["prev_close"]
    daily["dia"] = daily["close"] - daily["open"]
    return daily[["gap", "dia", "high", "low", "open", "close"]]


def summarize(candles: pd.DataFrame) -> dict:
    """Resumo numérico das assimetrias encontradas."""
    hours = by_hour(candles)
    regimes = daily_regimes(candles)
    breakouts = opening_range_study(candles)
    gaps = gap_study(candles)

    regime_counts = regimes["regime"].value_counts(normalize=True).round(3).to_dict()
    # Correlação negativa = gap tende a ser fechado no dia (reversão)
    gap_corr = float(np.corrcoef(gaps["gap"], gaps["dia"])[0, 1]) if len(gaps) > 2 else 0.0

    payload = {
        "pregoes": int(len(regimes)),
        "regimes_pct": regime_counts,
        "amplitude_media_dia": round(float(regimes["amplitude"].mean()), 1),
        "hora_mais_volatil": int(hours["volatilidade"].idxmax()),
        "hora_menos_volatil": int(hours["volatilidade"].idxmin()),
        "gap_corr_dia": round(gap_corr, 3),
    }
    if not breakouts.empty:
        payload.update(
            {
                "rompimentos": int(len(breakouts)),
                "mfe_medio": round(float(breakouts["mfe"].mean()), 1),
                "mae_medio": round(float(breakouts["mae"].mean()), 1),
                # > 1 = o movimento a favor supera o contra
                "razao_mfe_mae": round(
                    float(breakouts["mfe"].mean() / breakouts["mae"].mean()), 2
                ),
            }
        )
    return payload
