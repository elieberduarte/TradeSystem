"""Estudo estatístico do comportamento do ativo.

Antes de desenhar um setup, descobrir ONDE existe assimetria: em que
horários o mercado anda, quantos dias são de tendência, se rompimentos
têm continuação ou reversão. Cada função devolve um DataFrame pronto
para leitura (e para o painel).
"""

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
