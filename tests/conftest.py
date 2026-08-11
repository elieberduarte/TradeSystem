"""Geradores de candles sintéticos para os testes."""

import numpy as np
import pandas as pd
import pytest


def make_candles(closes: list[float], start="2026-08-10 10:00", freq="5min") -> pd.DataFrame:
    """Constrói OHLCV plausível a partir de uma série de fechamentos."""
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + 1.0
    lows = np.minimum(opens, closes) - 1.0
    index = pd.date_range(start=start, periods=len(closes), freq=freq)
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(len(closes), 100.0),
        },
        index=index,
    )


@pytest.fixture
def trending_up_candles() -> pd.DataFrame:
    """Queda seguida de forte alta — força um cruzamento de EMA para cima."""
    down = [1000 - i * 2 for i in range(100)]
    up = [down[-1] + i * 5 for i in range(1, 61)]
    return make_candles(down + up)
