"""Busca de padrões nos dados, com a honestidade embutida.

A ideia: dividir cada variável de estado em faixas e medir o retorno
futuro médio em cada uma. Onde o retorno destoa, há um padrão
candidato. Isso é exploração legítima — e barata: 100 mil candles com
dezenas de variáveis roda em segundos.

O perigo não é computacional, é estatístico. Testando muitas
combinações, algumas parecem significativas por acaso. A defesa
implementada aqui é o TESTE DE PERMUTAÇÃO: embaralha-se o alvo,
destruindo qualquer relação real, e repete-se a busca inteira. O
melhor padrão encontrado no dado embaralhado é a régua — qualquer
achado que não supere essa régua é ruído com cara de descoberta.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Pattern:
    feature: str
    bin_label: str
    samples: int
    mean_return: float
    t_stat: float
    hit_rate: float

    def __str__(self) -> str:
        return (
            f"{self.feature:<12} {self.bin_label:<22} n={self.samples:>6} "
            f"média={self.mean_return:+.4f} t={self.t_stat:+.2f} acerto={self.hit_rate:.1%}"
        )


def scan_feature(
    values: pd.Series, target: pd.Series, bins: int = 5, min_samples: int = 100
) -> list[Pattern]:
    """Divide uma variável em faixas e mede o alvo em cada uma."""
    frame = pd.concat([values.rename("x"), target.rename("y")], axis=1).dropna()
    frame = frame[np.isfinite(frame["x"]) & np.isfinite(frame["y"])]
    if len(frame) < min_samples * 2:
        return []

    unique = frame["x"].nunique()
    if unique <= bins * 2:
        frame["faixa"] = frame["x"]
    else:
        try:
            frame["faixa"] = pd.qcut(frame["x"], bins, duplicates="drop")
        except ValueError:
            return []

    patterns = []
    for label, group in frame.groupby("faixa", observed=True):
        if len(group) < min_samples:
            continue
        values_y = group["y"].to_numpy()
        mean = float(values_y.mean())
        std = float(values_y.std(ddof=1))
        if std == 0:
            continue
        t_stat = mean / (std / np.sqrt(len(values_y)))
        patterns.append(
            Pattern(
                feature=str(values.name),
                bin_label=str(label),
                samples=len(group),
                mean_return=mean,
                t_stat=float(t_stat),
                hit_rate=float((values_y > 0).mean()),
            )
        )
    return patterns


def scan_all(
    features: pd.DataFrame, target: pd.Series, bins: int = 5, min_samples: int = 100
) -> list[Pattern]:
    """Varre todas as variáveis e devolve os padrões ordenados por força."""
    found = []
    for column in features.columns:
        found.extend(scan_feature(features[column], target, bins, min_samples))
    return sorted(found, key=lambda p: abs(p.t_stat), reverse=True)


def permutation_baseline(
    features: pd.DataFrame,
    target: pd.Series,
    rounds: int = 20,
    bins: int = 5,
    min_samples: int = 100,
    seed: int = 42,
) -> list[float]:
    """Qual o melhor t observado quando NÃO existe padrão algum?

    Embaralha o alvo (destruindo qualquer relação) e roda a mesma
    busca. O maior |t| de cada rodada forma a distribuição do acaso.
    Um achado só vale se superar essa régua.
    """
    rng = np.random.default_rng(seed)
    values = target.to_numpy()
    best = []
    for _ in range(rounds):
        shuffled = pd.Series(rng.permutation(values), index=target.index, name=target.name)
        patterns = scan_all(features, shuffled, bins, min_samples)
        best.append(abs(patterns[0].t_stat) if patterns else 0.0)
    return best


def heatmap(
    feature_a: pd.Series, feature_b: pd.Series, target: pd.Series, bins: int = 4
) -> pd.DataFrame:
    """Retorno médio cruzando duas variáveis — o mapa de calor.

    Cruzar duas condições revela padrões que nenhuma delas mostra
    sozinha, mas também multiplica o número de células testadas e,
    com ele, a chance de ruído parecer sinal.
    """
    frame = pd.concat(
        [feature_a.rename("a"), feature_b.rename("b"), target.rename("y")], axis=1
    ).dropna()
    if frame.empty:
        return pd.DataFrame()

    def cut(series: pd.Series) -> pd.Series:
        if series.nunique() <= bins * 2:
            return series
        return pd.qcut(series, bins, duplicates="drop")

    frame["fa"] = cut(frame["a"])
    frame["fb"] = cut(frame["b"])
    return frame.pivot_table(index="fa", columns="fb", values="y", aggfunc="mean", observed=True)
