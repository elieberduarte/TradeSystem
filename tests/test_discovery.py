"""Testes da busca de padrões e da régua do acaso."""

import numpy as np
import pandas as pd

from src.bot.analysis.discovery import heatmap, permutation_baseline, scan_all, scan_feature


def synthetic(n: int = 2000, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Cria dados com UM padrão plantado: quando x1 > 0,8, o alvo sobe."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    x1 = pd.Series(rng.uniform(0, 1, n), index=index, name="x1")
    x2 = pd.Series(rng.uniform(0, 1, n), index=index, name="x2")
    noise = rng.normal(0, 1, n)
    target = pd.Series(noise + np.where(x1 > 0.8, 2.0, 0.0), index=index, name="alvo")
    return pd.DataFrame({"x1": x1, "x2": x2}), target


def test_encontra_o_padrao_plantado():
    features, target = synthetic()
    patterns = scan_all(features, target, bins=5)

    assert patterns
    # O padrão mais forte deve ser a faixa alta de x1
    assert patterns[0].feature == "x1"
    assert patterns[0].mean_return > 1.0


def test_variavel_sem_relacao_nao_domina():
    features, target = synthetic()
    patterns = scan_all(features, target, bins=5)
    x2_forte = max((p for p in patterns if p.feature == "x2"), key=lambda p: abs(p.t_stat))
    x1_forte = max((p for p in patterns if p.feature == "x1"), key=lambda p: abs(p.t_stat))
    assert abs(x1_forte.t_stat) > abs(x2_forte.t_stat)


def test_regua_do_acaso_e_menor_que_o_padrao_real():
    features, target = synthetic()
    patterns = scan_all(features, target, bins=5)
    baseline = permutation_baseline(features, target, rounds=5, bins=5)

    assert len(baseline) == 5
    # O padrão plantado supera com folga o melhor achado no ruído
    assert abs(patterns[0].t_stat) > max(baseline)


def test_regua_captura_ruido_em_dados_sem_padrao():
    rng = np.random.default_rng(3)
    index = pd.date_range("2024-01-01", periods=1500, freq="h")
    features = pd.DataFrame(
        {"a": rng.normal(size=1500), "b": rng.normal(size=1500)}, index=index
    )
    target = pd.Series(rng.normal(size=1500), index=index, name="alvo")

    patterns = scan_all(features, target, bins=5)
    baseline = permutation_baseline(features, target, rounds=10, bins=5)
    # Sem padrão real, o melhor achado fica na mesma ordem do ruído
    assert abs(patterns[0].t_stat) < max(baseline) * 2.5


def test_scan_ignora_amostra_pequena():
    index = pd.date_range("2024-01-01", periods=50, freq="h")
    values = pd.Series(range(50), index=index, name="x")
    target = pd.Series(range(50), index=index, dtype=float)
    assert scan_feature(values, target, min_samples=100) == []


def test_heatmap_cruza_duas_condicoes():
    features, target = synthetic()
    grid = heatmap(features["x1"], features["x2"], target, bins=3)

    assert not grid.empty
    assert grid.shape == (3, 3)
    # A linha de x1 mais alto tem média maior que a mais baixa
    assert grid.iloc[-1].mean() > grid.iloc[0].mean()
