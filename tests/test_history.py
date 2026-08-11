"""Testes do armazenamento local de histórico."""

from src.bot.data.history import HistoryStore
from tests.conftest import make_candles


def test_roundtrip_salvar_e_carregar(tmp_path):
    store = HistoryStore(tmp_path)
    candles = make_candles([1000.0, 1001.0, 1002.0])
    store.save("WINQ26", "5m", candles)

    loaded = store.load("WINQ26", "5m")
    assert loaded is not None
    assert len(loaded) == 3
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]


def test_load_sem_dados_retorna_none(tmp_path):
    store = HistoryStore(tmp_path)
    assert store.load("WDOU26", "5m") is None


def test_merge_deduplica_e_o_dado_novo_prevalece(tmp_path):
    store = HistoryStore(tmp_path)
    old = make_candles([1000.0, 1001.0, 1002.0], start="2026-08-10 10:00")
    store.save("WINQ26", "5m", old)

    # Sobreposição: mesmo horário dos 2 últimos candles + 2 novos
    new = make_candles([9999.0, 1003.0, 1004.0, 1005.0], start="2026-08-10 10:05")
    merged = store.save("WINQ26", "5m", new)

    assert len(merged) == 5
    # No timestamp duplicado (10:05), vale o valor do download novo
    assert merged.loc["2026-08-10 10:05", "close"] == 9999.0
    assert merged.index.is_monotonic_increasing


def test_arquivos_separados_por_simbolo_e_timeframe(tmp_path):
    store = HistoryStore(tmp_path)
    store.save("WINQ26", "5m", make_candles([1.0]))
    store.save("WINQ26", "15m", make_candles([2.0]))
    store.save("wdou26", "5m", make_candles([3.0]))

    assert store.path("WINQ26", "5m").exists()
    assert store.path("WINQ26", "15m").exists()
    # Símbolo é normalizado para maiúsculas
    assert store.path("WDOU26", "5m").exists()
