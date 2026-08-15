"""Coletor de dados abertos da CVM — a fonte oficial e gratuita.

Três conjuntos alimentam três frentes do projeto:

  FCA  cadastro anual: traz o CÓDIGO DE NEGOCIAÇÃO por CNPJ — o mapa
       entre o mundo da CVM (razão social) e o do pregão (PETR4).
  IPE  todos os fatos relevantes, comunicados e avisos, com data de
       entrega, categoria e link — o combustível do estudo de eventos.
  DFP/ITR demonstrações financeiras estruturadas — o combustível do
       painel de fundamentos (P/L, P/VP, ROE, dívida...).

Disciplina point-in-time embutida por design: toda tabela carrega a
DATA DE ENTREGA na CVM. Um balanço de dezembro só existe para o
mercado em março — usar a data de referência em backtest é
look-ahead, o mesmo pecado do zigzag que já evitamos duas vezes.

Os zips ficam em cache local (data/cvm/); nada é baixado duas vezes.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC"
CACHE = Path("data/cvm")


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    request = urllib.request.Request(url, headers={"User-Agent": "trade-bot/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        dest.write_bytes(response.read())
    return dest


def _read_zip_csv(zip_path: Path, name_contains: str) -> pd.DataFrame:
    """Lê o CSV cujo nome contém o padrão, no encoding da CVM (latin-1)."""
    with zipfile.ZipFile(zip_path) as archive:
        matches = [n for n in archive.namelist() if name_contains in n]
        if not matches:
            raise FileNotFoundError(f"{name_contains} não está em {zip_path.name}")
        with archive.open(matches[0]) as handle:
            return pd.read_csv(
                io.TextIOWrapper(handle, encoding="latin-1"),
                sep=";", low_memory=False,
            )


def _find_column(frame: pd.DataFrame, *candidates: str) -> str:
    """Acha a coluna cujo nome contém um dos padrões (a CVM já renomeou)."""
    lowered = {c.lower(): c for c in frame.columns}
    for candidate in candidates:
        for low, original in lowered.items():
            if candidate.lower() in low:
                return original
    raise KeyError(f"nenhuma coluna com {candidates} em {list(frame.columns)[:8]}...")


def ticker_map(years: range, cache: Path = CACHE) -> pd.DataFrame:
    """CNPJ → código de negociação, a partir do FCA (valor mobiliário)."""
    frames = []
    for year in years:
        try:
            path = _download(
                f"{BASE}/FCA/DADOS/fca_cia_aberta_{year}.zip",
                cache / f"fca_{year}.zip",
            )
            frame = _read_zip_csv(path, "valor_mobiliario")
        except Exception:  # noqa: BLE001 — ano sem arquivo não derruba o resto
            continue
        cnpj = _find_column(frame, "cnpj")
        code = _find_column(frame, "codigo_negociacao")
        name = _find_column(frame, "nome_empresarial", "denominacao")
        part = frame[[cnpj, code, name]].dropna(subset=[code]).copy()
        part.columns = ["cnpj", "ticker", "empresa"]
        part["ano"] = year
        frames.append(part)
    merged = pd.concat(frames, ignore_index=True)
    merged["ticker"] = merged["ticker"].str.strip().str.upper()
    # O cadastro mais recente prevalece para cada ticker
    return (
        merged.sort_values("ano")
        .drop_duplicates(subset=["ticker"], keep="last")
        .reset_index(drop=True)
    )


def load_events(years: range, cache: Path = CACHE) -> pd.DataFrame:
    """Fatos relevantes, comunicados e avisos (IPE), com data de entrega."""
    frames = []
    for year in years:
        try:
            path = _download(
                f"{BASE}/IPE/DADOS/ipe_cia_aberta_{year}.zip",
                cache / f"ipe_{year}.zip",
            )
            frame = _read_zip_csv(path, "ipe_cia_aberta")
        except Exception:  # noqa: BLE001
            continue
        cnpj = _find_column(frame, "cnpj")
        category = _find_column(frame, "categoria")
        subject = _find_column(frame, "assunto")
        delivered = _find_column(frame, "data_entrega")
        reference = _find_column(frame, "data_referencia")
        link = _find_column(frame, "link_download", "link")
        part = frame[[cnpj, category, subject, reference, delivered, link]].copy()
        part.columns = ["cnpj", "categoria", "assunto", "data_referencia",
                        "data_entrega", "link"]
        frames.append(part)
    events = pd.concat(frames, ignore_index=True)
    events["data_entrega"] = pd.to_datetime(events["data_entrega"], errors="coerce")
    return events.dropna(subset=["data_entrega"]).reset_index(drop=True)


def load_dfp(year: int, table: str = "DRE_con", cache: Path = CACHE) -> pd.DataFrame:
    """Uma demonstração financeira padronizada do ano (consolidada).

    Tabelas úteis: DRE_con (resultado), BPP_con (passivo + patrimônio),
    BPA_con (ativo), DFC_MD_con / DFC_MI_con (fluxo de caixa, onde
    vivem os dividendos pagos). Valores em ESCALA_MOEDA (MIL = ×1000).
    """
    path = _download(
        f"{BASE}/DFP/DADOS/dfp_cia_aberta_{year}.zip", cache / f"dfp_{year}.zip"
    )
    return _read_zip_csv(path, f"{table}_{year}")


def events_for_tickers(
    tickers: list[str], years: range, cache: Path = CACHE
) -> pd.DataFrame:
    """Eventos IPE já traduzidos para os tickers pedidos."""
    mapping = ticker_map(years, cache)
    wanted = mapping[mapping["ticker"].isin([t.upper() for t in tickers])]
    events = load_events(years, cache)
    merged = events.merge(wanted[["cnpj", "ticker", "empresa"]], on="cnpj")
    return merged.sort_values("data_entrega").reset_index(drop=True)
