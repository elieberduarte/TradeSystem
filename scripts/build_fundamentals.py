"""Constrói o painel de fundamentos: as 131 ações líquidas, precificadas.

Fontes: DFP consolidada da CVM (lucro, patrimônio, receita, dívida,
LPA, dividendos pagos) + preço atual do MT5. Métricas clássicas:

  P/L        preço ÷ lucro por ação (o LPA vem PUBLICADO na DRE —
             sem estimar quantidade de ações para isso)
  P/VP       via ações estimadas = lucro ÷ LPA (aproximação declarada)
  ROE        lucro ÷ patrimônio líquido
  margem     lucro ÷ receita
  dív/PL     empréstimos (circulante + longo prazo) ÷ patrimônio
  DY (aprox) dividendos+JCP pagos no ano (DFC) ÷ valor de mercado
  cresc.     receita vs ano anterior (a própria DFP traz os dois)

Point-in-time honesto: usamos o balanço MAIS RECENTE já entregue.
Para o PAINEL (foto de hoje) isso é correto; para backtest de fator,
a data de entrega será obrigatória.

Uso: python scripts/build_fundamentals.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MetaTrader5 as mt5
import pandas as pd

from src.bot.data.cvm import load_dfp, ticker_map

ROOT = Path(__file__).resolve().parents[1]
MIN_VOLUME = 5e6
BLUE_CHIP_VOLUME = 50e6       # corte de liquidez para o ranking blue chip
YEARS = (2026, 2025)          # zips da DFP; o mais novo prevalece (foto atual)
YEARS_HISTORY = (2022, 2023, 2024, 2025, 2026)   # série para consistência


def scale(frame: pd.DataFrame) -> pd.Series:
    factor = frame["ESCALA_MOEDA"].map({"MIL": 1_000.0}).fillna(1.0)
    return frame["VL_CONTA"] * factor


def account(frame: pd.DataFrame, cnpj_col: str, *codes: str, order: str = "ÚLTIMO") -> pd.Series:
    """Soma das contas pedidas por CNPJ, no exercício indicado."""
    mask = frame["ORDEM_EXERC"].str.upper().str.startswith(order[:3].upper())
    part = frame[mask & frame["CD_CONTA"].isin(codes)].copy()
    part["valor"] = scale(part)
    return part.groupby(cnpj_col)["valor"].sum()


def lpa_by_class(dre: pd.DataFrame, cnpj_col: str) -> pd.DataFrame:
    """LPA básico por classe (ON/PN), publicado nas contas 3.99.*.

    IMPORTANTE: o LPA já vem em R$ POR AÇÃO — a ESCALA_MOEDA (mil)
    vale para as contas monetárias, não para esta. Escalar aqui foi o
    bug que deixou todos os P/L em 0,0 na primeira rodada.
    """
    mask = (
        dre["ORDEM_EXERC"].str.upper().str.startswith("ÚLT")
        & dre["CD_CONTA"].str.startswith("3.99.01")
    )
    part = dre[mask].copy()
    part["classe"] = part["DS_CONTA"].str.upper().str.extract(r"\b(ON|PN)\b")[0]
    part["valor"] = part["VL_CONTA"]
    return part.dropna(subset=["classe"]).groupby([cnpj_col, "classe"])["valor"].first().unstack()


def equity_series(bpp: pd.DataFrame, cnpj_col: str) -> pd.Series:
    """Patrimônio líquido consolidado — 2.03 nas empresas, 2.08 nos bancos.

    A seleção exige "Patrimônio" na descrição para o código de banco
    não capturar outra conta em empresas comuns.
    """
    mask = (
        bpp["ORDEM_EXERC"].str.upper().str.startswith("ÚLT")
        & bpp["CD_CONTA"].isin(["2.03", "2.08"])
        & bpp["DS_CONTA"].str.contains("Patrim", case=False, na=False)
    )
    part = bpp[mask].copy()
    part["valor"] = scale(part)
    return part.groupby(cnpj_col)["valor"].first()


def dividends_paid(cnpj_col: str) -> pd.Series:
    """Dividendos + JCP pagos no ano, pelo fluxo de caixa (valor absoluto)."""
    pieces = []
    for year in YEARS:
        for table in ("DFC_MD_con", "DFC_MI_con"):
            try:
                dfc = load_dfp(year, table)
            except Exception:  # noqa: BLE001
                continue
            descriptions = dfc["DS_CONTA"].str.lower()
            mask = (
                dfc["ORDEM_EXERC"].str.upper().str.startswith("ÚLT")
                & descriptions.str.contains("dividendo|juros sobre", na=False)
                # "dividendos RECEBIDOS" é caixa que entra — não é payout
                & ~descriptions.str.contains("receb", na=False)
            )
            part = dfc[mask].copy()
            part["valor"] = scale(part).abs()
            part["ano"] = year
            pieces.append(part[[cnpj_col, "valor", "ano"]])
    if not pieces:
        return pd.Series(dtype=float)
    merged = pd.concat(pieces)
    latest = merged.sort_values("ano").groupby(cnpj_col).tail(0)  # placeholder
    # Ano mais recente com dado prevalece
    best_year = merged.groupby(cnpj_col)["ano"].max()
    merged = merged.merge(best_year.rename("melhor"), on=cnpj_col)
    return merged[merged["ano"] == merged["melhor"]].groupby(cnpj_col)["valor"].sum()


def yearly_history(cnpj_col_hint: str | None = None) -> pd.DataFrame:
    """Receita, lucro e PL por empresa e ANO FISCAL, 2022-2026.

    A base da medição de consistência: crescimento composto e anos
    lucrativos só fazem sentido com série, não com foto.
    """
    rows = []
    for year in YEARS_HISTORY:
        try:
            dre = load_dfp(year, "DRE_con")
            bpp = load_dfp(year, "BPP_con")
        except Exception:  # noqa: BLE001
            continue
        cnpj_col = [c for c in dre.columns if "CNPJ" in c.upper()][0]
        fiscal = dre["DT_FIM_EXERC"].astype(str).str[:4].astype(int)
        dre = dre.assign(_ano=fiscal)
        for ano, dre_y in dre.groupby("_ano"):
            profit = account(dre_y, cnpj_col, "3.11")
            revenue = account(dre_y, cnpj_col, "3.01")
            equity = equity_series(bpp, cnpj_col)
            for cnpj in profit.index:
                rows.append({
                    "cnpj": cnpj, "ano": int(ano),
                    "lucro": float(profit[cnpj]),
                    "receita": float(revenue.get(cnpj, float("nan"))),
                    "pl": float(equity.get(cnpj, float("nan"))),
                })
    history = pd.DataFrame(rows)
    return history.drop_duplicates(subset=["cnpj", "ano"], keep="last")


def consistency_metrics(history: pd.DataFrame) -> pd.DataFrame:
    """CAGR de receita, anos lucrativos e ROE médio por empresa."""
    out = []
    for cnpj, group in history.groupby("cnpj"):
        group = group.sort_values("ano")
        if len(group) < 3:
            continue
        revenue = group["receita"].dropna()
        cagr = None
        if len(revenue) >= 3 and revenue.iloc[0] > 0 and revenue.iloc[-1] > 0:
            span = group["ano"].iloc[-1] - group["ano"].iloc[0]
            if span > 0:
                cagr = (revenue.iloc[-1] / revenue.iloc[0]) ** (1 / span) - 1
        profitable = int((group["lucro"] > 0).sum())
        roe_values = (group["lucro"] / group["pl"]).replace(
            [float("inf"), float("-inf")], float("nan")
        ).dropna()
        out.append({
            "cnpj": cnpj,
            "cagr_receita": round(cagr, 4) if cagr is not None else None,
            "anos_lucrativos": profitable,
            "anos_observados": len(group),
            "roe_medio": round(float(roe_values.mean()), 4) if len(roe_values) else None,
        })
    return pd.DataFrame(out).set_index("cnpj")


def blue_chip_ranking(frame: pd.DataFrame) -> pd.Series:
    """Ranking composto de QUALIDADE entre os papéis mais líquidos.

    Não é previsão de crescimento — crescimento passado não persiste
    (Chan/Karceski/Lakonishok). O que persiste é qualidade: ROE médio,
    histórico de lucro, endividamento, payout. Média de postos com
    igual peso; ausência de dado vale o pior posto (conservador).
    """
    eligible = frame[frame["volume_mediano_mi"] >= BLUE_CHIP_VOLUME / 1e6].copy()
    if eligible.empty:
        return pd.Series(dtype=float)
    ranks = pd.DataFrame(index=eligible.index)
    ranks["roe"] = eligible["roe_medio"].rank(ascending=False)
    ranks["lucrativo"] = (
        eligible["anos_lucrativos"] / eligible["anos_observados"]
    ).rank(ascending=False)
    ranks["divida"] = eligible["divida_pl"].rank(ascending=True)
    ranks["dy"] = eligible["dy"].rank(ascending=False)
    ranks["cagr"] = eligible["cagr_receita"].rank(ascending=False)
    worst = len(eligible)
    score = ranks.fillna(worst).mean(axis=1)
    return score.rank(method="first").astype(int)


SMALL_CAP_MAX_BI = 10.0       # valor de mercado máximo (R$ bi) para small cap


def small_growth_ranking(frame: pd.DataFrame) -> pd.Series:
    """Small caps de crescimento LUCRATIVO — não de crescimento a qualquer preço.

    "Small growth" puro é o pior canto histórico do mercado (as ações-
    loteria); o prêmio documentado em small caps é o de QUALIDADE. O
    ranking exige as duas coisas ao mesmo tempo: expansão de receita E
    retorno sobre capital, com histórico de lucro e dívida sob controle.
    """
    eligible = frame[
        frame["valor_mercado_bi"].notna()
        & (frame["valor_mercado_bi"] <= SMALL_CAP_MAX_BI)
    ].copy()
    if eligible.empty:
        return pd.Series(dtype=float)
    ranks = pd.DataFrame(index=eligible.index)
    ranks["cagr"] = eligible["cagr_receita"].rank(ascending=False)
    ranks["roe"] = eligible["roe_medio"].rank(ascending=False)
    ranks["lucrativo"] = (
        eligible["anos_lucrativos"] / eligible["anos_observados"]
    ).rank(ascending=False)
    ranks["margem"] = eligible["margem"].rank(ascending=False)
    ranks["divida"] = eligible["divida_pl"].rank(ascending=True)
    worst = len(eligible)
    score = ranks.fillna(worst).mean(axis=1)
    return score.rank(method="first").astype(int)


def main() -> None:
    scan = json.loads((ROOT / "web" / "universe_scan.json").read_text(encoding="utf-8"))
    liquid = {r["symbol"]: r for r in scan
              if r["tipo"] == "ação" and r["volume_mediano"] >= MIN_VOLUME}
    print(f"{len(liquid)} ações líquidas no censo")

    mapping = ticker_map(range(2023, 2027))
    mapping = mapping[mapping["ticker"].isin(liquid)]
    print(f"{len(mapping)} com CNPJ no cadastro FCA")

    # Demonstrações: o zip mais novo por empresa prevalece
    frames = {"dre": [], "bpp": []}
    for year in YEARS:
        try:
            frames["dre"].append(load_dfp(year, "DRE_con").assign(_ano=year))
            frames["bpp"].append(load_dfp(year, "BPP_con").assign(_ano=year))
        except Exception as exc:  # noqa: BLE001
            print(f"  DFP {year}: indisponível ({exc})")
    dre = pd.concat(frames["dre"], ignore_index=True)
    bpp = pd.concat(frames["bpp"], ignore_index=True)
    cnpj_col = [c for c in dre.columns if "CNPJ" in c.upper()][0]

    # Mantém só o ano-zip mais recente de cada empresa
    for name, frame in (("dre", dre), ("bpp", bpp)):
        newest = frame.groupby(cnpj_col)["_ano"].transform("max")
        if name == "dre":
            dre = frame[frame["_ano"] == newest]
        else:
            bpp = frame[frame["_ano"] == newest]

    profit = account(dre, cnpj_col, "3.11")
    profit_prev = account(dre, cnpj_col, "3.11", order="PENÚLTIMO")
    revenue = account(dre, cnpj_col, "3.01")
    revenue_prev = account(dre, cnpj_col, "3.01", order="PENÚLTIMO")
    equity = equity_series(bpp, cnpj_col)
    debt = account(bpp, cnpj_col, "2.01.04", "2.02.01")
    lpa = lpa_by_class(dre, cnpj_col)
    dividends = dividends_paid(cnpj_col)

    if not mt5.initialize():
        raise SystemExit(f"MT5 indisponível: {mt5.last_error()}")

    rows = []
    for _, item in mapping.iterrows():
        ticker, cnpj = item["ticker"], item["cnpj"]
        if cnpj not in profit.index or cnpj not in equity.index:
            continue
        rates = mt5.copy_rates_from_pos(ticker, mt5.TIMEFRAME_D1, 0, 1)
        if rates is None or not len(rates):
            continue
        price = float(rates[-1]["close"])

        klass = "PN" if ticker.endswith(("4", "5", "6")) else "ON"
        lpa_value = None
        if cnpj in lpa.index:
            row = lpa.loc[cnpj]
            # NaN é "verdadeiro" no Python e vaza por qualquer `or` — o
            # filtro precisa ser explícito
            options = [row.get(klass), row.get("ON"), row.get("PN")]
            lpa_value = next(
                (float(v) for v in options if v is not None and pd.notna(v)), None
            )

        ll, pl = float(profit[cnpj]), float(equity[cnpj])
        shares = ll / lpa_value if lpa_value else None
        market_cap = price * shares if shares and shares > 0 else None

        rec = float(revenue.get(cnpj, float("nan")))
        rec_prev = float(revenue_prev.get(cnpj, float("nan")))
        rows.append({
            "ticker": ticker,
            "empresa": item["empresa"][:40],
            "preco": round(price, 2),
            "volume_mediano_mi": round(liquid[ticker]["volume_mediano"] / 1e6, 1),
            "pl": round(price / lpa_value, 2) if lpa_value and lpa_value > 0 else None,
            "pvp": round(market_cap / pl, 2) if market_cap and pl > 0 else None,
            "roe": round(ll / pl, 4) if pl > 0 else None,
            "margem": round(ll / rec, 4) if rec and rec > 0 else None,
            "divida_pl": round(float(debt.get(cnpj, 0.0)) / pl, 2) if pl > 0 else None,
            "dy": round(float(dividends.get(cnpj, 0.0)) / market_cap, 4)
                  if market_cap and cnpj in dividends.index else None,
            "cresc_receita": round(rec / rec_prev - 1, 4)
                             if rec and rec_prev and rec_prev > 0 else None,
            "lucro_mi": round(ll / 1e6, 0),
            "valor_mercado_bi": round(market_cap / 1e9, 2) if market_cap else None,
        })
    mt5.shutdown()

    frame = pd.DataFrame(rows)

    # Série histórica 2022-2026: consistência é o que persiste
    print("Construindo histórico de 5 anos (DFP 2022-2026)…")
    history = consistency_metrics(yearly_history())
    frame = frame.merge(
        history, left_on=frame["ticker"].map(
            mapping.set_index("ticker")["cnpj"]
        ).rename("cnpj"), right_index=True, how="left",
    )
    frame["blue_chip_rank"] = blue_chip_ranking(frame)
    frame["blue_chip_rank"] = frame["blue_chip_rank"].astype("Int64")
    frame["small_growth_rank"] = small_growth_ranking(frame)
    frame["small_growth_rank"] = frame["small_growth_rank"].astype("Int64")

    # Cruzamento barato × qualidade × solidez (medianas do próprio universo)
    valid = frame.dropna(subset=["pl", "roe", "divida_pl"])
    positive = valid[valid["pl"] > 0]
    med_pl = positive["pl"].median()
    med_roe = valid["roe"].median()
    med_debt = valid["divida_pl"].median()
    frame["barato_qualidade"] = (
        (frame["pl"] > 0) & (frame["pl"] <= med_pl)
        & (frame["roe"] >= med_roe) & (frame["divida_pl"] <= med_debt)
    )

    out = ROOT / "web" / "fundamentals.json"
    # to_json converte NaN em null (JSON válido); json.dumps escreveria
    # o token NaN e o navegador rejeitaria o arquivo inteiro
    records = json.loads(frame.to_json(orient="records", force_ascii=False))
    out.write_text(json.dumps({
        "atualizado": pd.Timestamp.now().isoformat(timespec="seconds"),
        "medianas": {"pl": round(med_pl, 1), "roe": round(med_roe, 3),
                     "divida_pl": round(med_debt, 2)},
        "papeis": records,
    }, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")

    picks = frame[frame["barato_qualidade"]].sort_values("pl")
    print(f"\n{len(frame)} ações com balanço + preço · "
          f"medianas: P/L {med_pl:.1f} · ROE {med_roe:.1%} · dív/PL {med_debt:.2f}")

    smalls = frame.dropna(subset=["small_growth_rank"]).sort_values("small_growth_rank")
    print(f"\n── Small caps de crescimento lucrativo (valor de mercado ≤ "
          f"R$ {SMALL_CAP_MAX_BI:.0f} bi) ──")
    print(f"{'#':>3} {'ticker':<8} {'VM(bi)':>7} {'CAGR rec':>9} {'ROE 5a':>8} "
          f"{'lucros':>7} {'margem':>8} {'P/L':>6}")
    for _, r in smalls.head(15).iterrows():
        cagr = f"{r['cagr_receita']:.1%}" if pd.notna(r["cagr_receita"]) else "—"
        roe_m = f"{r['roe_medio']:.1%}" if pd.notna(r["roe_medio"]) else "—"
        margem = f"{r['margem']:.1%}" if pd.notna(r["margem"]) else "—"
        pl_txt = f"{r['pl']:.1f}" if pd.notna(r["pl"]) else "—"
        print(f"{int(r['small_growth_rank']):>3} {r['ticker']:<8} "
              f"{r['valor_mercado_bi']:>7} {cagr:>9} {roe_m:>8} "
              f"{int(r['anos_lucrativos'])}/{int(r['anos_observados']):<5} {margem:>8} {pl_txt:>6}")

    blues = frame.dropna(subset=["blue_chip_rank"]).sort_values("blue_chip_rank")
    print(f"\n── Carteira blue chip candidata (qualidade composta, "
          f"liquidez ≥ R$ {BLUE_CHIP_VOLUME / 1e6:.0f} mi/dia) ──")
    print(f"{'#':>3} {'ticker':<8} {'ROE méd':>8} {'lucro':>7} {'CAGR rec':>9} "
          f"{'DY':>6} {'dív/PL':>7} {'P/L':>6}")
    for _, r in blues.head(15).iterrows():
        cagr = f"{r['cagr_receita']:.1%}" if pd.notna(r["cagr_receita"]) else "—"
        dy = f"{r['dy']:.1%}" if pd.notna(r["dy"]) else "—"
        roe_m = f"{r['roe_medio']:.1%}" if pd.notna(r["roe_medio"]) else "—"
        pl_txt = f"{r['pl']:.1f}" if pd.notna(r["pl"]) else "—"
        print(f"{int(r['blue_chip_rank']):>3} {r['ticker']:<8} {roe_m:>8} "
              f"{int(r['anos_lucrativos'])}/{int(r['anos_observados']):<5} {cagr:>9} "
              f"{dy:>6} {r['divida_pl'] if pd.notna(r['divida_pl']) else '—':>7} {pl_txt:>6}")
    print(f"\n── Barato × qualidade × solidez ({len(picks)} papéis) ──")
    print(f"{'ticker':<8} {'P/L':>6} {'P/VP':>6} {'ROE':>7} {'DY':>6} {'dív/PL':>7}")
    for _, r in picks.head(20).iterrows():
        dy = f"{r['dy']:.1%}" if r["dy"] is not None else "—"
        pvp = f"{r['pvp']:.1f}" if r["pvp"] is not None else "—"
        print(f"{r['ticker']:<8} {r['pl']:>6.1f} {pvp:>6} {r['roe']:>6.1%} "
              f"{dy:>6} {r['divida_pl']:>7.2f}")
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
