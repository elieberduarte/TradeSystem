"""Exporta a anatomia das operações para visualização — multiestratégia.

Para cada trade real do backtest: os candles ao redor, as linhas do
indicador que gerou o gatilho (canal de Donchian, bandas de Bollinger
ou Tenkan/Kijun), entrada, stop, alvo, ponto de saída, o MOTIVO da
saída e o GATILHO da entrada escrito por extenso. É o que permite ver
a mecânica em vez de só o resultado agregado.

Uso: python scripts/export_trade_anatomy.py [quantidade_por_estrategia]
"""

import json
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.bot.backtest.engine import BacktestEngine
from src.bot.data.history import HistoryStore
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.strategies.ichimoku import IchimokuCrossStrategy, ichimoku_lines
from src.bot.strategies.squeeze import SqueezeBreakoutStrategy, bollinger
from src.bot.strategies.swing_reversion import atr

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 150_000.0
POINT_VALUE = {"WIN$N": 0.20, "IND$N": 1.00, "WDO$N": 10.00, "DOL$N": 50.00}
PADDING = 25
PREFERRED = ["WIN$N", "ABEV3", "ITUB4", "VALE3", "WDO$N", "IVVB11", "PETR4", "WEGE3"]


def donchian_overlay(candles: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    upper = candles["high"].rolling(20).max().shift(1)
    lower = candles["low"].rolling(20).min().shift(1)
    return upper, lower


def donchian_trigger(candles, i, upper, lower, side) -> str:
    close = float(candles["close"].iloc[i])
    if side == "buy":
        return (f"fechou em {close:,.2f}, ACIMA da máxima dos últimos 20 pregões "
                f"({float(upper.iloc[i]):,.2f}) — rompimento de canal")
    return (f"fechou em {close:,.2f}, ABAIXO da mínima dos últimos 20 pregões "
            f"({float(lower.iloc[i]):,.2f}) — rompimento de canal")


def squeeze_overlay(candles: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    bands = bollinger(candles["close"])
    return bands["upper"], bands["lower"]


def squeeze_trigger(candles, i, upper, lower, side) -> str:
    bands = bollinger(candles["close"])
    window = bands["bandwidth"].iloc[max(0, i - 121) : i - 1].dropna()
    rank = float((window <= float(bands["bandwidth"].iloc[i - 1])).mean()) if len(window) else float("nan")
    band = float(upper.iloc[i]) if side == "buy" else float(lower.iloc[i])
    lado = "ACIMA da banda superior" if side == "buy" else "ABAIXO da banda inferior"
    return (f"largura de banda de ontem no percentil {rank:.0%} dos últimos 120 pregões "
            f"(mercado comprimido) e fechamento {lado} ({band:,.2f}) — explosão do squeeze")


def tk_overlay(candles: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    lines = ichimoku_lines(candles)
    return lines["tenkan"], lines["kijun"]


def tk_trigger(candles, i, tenkan, kijun, side) -> str:
    verbo = "cruzou para CIMA do" if side == "buy" else "cruzou para BAIXO do"
    return (f"Tenkan(9) {verbo} Kijun(26): {float(tenkan.iloc[i]):,.2f} × "
            f"{float(kijun.iloc[i]):,.2f} — o ponto médio curto tomou a frente do longo")


STRATEGIES = {
    "donchian": {
        "label": "Donchian 20 (titular da carteira)",
        "factory": lambda: DonchianStrategy({"channel": 20, "stop_atr": 2.0, "rr": 3.0, "long_only": True}),
        "overlay": donchian_overlay,
        "overlay_label": "canal de 20 pregões (máxima e mínima)",
        "trigger": donchian_trigger,
        "params_text": "canal 20 pregões · stop 2×ATR(14) · alvo 3× o risco · só compra",
        "rules": [
            {"k": "Gatilho de entrada", "cls": "",
             "v": "O preço FECHA acima da máxima dos últimos 20 pregões. Espera-se o fechamento de propósito: pavio que fura o canal e volta não conta."},
            {"k": "Stop", "cls": "stop",
             "v": "Entrada − 2×ATR(14). O ATR mede a oscilação típica recente, então o stop se adapta: largo em ativo volátil, curto em ativo calmo."},
            {"k": "Alvo", "cls": "target",
             "v": "3× a distância do stop. Acertar ~35% já é lucrativo: perdas pequenas e frequentes, ganhos grandes e raros."},
            {"k": "Tamanho", "cls": "size",
             "v": "(1% do capital) ÷ (distância do stop × valor do ponto). Risco em reais sempre igual; a quantidade é que varia. Sem preço médio, nunca."},
        ],
    },
    "squeeze": {
        "label": "Squeeze de Bollinger (reserva, Calmar 12,9 nos futuros)",
        "factory": SqueezeBreakoutStrategy,
        "overlay": squeeze_overlay,
        "overlay_label": "bandas de Bollinger (20, 2 desvios)",
        "trigger": squeeze_trigger,
        "params_text": "banda 20/2σ · squeeze = largura no quinto inferior de 120 pregões · stop 2×ATR · alvo 3R · só compra",
        "rules": [
            {"k": "Gatilho de entrada", "cls": "",
             "v": "Duas condições: ONTEM a largura das bandas estava no quinto mais estreito dos últimos 120 pregões (mercado comprimido) e HOJE o preço fechou acima da banda superior (a compressão explodiu para cima)."},
            {"k": "Stop", "cls": "stop",
             "v": "Entrada − 2×ATR(14), igual ao titular. O squeeze muda o gatilho, não a gestão — assim a comparação isola o sinal."},
            {"k": "Alvo", "cls": "target",
             "v": "3× a distância do stop. Acerto medido: 58,5% nos futuros — bem acima do necessário para esse payoff."},
            {"k": "Por que funciona", "cls": "size",
             "v": "Compressão precede expansão: é a MESMA tese do Inside Day (o único setup dos livros que replicou), em versão contínua e ajustada à volatilidade do próprio ativo."},
        ],
    },
    "tk_cross": {
        "label": "Tenkan × Kijun (arquivada: replica 92%, fraca nos futuros)",
        "factory": IchimokuCrossStrategy,
        "overlay": tk_overlay,
        "overlay_label": "Tenkan 9 (rápida) e Kijun 26 (lenta)",
        "trigger": tk_trigger,
        "params_text": "Tenkan 9 · Kijun 26 · stop 2×ATR · alvo 3R · só compra",
        "rules": [
            {"k": "Gatilho de entrada", "cls": "",
             "v": "A Tenkan (ponto médio de 9 pregões) cruza para cima da Kijun (ponto médio de 26). São canais de Donchian disfarçados: o Ichimoku é da mesma família do titular."},
            {"k": "Stop", "cls": "stop",
             "v": "Entrada − 2×ATR(14), padrão da casa."},
            {"k": "Alvo", "cls": "target",
             "v": "3× a distância do stop."},
            {"k": "Status", "cls": "size",
             "v": "Replica em 92% dos 28 instrumentos (mais que o próprio Donchian!) mas rende pouco nos futuros (Calmar 2,1). Arquivada como candidata — correlação de só 0,32 com o titular."},
        ],
    },
}


def risk_manager() -> RiskManager:
    return RiskManager(
        RiskConfig(
            capital=CAPITAL, max_risk_per_trade_pct=1.0,
            max_daily_loss_pct=100.0, max_weekly_loss_pct=6.0,
            max_open_positions=1, mode="swing_trade",
            trading_start=time(0, 0), trading_end=time(23, 59),
            max_consecutive_losses=0,
        )
    )


def run(config: dict, symbol: str, store: HistoryStore):
    candles = store.load(symbol, "1d")
    if candles is None or len(candles) < 700:
        return None, None
    point_value = POINT_VALUE.get(symbol, 1.0)
    slippage = 10.0 if symbol in ("WIN$N", "IND$N") else (0.5 if symbol in POINT_VALUE else 0.01)
    cost = 1.0 if symbol in POINT_VALUE else 0.01
    engine = BacktestEngine(
        config["factory"](), risk_manager(), point_value=point_value, warmup=210,
        slippage_points=slippage, cost_per_contract=cost,
    )
    return engine.run(symbol, candles), candles


def build_payload(key: str, config: dict, store: HistoryStore, limit: int) -> dict | None:
    result = candles = symbol = None
    for candidate in PREFERRED:
        result, candles = run(config, candidate, store)
        if result is not None and len(result.trades) >= 6:
            symbol = candidate
            break
    if symbol is None:
        return None

    upper, lower = config["overlay"](candles)
    volatility = atr(candles, 14)
    positions = {ts: i for i, ts in enumerate(candles.index)}

    winners = [t for t in result.trades if t.pnl > 0][::-1]
    losers = [t for t in result.trades if t.pnl <= 0][::-1]
    sample, i = [], 0
    while len(sample) < limit and (i < len(winners) or i < len(losers)):
        if i < len(winners):
            sample.append(winners[i])
        if len(sample) < limit and i < len(losers):
            sample.append(losers[i])
        i += 1
    sample.sort(key=lambda t: t.entry_time)

    trades = []
    for trade in sample:
        entry_idx = positions[trade.entry_time]
        start = max(0, entry_idx - PADDING)
        end = min(len(candles) - 1, positions[trade.exit_time] + PADDING)
        window = candles.iloc[start : end + 1]
        trades.append({
            "symbol": symbol,
            "side": trade.side,
            "entry_time": str(trade.entry_time.date()),
            "entry_price": round(trade.entry_price, 2),
            "exit_time": str(trade.exit_time.date()),
            "exit_price": round(trade.exit_price, 2),
            "exit_reason": trade.exit_reason,
            "stop_loss": round(trade.stop_loss, 2),
            "take_profit": round(trade.take_profit, 2),
            "quantity": trade.quantity,
            "pnl": round(trade.pnl, 2),
            "bars_held": (trade.exit_time - trade.entry_time).days,
            "atr_at_entry": round(float(volatility.loc[trade.entry_time]), 2),
            "trigger": config["trigger"](candles, entry_idx, upper, lower, trade.side),
            "candles": [
                {
                    "date": str(ts.date()),
                    "o": round(float(row["open"]), 2),
                    "h": round(float(row["high"]), 2),
                    "l": round(float(row["low"]), 2),
                    "c": round(float(row["close"]), 2),
                    "up": None if pd.isna(upper.loc[ts]) else round(float(upper.loc[ts]), 2),
                    "lo": None if pd.isna(lower.loc[ts]) else round(float(lower.loc[ts]), 2),
                }
                for ts, row in window.iterrows()
            ],
        })

    wins = sum(1 for t in result.trades if t.pnl > 0)
    return {
        "key": key,
        "label": config["label"],
        "symbol": symbol,
        "params_text": config["params_text"],
        "overlay_label": config["overlay_label"],
        "rules": config["rules"],
        "point_value": POINT_VALUE.get(symbol, 1.0),
        "capital": CAPITAL,
        "summary": {
            "trades": len(result.trades),
            "win_rate": round(wins / len(result.trades), 3),
            "total_pnl": round(result.total_pnl, 2),
            "expectancy": round(result.expectancy, 2),
            "max_drawdown": round(result.max_drawdown, 2),
            "longest_losing_streak": result.longest_losing_streak,
        },
        "trades_sample": trades,
    }


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    store = HistoryStore()

    payloads = {}
    for key, config in STRATEGIES.items():
        payload = build_payload(key, config, store, limit)
        if payload is None:
            print(f"{key}: nenhum símbolo com trades suficientes")
            continue
        payloads[key] = payload
        s = payload["summary"]
        print(f"{key} · {payload['symbol']}: {s['trades']} trades, "
              f"{s['win_rate']:.1%} acerto, amostra de {len(payload['trades_sample'])}")

    out = ROOT / "web" / "anatomy.json"
    out.write_text(
        json.dumps({"strategies": payloads}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\nExportado para {out}")


if __name__ == "__main__":
    main()
