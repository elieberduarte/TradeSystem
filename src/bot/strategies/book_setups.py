"""Setups extraídos dos livros de price action, com as lacunas fechadas.

⚠️ AVISO DE HONESTIDADE, válido para todos os setups deste módulo:
nenhum dos autores especifica entrada, stop E alvo ao mesmo tempo. As
regras de saída abaixo foram fixadas por nós, UMA VEZ, antes de rodar
qualquer teste, seguindo a formulação original quando ela existe. Não
foram otimizadas. Qualquer resultado positivo deve ser reportado como
"setup inspirado em X com saída definida por nós" — nunca como
"o setup de X funciona".
"""

import pandas as pd

from src.bot.strategies.base import BaseStrategy, Signal, SignalType
from src.bot.strategies.swing_reversion import atr


class OopsStrategy(BaseStrategy):
    """Oops Trade — Larry Williams, via Bill Eykyn.

    Regra do glossário de Eykyn, literal: *"If the market opens below
    Yesterday's Low and trades back to Yesterday's Low, then buy
    Yesterday's Low."*

    A tese é de exaustão do pânico: a abertura em gap de baixa
    liquida quem tinha stop abaixo da mínima de ontem; se o preço
    volta ao nível, o movimento era liquidação, não informação.

    Eykyn afirma que o lado vendido NÃO funciona ("no it doesn't seem
    to work if the market opens above YH"). Não apresenta evidência —
    por isso o parâmetro `side` permite falsificar a afirmação.

    SAÍDAS FIXADAS POR NÓS (os autores não as dão):
      stop  = abertura do dia, que por definição do gap está além da
              entrada e é um nível conhecido no momento da entrada
      alvo  = fechamento do dia (MOC), como na formulação original de
              Williams. No motor, um alvo distante mais a saída por
              tempo de 1 barra reproduz isso.
    """

    mode = "swing_trade"
    DEFAULTS = {
        # "long" (só compras, como o autor recomenda), "short" ou "both"
        "side": "long",
        # Gap mínimo em múltiplos do ATR para o setup contar
        "min_gap_atr": 0.0,
        "atr_period": 14,
    }

    def __init__(self, params: dict | None = None):
        super().__init__({**self.DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        if len(candles) < p["atr_period"] + 3:
            return hold

        today = candles.iloc[-1]
        yesterday = candles.iloc[-2]
        volatility = float(atr(candles, p["atr_period"]).iloc[-1])
        if volatility <= 0:
            return hold

        open_, high, low = float(today["open"]), float(today["high"]), float(today["low"])
        y_low, y_high = float(yesterday["low"]), float(yesterday["high"])

        # Compra: abriu abaixo da mínima de ontem e voltou a tocá-la
        if p["side"] in ("long", "both"):
            gap = y_low - open_
            if gap > 0 and gap >= p["min_gap_atr"] * volatility and high >= y_low:
                entry = y_low
                if open_ < entry:
                    return Signal(
                        symbol=symbol, type=SignalType.BUY, entry_price=entry,
                        stop_loss=open_,           # a própria abertura em gap
                        take_profit=entry + 10 * (entry - open_),  # longe: quem fecha é o tempo
                    )

        # Venda: espelho — o autor afirma que não funciona
        if p["side"] in ("short", "both"):
            gap = open_ - y_high
            if gap > 0 and gap >= p["min_gap_atr"] * volatility and low <= y_high:
                entry = y_high
                if open_ > entry:
                    return Signal(
                        symbol=symbol, type=SignalType.SELL, entry_price=entry,
                        stop_loss=open_,
                        take_profit=entry - 10 * (open_ - entry),
                    )

        return hold


class GapFadeStrategy(BaseStrategy):
    """Devolve o gap de abertura, mirando o fechamento anterior.

    O estudo de 39.497 gaps confirmou metade da afirmação de Eykyn: a
    probabilidade de o gap fechar no mesmo dia cai de 90% (gaps abaixo
    de 0,10 ATR) para 6,5% (acima de 1,5 ATR), com correlação de
    −0,402. A outra metade foi refutada: a continuação fica em 48–50%
    em TODAS as faixas, ou seja, o gap não prevê a direção do dia.

    Resta a pergunta que o estudo não responde: a alta probabilidade
    de fechamento vira lucro depois de pôr um stop? Um gap pequeno
    fecha quase sempre, mas rende pouco; um grande rende muito e quase
    nunca fecha. É o mesmo dilema que já medimos em toda parte, e só
    o backtest com stop resolve.

    ENTRADA: na abertura, contra o gap.
    ALVO: o fechamento do dia anterior (o gap "fechado").
    STOP: múltiplo do gap, além da abertura — fixado por nós.
    """

    mode = "swing_trade"
    DEFAULTS = {
        "min_gap_atr": 0.10,
        "max_gap_atr": 1.00,
        # Stop em múltiplos do próprio gap, além da abertura
        "stop_mult": 1.0,
        "atr_period": 14,
    }

    def __init__(self, params: dict | None = None):
        super().__init__({**self.DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        if len(candles) < p["atr_period"] + 3:
            return hold

        volatility = float(atr(candles, p["atr_period"]).iloc[-1])
        if volatility <= 0:
            return hold

        open_ = float(candles["open"].iloc[-1])
        prev_close = float(candles["close"].iloc[-2])
        gap = open_ - prev_close
        size = abs(gap) / volatility
        if not (p["min_gap_atr"] <= size <= p["max_gap_atr"]):
            return hold

        distance = abs(gap)
        stop_distance = distance * p["stop_mult"]

        if gap > 0:      # abriu acima: vende mirando a volta
            return Signal(
                symbol=symbol, type=SignalType.SELL, entry_price=open_,
                stop_loss=open_ + stop_distance, take_profit=prev_close,
            )
        return Signal(   # abriu abaixo: compra mirando a volta
            symbol=symbol, type=SignalType.BUY, entry_price=open_,
            stop_loss=open_ - stop_distance, take_profit=prev_close,
        )


class InsideDayStrategy(BaseStrategy):
    """Rompimento de dia de dentro (Bill Eykyn).

    *"An Inside Day is defined as having a range which is within the
    high and low of the previous day. The rule here is to buy a break
    on Yesterday's High or sell a break on Yesterday's Low."*

    Após um dia inteiramente contido no anterior, opera-se o
    rompimento dos extremos DESSE dia de dentro (confirmado pelos
    exemplos do autor, não os extremos da barra-mãe).

    Filtro opcional de range estreito (Narrow Range, também do Eykyn):
    ele observa que dias de dentro costumam ser dias de amplitude
    reduzida e que a compressão precede o movimento. Aqui a
    compressão é medida em percentil do próprio ativo, porque os "12
    ticks" do T-Bond de 2003 não transferem para outro instrumento.

    SAÍDAS FIXADAS POR NÓS: stop no extremo oposto do dia de dentro
    (o nível que invalida a tese), alvo em múltiplo do risco.
    """

    mode = "swing_trade"
    DEFAULTS = {
        "side": "both",
        "rr": 2.0,
        # 0 desativa; senão exige range no percentil N dos últimos 20 dias
        "narrow_pct": 0.0,
        "narrow_lookback": 20,
    }

    def __init__(self, params: dict | None = None):
        super().__init__({**self.DEFAULTS, **(params or {})})

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        p = self.params
        hold = Signal(symbol=symbol, type=SignalType.HOLD)
        if len(candles) < p["narrow_lookback"] + 4:
            return hold

        today = candles.iloc[-1]
        inside = candles.iloc[-2]       # candidato a dia de dentro
        mother = candles.iloc[-3]

        is_inside = (
            float(inside["high"]) < float(mother["high"])
            and float(inside["low"]) > float(mother["low"])
        )
        if not is_inside:
            return hold

        if p["narrow_pct"]:
            ranges = (candles["high"] - candles["low"]).iloc[-p["narrow_lookback"] - 2 : -2]
            threshold = float(ranges.quantile(p["narrow_pct"]))
            if float(inside["high"] - inside["low"]) > threshold:
                return hold

        top, bottom = float(inside["high"]), float(inside["low"])
        risk = top - bottom
        if risk <= 0:
            return hold

        high, low = float(today["high"]), float(today["low"])

        # Rompimento para cima: entra no nível, stop no extremo oposto
        if p["side"] in ("long", "both") and high > top:
            return Signal(
                symbol=symbol, type=SignalType.BUY, entry_price=top,
                stop_loss=bottom, take_profit=top + p["rr"] * risk,
            )
        if p["side"] in ("short", "both") and low < bottom:
            return Signal(
                symbol=symbol, type=SignalType.SELL, entry_price=bottom,
                stop_loss=top, take_profit=bottom - p["rr"] * risk,
            )
        return hold
