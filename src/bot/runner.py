"""O ciclo diário do bot: dados → sinais → decisões → ordens.

A carteira aprovada é 100% mecânica e de baixa frequência: Donchian
só-compra em candles diários, stop e alvo anexados à ordem. Isso
define a arquitetura:

- O MT5 executa stop e alvo sozinho, 24h, sem o bot ligado.
- O bot só precisa acordar UMA vez por dia, perto do fechamento
  (17h40), para: rolar contratos a vencer, ler os sinais do dia e
  abrir as posições que couberem nas vagas.
- Uma rodada opcional de manhã (9h05) repesca sinais de véspera de
  mercados que já estavam fechados às 17h40 (agro encerra ~16h) —
  nela o candle parcial de hoje é descartado.

O planejamento (`plan_cycle`) é uma função PURA: recebe candles,
posições e configuração, devolve decisões com o motivo escrito. Toda
regra auditável está ali, testável sem corretora. A execução
(`Runner`) só traduz decisões em ordens — e se recusa a operar conta
real sem opt-in explícito no config.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.bot.backtest.slots import RULES
from src.bot.data.history import HistoryStore
from src.bot.execution.broker import BrokerInterface, Order, Position
from src.bot.risk.manager import RiskConfig, RiskManager
from src.bot.strategies.donchian import DonchianStrategy
from src.bot.universe import FUT_MARGIN, FUT_POINT_VALUE, FUTUROS, fut_block_of

MONTH_CODE = r"[FGHJKMNQUVXZ]\d\d$"


@dataclass
class RunnerConfig:
    capital: float = 100_000.0
    slots: int = 10
    rule: str = "bloco"
    max_risk_per_trade_pct: float = 1.0
    max_weekly_loss_pct: float = 6.0
    symbols: list[str] = field(default_factory=lambda: list(FUTUROS))
    channel: int = 20
    stop_atr: float = 2.0
    rr: float = 3.0
    # Rola posições a N dias do vencimento; não abre contrato mais curto
    roll_days: int = 3
    min_history: int = 300
    # Trava de segurança: conta real exige allow_real=True no config
    allow_real: bool = False


@dataclass
class Decision:
    action: str            # "abrir" | "rolar" | "vetar" | "observar"
    symbol: str            # símbolo contínuo (sinais) ou contrato (rolagem)
    reason: str
    side: str = "buy"
    quantity: int = 0
    entry_ref: float = 0.0       # preço de referência do sinal
    stop_distance: float = 0.0   # distâncias em pontos, ancoradas no fill
    target_distance: float = 0.0


def root_of(symbol: str) -> str:
    """'WIN$N' → 'WIN'; 'DI1F27' → 'DI1F27' (contrato exato é a própria raiz)."""
    return symbol[:-2] if symbol.endswith("$N") else symbol


def _continuous_of(root: str) -> str:
    """'WIN' → 'WIN$N'; 'DI1F27' → 'DI1F27' (o contrato exato é o símbolo)."""
    continuous = f"{root}$N"
    return continuous if continuous in FUTUROS else root


def position_root(position_symbol: str, known_roots: list[str]) -> str | None:
    """Mapeia 'WINV26' de volta à raiz 'WIN' do nosso universo."""
    if position_symbol in known_roots:      # DI1F27 etc.
        return position_symbol
    for root in sorted(known_roots, key=len, reverse=True):
        if position_symbol.startswith(root) and re.fullmatch(
            MONTH_CODE, position_symbol[len(root):]
        ):
            return root
    return None


@dataclass
class _Candidate:
    """Adaptador para as regras de seleção de `slots.py`."""
    symbol: str
    block: str
    margin: float
    decision: Decision


def plan_cycle(
    candles: dict[str, pd.DataFrame],
    positions: list[Position],
    expiries: dict[str, datetime | None],
    weekly_pnl: float,
    config: RunnerConfig,
    now: datetime,
) -> list[Decision]:
    """Decide o ciclo inteiro a partir de dados já coletados. Função pura."""
    decisions: list[Decision] = []
    known_roots = [root_of(s) for s in config.symbols]
    held_roots: set[str] = set()

    # 1. Posições existentes: identificar e rolar as que estão vencendo
    for position in positions:
        root = position_root(position.symbol, known_roots)
        if root is None:
            continue                        # posição alheia ao bot: não toca
        held_roots.add(root)
        expiry = expiries.get(position.symbol)
        if expiry is not None:
            days_left = (expiry - now.astimezone(timezone.utc)).days
            if days_left <= config.roll_days:
                decisions.append(Decision(
                    action="rolar", symbol=position.symbol,
                    side=position.side, quantity=int(position.quantity),
                    reason=f"vence em {days_left} dia(s) — rolar para o próximo",
                ))

    # 2. Trava semanal: com a perda da semana no limite, nada de novo
    weekly_limit = config.capital * config.max_weekly_loss_pct / 100
    if config.max_weekly_loss_pct and weekly_pnl <= -weekly_limit:
        decisions.append(Decision(
            action="vetar", symbol="*",
            reason=(f"perda semanal de {weekly_pnl:,.0f} atingiu o limite de "
                    f"{weekly_limit:,.0f} — sem entradas até segunda"),
        ))
        return decisions

    # 3. Sinais do dia
    risk = RiskManager(RiskConfig(
        capital=config.capital,
        max_risk_per_trade_pct=config.max_risk_per_trade_pct,
        max_daily_loss_pct=100.0, max_open_positions=config.slots,
        mode="swing_trade", max_consecutive_losses=0,
        risk_slots=1, cash_slots=config.slots, enforce_cash=True,
    ))
    strategy_params = {
        "channel": config.channel, "stop_atr": config.stop_atr,
        "rr": config.rr, "long_only": True,
    }

    candidates: list[_Candidate] = []
    for symbol in config.symbols:
        frame = candles.get(symbol)
        if frame is None or len(frame) < config.min_history:
            continue
        root = root_of(symbol)
        if root in held_roots:
            continue                        # já posicionado: sinal repetido

        signal = DonchianStrategy(strategy_params).generate_signal(symbol, frame)
        if signal.type.name != "BUY":
            continue

        quantity = int(risk.position_size(
            signal.entry_price, signal.stop_loss,
            FUT_POINT_VALUE.get(symbol, 1.0), FUT_MARGIN.get(symbol),
        ))
        decision = Decision(
            action="abrir", symbol=symbol, side="buy", quantity=quantity,
            entry_ref=signal.entry_price,
            stop_distance=signal.entry_price - signal.stop_loss,
            target_distance=signal.take_profit - signal.entry_price,
            reason=f"rompimento do canal de {config.channel} dias",
        )
        if quantity <= 0:
            decision.action = "vetar"
            decision.reason = "sinal válido, mas risco/caixa não compram 1 contrato"
            decisions.append(decision)
            continue
        candidates.append(_Candidate(
            symbol=symbol, block=fut_block_of(symbol),
            margin=FUT_MARGIN.get(symbol, 0.0), decision=decision,
        ))

    # 4. Vagas: quem entra quando há mais sinais que espaço
    free = config.slots - len(held_roots)
    if len(candidates) > max(free, 0):
        held = [
            _Candidate(root, fut_block_of(_continuous_of(root)), 0.0, None)  # type: ignore[arg-type]
            for root in held_roots
        ]
        ranked = RULES[config.rule](candidates, held, None)
        chosen = set(id(c) for c in ranked[: max(free, 0)])
        for candidate in candidates:
            if id(candidate) in chosen:
                decisions.append(candidate.decision)
            else:
                skipped = candidate.decision
                skipped.action = "observar"
                skipped.reason = f"sem vaga ({config.slots} ocupadas) — regra {config.rule}"
                decisions.append(skipped)
    else:
        decisions.extend(c.decision for c in candidates)

    return decisions


class Runner:
    """Orquestra o ciclo: coleta dados, planeja, executa e registra."""

    def __init__(
        self,
        broker: BrokerInterface,
        store: HistoryStore,
        config: RunnerConfig,
        journal_path: Path | str = "data/journal.jsonl",
        status_path: Path | str = "web/live_status.json",
    ):
        self.broker = broker
        self.store = store
        self.config = config
        self.journal_path = Path(journal_path)
        self.status_path = Path(status_path)

    # ------------------------------------------------------------- ciclo

    def cycle(self, execute: bool = False, include_today: bool = True) -> list[Decision]:
        now = datetime.now()
        self._refresh_history()

        candles = {}
        for symbol in self.config.symbols:
            frame = self.store.load(symbol, "1d")
            if frame is None:
                continue
            # Rodada matinal: o candle de hoje está pela metade e mentiria
            # para a estratégia — o sinal é o de ontem, executado na abertura
            if not include_today and len(frame) and frame.index[-1].date() == now.date():
                frame = frame.iloc[:-1]
            candles[symbol] = frame

        positions = self.broker.get_open_positions()
        expiries = {p.symbol: self._expiry_of(p.symbol) for p in positions}
        weekly_pnl = self.broker.realized_pnl(since_days=now.weekday() + 1)

        decisions = plan_cycle(candles, positions, expiries, weekly_pnl, self.config, now)

        if execute:
            self._execute(decisions)
        self._journal(decisions, executed=execute)
        self._write_status(decisions, positions, executed=execute)
        return decisions

    # ---------------------------------------------------------- internos

    def _refresh_history(self) -> None:
        for symbol in self.config.symbols:
            try:
                self.store.update_from_broker(self.broker, symbol, "1d", limit=1_000)
            except Exception as exc:  # noqa: BLE001 — um símbolo fora do ar não para o ciclo
                print(f"  aviso: {symbol} sem atualização ({exc})")
        self._accumulate_b3_flow()

    def _accumulate_b3_flow(self) -> None:
        """Acumula o fluxo oficial dos players (BDI) a cada ciclo.

        A API da B3 só retém ~21 pregões; rodando junto com o bot, o
        acervo local cresce sem intervenção. Falha aqui nunca derruba o
        ciclo — é coleta de pesquisa, não de operação.
        """
        try:
            from datetime import date
            from src.bot.data.b3_bdi import collect_open_interest, collect_participacao, workdays
            days = sorted(workdays(str(date.today())))
            collect_participacao(days)
            collect_open_interest(days)
        except Exception as exc:  # noqa: BLE001
            print(f"  aviso: fluxo B3 não acumulado neste ciclo ({exc})")

    def _expiry_of(self, symbol: str):
        probe = getattr(self.broker, "contract_expiry", None)
        return probe(symbol) if probe else None

    def _execute(self, decisions: list[Decision]) -> None:
        if not self.broker.is_demo() and not self.config.allow_real:
            raise RuntimeError(
                "Conta REAL detectada e allow_real=False no config. "
                "O bot só opera conta real com opt-in explícito."
            )
        for decision in decisions:
            try:
                if decision.action == "abrir":
                    self._open(decision)
                elif decision.action == "rolar":
                    self._roll(decision)
            except Exception as exc:  # noqa: BLE001 — registra e segue para a próxima
                decision.action = "falhou"
                decision.reason += f" | erro: {exc}"

    def _open(self, decision: Decision) -> None:
        root = root_of(decision.symbol)
        contract = (
            self.broker.front_contract(root, self.config.roll_days)
            if decision.symbol.endswith("$N") else decision.symbol
        )
        price = self.broker.last_price(contract)
        order = Order(
            symbol=contract, side="buy", quantity=float(decision.quantity),
            stop_loss=price - decision.stop_distance,
            take_profit=price + decision.target_distance,
        )
        order_id = self.broker.place_order(order)
        decision.reason += f" | executada: {decision.quantity}x {contract} @~{price} (#{order_id})"

    def _roll(self, decision: Decision) -> None:
        """Fecha o contrato a vencer e reabre igual no próximo vencimento.

        Stop e alvo preservam as DISTÂNCIAS até o preço atual do contrato
        velho — os níveis absolutos mudam com o novo contrato (carrego),
        as distâncias não.
        """
        old_symbol = decision.symbol
        positions = [p for p in self.broker.get_open_positions() if p.symbol == old_symbol]
        if not positions:
            decision.reason += " | posição não encontrada — nada a rolar"
            return
        position = positions[0]
        old_price = self.broker.last_price(old_symbol)
        stop_distance = abs(old_price - position.stop_loss) if position.stop_loss else 0.0
        target_distance = abs(position.take_profit - old_price) if position.take_profit else 0.0

        root = position_root(old_symbol, [root_of(s) for s in self.config.symbols])
        new_contract = self.broker.front_contract(root, self.config.roll_days)
        self.broker.close_position(old_symbol)

        direction = 1 if position.side == "buy" else -1
        new_price = self.broker.last_price(new_contract)
        order = Order(
            symbol=new_contract, side=position.side, quantity=position.quantity,
            stop_loss=(new_price - direction * stop_distance) if stop_distance else None,
            take_profit=(new_price + direction * target_distance) if target_distance else None,
        )
        order_id = self.broker.place_order(order)
        decision.reason += f" | rolada: {old_symbol} → {new_contract} (#{order_id})"

    def _journal(self, decisions: list[Decision], executed: bool) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().isoformat(timespec="seconds")
        with self.journal_path.open("a", encoding="utf-8") as handle:
            for decision in decisions:
                handle.write(json.dumps(
                    {"ts": stamp, "executado": executed, **asdict(decision)},
                    ensure_ascii=False,
                ) + "\n")

    def _write_status(self, decisions, positions, executed: bool) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps({
            "atualizado": datetime.now().isoformat(timespec="seconds"),
            "executado": executed,
            "saldo": self.broker.get_balance(),
            "posicoes": [asdict(p) for p in positions],
            "decisoes": [asdict(d) for d in decisions],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
