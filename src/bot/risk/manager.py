"""Gestão de risco: valida sinais antes de virarem ordens.

O risk manager tem poder de veto — nenhuma ordem é enviada sem passar
por aqui. Regras: risco máximo por trade, perda máxima diária, limite
de posições abertas e janela de horário permitida.
"""

from dataclasses import dataclass, field
from datetime import datetime, time


@dataclass
class RiskConfig:
    capital: float
    max_risk_per_trade_pct: float
    max_daily_loss_pct: float
    max_open_positions: int
    trading_start: time = time(9, 15)
    trading_end: time = time(17, 30)


@dataclass
class RiskManager:
    config: RiskConfig
    daily_pnl: float = 0.0
    open_positions_count: int = 0
    _blocked_today: bool = field(default=False, init=False)

    def can_open_position(self, now: datetime | None = None) -> tuple[bool, str]:
        """Retorna (permitido, motivo). Motivo explica o veto quando negado."""
        now = now or datetime.now()

        if self._blocked_today:
            return False, "Limite de perda diária atingido — bot pausado até amanhã"

        max_loss = self.config.capital * self.config.max_daily_loss_pct / 100
        if self.daily_pnl <= -max_loss:
            self._blocked_today = True
            return False, f"Perda diária de {self.daily_pnl:.2f} atingiu o limite de {max_loss:.2f}"

        if self.open_positions_count >= self.config.max_open_positions:
            return False, f"Já existem {self.open_positions_count} posições abertas (máx: {self.config.max_open_positions})"

        if not (self.config.trading_start <= now.time() <= self.config.trading_end):
            return False, f"Fora da janela de operação ({self.config.trading_start}–{self.config.trading_end})"

        return True, "ok"

    def position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calcula a quantidade com base no risco máximo por trade.

        Quantidade tal que, se o stop for atingido, a perda não passa de
        max_risk_per_trade_pct do capital.
        """
        risk_amount = self.config.capital * self.config.max_risk_per_trade_pct / 100
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return 0.0
        return risk_amount / risk_per_unit

    def register_trade_result(self, pnl: float) -> None:
        self.daily_pnl += pnl

    def reset_day(self) -> None:
        self.daily_pnl = 0.0
        self._blocked_today = False
