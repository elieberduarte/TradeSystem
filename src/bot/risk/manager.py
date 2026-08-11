"""Gestão de risco: valida sinais antes de virarem ordens.

O risk manager tem poder de veto — nenhuma ordem é enviada sem passar
por aqui. Camadas:

1. Por operação: risco máximo por trade define o tamanho da posição.
2. Por dia: perda máxima diária, limite de trades, trava de derrotas
   consecutivas e janela de horário (com zeragem antes do fechamento).
"""

from dataclasses import dataclass, field
from datetime import datetime, time


@dataclass
class RiskConfig:
    capital: float
    max_risk_per_trade_pct: float
    max_daily_loss_pct: float
    max_open_positions: int
    # day_trade = zera no fim do pregão | swing_trade = carrega overnight
    mode: str = "day_trade"
    trading_start: time = time(9, 15)
    trading_end: time = time(17, 30)
    # Horário de zeragem: posições abertas devem ser fechadas (day trade
    # não dorme posicionado). Novas entradas param em trading_end.
    flat_time: time = time(17, 45)
    # 0 = sem limite
    max_trades_per_day: int = 0
    # Derrotas seguidas que pausam o dia (0 = desativado)
    max_consecutive_losses: int = 3
    # Perda máxima na semana em % do capital (0 = desativado) — pensada
    # para swing, onde a perda se acumula em dias, não em horas
    max_weekly_loss_pct: float = 0.0


@dataclass
class RiskManager:
    config: RiskConfig
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    open_positions_count: int = 0
    trades_today: int = 0
    consecutive_losses: int = 0
    _blocked_today: bool = field(default=False, init=False)
    _block_reason: str = field(default="", init=False)

    def can_open_position(self, now: datetime | None = None) -> tuple[bool, str]:
        """Retorna (permitido, motivo). Motivo explica o veto quando negado."""
        now = now or datetime.now()

        if self._blocked_today:
            return False, self._block_reason

        max_loss = self.config.capital * self.config.max_daily_loss_pct / 100
        if self.daily_pnl <= -max_loss:
            self._block_day(
                f"Perda diária de {self.daily_pnl:.2f} atingiu o limite de {max_loss:.2f}"
            )
            return False, self._block_reason

        if self.config.max_weekly_loss_pct:
            max_weekly = self.config.capital * self.config.max_weekly_loss_pct / 100
            if self.weekly_pnl <= -max_weekly:
                return False, (
                    f"Perda semanal de {self.weekly_pnl:.2f} atingiu o limite "
                    f"de {max_weekly:.2f} — sem novas entradas nesta semana"
                )

        if (
            self.config.max_consecutive_losses
            and self.consecutive_losses >= self.config.max_consecutive_losses
        ):
            self._block_day(
                f"{self.consecutive_losses} derrotas consecutivas — dia encerrado"
            )
            return False, self._block_reason

        if self.config.max_trades_per_day and self.trades_today >= self.config.max_trades_per_day:
            return False, f"Limite de {self.config.max_trades_per_day} trades no dia atingido"

        if self.open_positions_count >= self.config.max_open_positions:
            return False, f"Já existem {self.open_positions_count} posições abertas (máx: {self.config.max_open_positions})"

        if not (self.config.trading_start <= now.time() <= self.config.trading_end):
            return False, f"Fora da janela de operação ({self.config.trading_start}–{self.config.trading_end})"

        return True, "ok"

    def should_flatten(self, now: datetime | None = None) -> bool:
        """Chegou a hora de zerar posições abertas (fim do dia)?

        Swing trade carrega posição overnight — nunca zera por horário.
        """
        if self.config.mode == "swing_trade":
            return False
        now = now or datetime.now()
        return now.time() >= self.config.flat_time

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
        self.weekly_pnl += pnl
        self.trades_today += 1
        if pnl < 0:
            self.consecutive_losses += 1
        elif pnl > 0:
            self.consecutive_losses = 0

    def reset_day(self) -> None:
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self._blocked_today = False
        self._block_reason = ""

    def reset_week(self) -> None:
        self.weekly_pnl = 0.0

    def _block_day(self, reason: str) -> None:
        self._blocked_today = True
        self._block_reason = f"{reason} — bot pausado até amanhã"
