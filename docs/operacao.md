# Operação do bot — como fica ligado

Desde 17/08/2026 o bot e o gravador de fluxo rodam pelo **Agendador de
Tarefas do Windows**, independentes de qualquer terminal (sobrevivem a
logout e reinicialização; o PC precisa estar ligado e sem suspender nos
horários):

| tarefa               | horário (dias úteis) | comando                          | log                              |
|----------------------|----------------------|----------------------------------|----------------------------------|
| TradeSystem-Flow     | 09:00 → 18:20        | `scripts/record_flow.py 18:20`   | `data/logs/TradeSystem-Flow.log`   |
| TradeSystem-Morning  | 09:05                | `main.py morning` (repesca)      | `data/logs/TradeSystem-Morning.log`|
| TradeSystem-Run      | 17:40                | `main.py run` (ciclo principal)  | `data/logs/TradeSystem-Run.log`    |

Por que não `main.py loop` num terminal: no primeiro dia o processo
morreu junto com a sessão que o abriu (17:37, três minutos antes do
ciclo). Tarefas agendadas não têm pai para morrer junto.

## Conferir

- `python main.py status` — conta, posições, PnL da semana
- `data/journal.jsonl` — uma linha por decisão; ciclos sem sinal também
  registram (`"action": "ciclo"`) para distinguir "rodou e não achou
  nada" de "não rodou"
- `Get-ScheduledTaskInfo TradeSystem-Run` — LastRunTime / LastTaskResult
  (0 = ok; 267009 = ainda executando)

## Religar/alterar

```powershell
Get-ScheduledTask -TaskName "TradeSystem-*"          # listar
Start-ScheduledTask -TaskName "TradeSystem-Run"     # disparar agora
Disable-ScheduledTask -TaskName "TradeSystem-Flow"  # pausar uma
```

Requisitos permanentes: MT5 aberto e logado na conta demo XP; `config/
config.yaml` com `allow_real: false` (conta real exige opt-in explícito).
