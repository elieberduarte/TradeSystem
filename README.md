# Trade Bot

Bot para operações de **day trade** e **swing trade**, com arquitetura modular que separa coleta de dados, estratégias, gestão de risco e execução de ordens.

## Estrutura do projeto

```
Trade/
├── config/
│   └── config.example.yaml   # Modelo de configuração (copie para config.yaml)
├── src/
│   └── bot/
│       ├── data/             # Coleta e armazenamento de dados de mercado (candles, book, etc.)
│       ├── strategies/       # Estratégias de trading (sinais de compra/venda)
│       ├── risk/             # Gestão de risco (stop loss, position sizing, limites diários)
│       ├── execution/        # Envio de ordens à corretora/exchange
│       ├── backtest/         # Backtesting das estratégias com dados históricos
│       └── main.py           # Ponto de entrada do bot
├── tests/                    # Testes automatizados
├── requirements.txt
└── README.md
```

## Princípios

1. **Corretora plugável** — a camada de execução usa uma interface abstrata (`BrokerInterface`), permitindo conectar B3 (MetaTrader 5, Profit), cripto (Binance, ccxt) ou outra corretora sem mudar as estratégias.
2. **Backtest antes de operar** — toda estratégia deve ser validada com dados históricos antes de ir para conta real.
3. **Risco em primeiro lugar** — o módulo de risco tem poder de veto sobre qualquer ordem (limite de perda diária, tamanho máximo de posição, horários permitidos).
4. **Paper trading** — modo simulado para validar o bot em tempo real sem arriscar capital.

## Como começar

```bash
# Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate    # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar
copy config\config.example.yaml config\config.yaml
# Edite config\config.yaml com suas credenciais e parâmetros

# Rodar o bot (modo paper trading por padrão)
python -m src.bot.main
```

## Aviso

Este software é fornecido para fins educacionais. Operações de day trade e swing trade envolvem risco significativo de perda financeira. Use por sua conta e risco, e sempre teste em modo simulado antes de operar com capital real.
