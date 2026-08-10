"""Ponto de entrada do bot de trading.

Carrega a configuração, instancia corretora, estratégia e gestor de risco,
e inicia o loop principal de operação.
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {CONFIG_PATH}\n"
            "Copie config/config.example.yaml para config/config.yaml e preencha."
        )
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    mode = config["bot"]["mode"]
    print(f"Trade Bot iniciando em modo '{mode}'...")
    # TODO: instanciar broker, estratégia e risk manager a partir da config
    # TODO: loop principal — coletar dados, gerar sinal, validar risco, executar


if __name__ == "__main__":
    main()
