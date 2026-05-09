from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path=CONFIG_PATH):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
