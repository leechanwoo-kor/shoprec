# src/utils/config.py
import yaml


def load_config(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
