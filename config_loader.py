# config_loader.py
import json
import os

def load_config(path: str = "configure_llm.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config
