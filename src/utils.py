# src/utils.py

import yaml

def load_config(config_path: str) -> dict:
    """YAML 설정 파일을 불러옵니다."""
    with open(config_path, "r", encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config