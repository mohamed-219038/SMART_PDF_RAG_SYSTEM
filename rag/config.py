import os
import yaml


DEFAULT_CONFIG = {
    "env": {
        "dotenv_path": "pass.env",
        "api_key_name": "API_KEY",
    },
    "chunking": {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "separators": ["\n\n", "\n", ".", " ", ""],
        "add_start_index": True,
    },
    "embeddings": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "retrieval": {
        "top_k": 4,
    },
    "llm": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "temperature": 0,
    },
}


def load_config(path: str = "config.yaml"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        return deep_merge(DEFAULT_CONFIG, user_cfg)
    return DEFAULT_CONFIG.copy()


def deep_merge(base: dict, override: dict):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
