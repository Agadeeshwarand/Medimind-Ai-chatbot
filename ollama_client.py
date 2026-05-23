from __future__ import annotations

import os
from typing import Optional

import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")


def is_ollama_enabled() -> bool:
    return bool(OLLAMA_MODEL.strip())


def generate_with_ollama(prompt: str, timeout: int = 60) -> Optional[str]:
    if not is_ollama_enabled():
        return None

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return None

    data = response.json()
    answer = data.get("response", "").strip()
    return answer or None
