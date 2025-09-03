
import json, requests
from typing import Dict, Any

def ollama_summarize(endpoint: str, model: str, prompt: str, temperature: float = 0.2) -> str:
    """Minimal Ollama client (compatible with /api/generate)."""
    url = endpoint.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "temperature": temperature, "stream": False}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"[LLM error] {e}"


def call_ollama(prompt: str, model: str = "mistral", temperature: float = 0.2, timeout: int = 120) -> str:
    """Call a local Ollama server (http://localhost:11434) to generate LLM text."""
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": float(temperature)}},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        st.error(f"Couldn't reach Ollama at http://localhost:11434. Is it running? Try: `ollama run {model}`")
        return ""
    except Exception as e:
        st.error(f"LLM error: {e}")
        return ""

