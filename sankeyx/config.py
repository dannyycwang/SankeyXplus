
from dataclasses import dataclass, field
from typing import List, Dict, Optional

INTENT_LABELS = [
    "Hesitant", "Exploratory", "Engaged", "Intermittent", "Comparative", "Uncertain"
]

@dataclass
class UtilityMatrix:
    tp: float = 2.0
    tn: float = 0.5
    fp: float = -1.0
    fn: float = -2.0

@dataclass
class AppSettings:
    use_bridge_na: bool = True
    inertia_k: int = 2           # for hys strategy
    strategy: str = "hys"        # "late" or "hys"
    max_steps: int = 11
    ollama_endpoint: str = "http://localhost:11434"
    ollama_model: str = "mistral"

DEFAULT_SETTINGS = AppSettings()
DEFAULT_UTILITY = UtilityMatrix()
