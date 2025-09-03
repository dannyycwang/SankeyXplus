
import pandas as pd
from typing import Tuple, List

REQUIRED_COLS = ["truncated_sequence", "purchase", "y_pred"]
OPTIONAL_COLS = ["session_id_hash", "Intent_type"]

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return (len(missing) == 0, missing)


def extract_json_block(text: str):
    """
    Extract the last ```json ... ``` fenced block, or try raw JSON parsing.
    """
    if not text:
        return None
    code_blocks = re.findall(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = code_blocks[-1].strip() if code_blocks else text.strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None

