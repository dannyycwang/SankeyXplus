
import ast
from typing import List

def parse_sequence(seq_str: str) -> List[int]:
    """Robustly parse list-like string such as "[1,2,3]" into a python list."""
    if not isinstance(seq_str, str):
        return []
    try:
        v = ast.literal_eval(seq_str)
        if isinstance(v, list):
            return [int(x) for x in v if x is not None]
    except Exception:
        pass
    return []
