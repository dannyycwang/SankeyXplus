
"""SankeyX — Intent Dynamics (modular package).

High-level modules:
  - data_io: CSV loading, schema validation, caching
  - intent: dynamic intent timeline strategies (late/hys) + bridge NA
  - plot: plotly sankey construction + Streamlit adapters
  - gif: timeline GIF export (matplotlib + Pillow)
  - llm: optional Ollama inference helpers
  - config: constants and typed settings
  - utils: shared helpers

The legacy monolithic app is still available at the repo root.
"""

__all__ = [
    "data_io", "intent", "plot", "gif", "llm", "config", "utils"
]
