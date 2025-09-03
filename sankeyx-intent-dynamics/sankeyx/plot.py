
import plotly.graph_objects as go
from typing import Dict, List, Tuple

def build_sankey(nodes: List[str], links: Dict[str, List[int]]) -> go.Figure:
    """Minimal Plotly sankey builder.
    nodes: label list
    links: dict with keys source, target, value (int arrays)
    """
    fig = go.Figure(data=[
        go.Sankey(
            node=dict(label=nodes, pad=15, thickness=15),
            link=dict(
                source=links.get("source", []),
                target=links.get("target", []),
                value=links.get("value", []),
            ),
        )
    ])
    fig.update_layout(title="SankeyX — Intent Dynamics", font_size=12)
    return fig


def ensure_node(name, label, color, session_index=None):
    if name not in node_idx:
        node_idx[name] = len(nodes)
        nodes.append(name); node_labels.append(label); node_colors.append(color)
        if session_index is not None:
            node_idx_to_session[node_idx[name]] = session_index
    return node_idx[name]

