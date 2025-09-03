
# -*- coding: utf-8 -*-
import os, runpy, streamlit as st
import pandas as pd
from sankeyx import data_io, intent, plot, gif, config, utils

st.set_page_config(layout="wide")
st.title("SankeyX — Intent Dynamics (Modular)")
st.caption("Auto-extracted modules available: intent/plot/gif/data_io/llm/extras")

st.sidebar.header("Data & Options")
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
strategy = st.sidebar.selectbox("Intent Strategy", ["hys", "late"], index=0)
inertia_k = st.sidebar.slider("Inertia K", 1, 5, 2)
use_bridge_na = st.sidebar.checkbox("Bridge NA", value=True)

settings = config.AppSettings(strategy=strategy, inertia_k=inertia_k, use_bridge_na=use_bridge_na)

# Fallback control
use_legacy = st.sidebar.checkbox("Force legacy runner", value=False)

if csv_file is None:
    st.info("Upload a CSV to start. See schema in README.")
    st.stop()

# Load & validate
df = pd.read_csv(csv_file)
ok, missing = data_io.validate_schema(df)
if not ok:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Try to compute first-row timeline via modular intent
try:
    seq = utils.parse_sequence(df.iloc[0]["truncated_sequence"])[: settings.max_steps]
    timeline = intent.compute_intent_timeline(seq, settings)
    st.write("First-row timeline (demo):", timeline)

    # Make a tiny demo Sankey (nodes + one link)
    nodes = ["Intent", "Click 1", "Prediction", "Utility"]
    links = {"source": [0], "target": [1], "value": [1]}
    fig = plot.build_sankey(nodes, links)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"Modular path had an error: {e}. Falling back to legacy app.")
    use_legacy = True

# Fallback: run legacy script within Streamlit app context
if use_legacy:
    st.info("Running legacy app: SankeyX_with_intent_dynamics.py")
    legacy_path = os.path.join(os.path.dirname(__file__), "SankeyX_with_intent_dynamics.py")
    if os.path.exists(legacy_path):
        runpy.run_path(legacy_path, run_name="__main__")
    else:
        st.error("Legacy file not found.")
