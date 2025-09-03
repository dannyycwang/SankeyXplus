# -*- coding: utf-8 -*-
import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plotly.graph_objects as go
import ast
from collections import Counter, defaultdict
import requests
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import tempfile
from io import BytesIO
import uuid
# ──────────────────────────────────────────────────────────────────────────────
# Force GPU for Ollama (Windows 可在 CMD: set OLLAMA_USE_GPU=1)
# ──────────────────────────────────────────────────────────────────────────────
os.environ["OLLAMA_USE_GPU"] = "1"

# Try to enable click events on Plotly (optional dependency)
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Data & Viz
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("Parameter Setting")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Please upload a clickstream CSV file to begin.")
    st.stop()

def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return val

# Basic guards
if 'truncated_sequence' not in df.columns:
    st.error("CSV must contain 'truncated_sequence' column (list-like string of ints).")
    st.stop()

# Parse sequence column
df['truncated_sequence'] = df['truncated_sequence'].apply(safe_eval)

max_rows = df.shape[0]
num_sessions = st.sidebar.slider("N Sessions", 1, min(200, max_rows), 20)
max_steps = st.sidebar.slider("Max Steps (Clicks)", 1, 20, 5)
session_mode = st.sidebar.radio("Order", options=["first", "last"], index=0)
intent_types = ["All"] + sorted(df['Intent_type'].dropna().unique()) if 'Intent_type' in df.columns else ["All"]
intent_filter = st.sidebar.selectbox("Filter by (static) Intent Type", options=intent_types, index=0)
shap_mult = st.sidebar.slider("SHAP ×", 1, 100, 10)

# Important: enable per-session nodes/links
separate_mode = st.sidebar.checkbox("Separate Sessions (click to select)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Event Colors**")
event_color_map = {
    1: st.sidebar.color_picker("Browse", "#1f77b4"),
    2: st.sidebar.color_picker("Detail", "#ffbe0b"),
    3: st.sidebar.color_picker("Add", "#43aa8b"),
    4: st.sidebar.color_picker("Remove", "#fb8500"),
    5: st.sidebar.color_picker("Purchase", "#3a86ff"),
}
intent_flow_color = st.sidebar.color_picker("Intent ➜ First Event", "#c8d6e5")
show_stage_headers = st.sidebar.checkbox("Show stage headers", value=True)
show_session_ids = st.sidebar.checkbox("Show Session IDs", value=True)

INTENTS_ORDER = ["Hesitant Buyer","Exploratory Buyer","Engaged Buyer",
                 "Intermittent Revisitor","Comparative Buyer","Uncertain Buyer"]
# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Intent Dynamics settings
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Intent Dynamics")
dyn_strategy = st.sidebar.selectbox("Selection strategy", options=["late", "hys (late + inertia)"], index=0)
dyn_K = st.sidebar.slider("Inertia K (steps)", 1, 5, 2)
dyn_bridge = st.sidebar.checkbox("Bridge NA (carry last intent across NA)", value=True)
use_dynamic_as_left_column = st.sidebar.checkbox("Main Sankey uses dynamic intent (last at Max Steps)", value=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — LLM Settings
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("LLM Settings (Ollama)")
llm_model = st.sidebar.text_input("Model", value="mistral", help="Any local Ollama model tag: mistral, llama3.2, qwen2.5, etc.")
llm_temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
send_mode = st.sidebar.radio("Context Mode", options=["Compact summary"], index=0)
auto_generate = st.sidebar.checkbox("Auto-generate explanation", value=False)

# ──────────────────────────────────────────────────────────────────────────────
# LLM helpers (Ollama)
# ──────────────────────────────────────────────────────────────────────────────




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

# ── PATCH 1: robust paragraphs renderer ───────────────────────────────────────
def render_paragraphs(text):
    """
    Split text into paragraphs by blank lines and render each as a block.
    Bullet-like lines are rendered as list items.
    Tolerates dict/list by rendering JSON directly.
    """
    if text is None:
        return
    # dict / list 直接用 JSON 呈現
    if isinstance(text, (dict, list)):
        st.json(text)
        return
    # 其他非字串型別，轉成字串
    if not isinstance(text, str):
        text = str(text)

    t = text.replace('\r\n', '\n').replace('\r', '\n').strip()
    blocks = [b.strip() for b in t.split('\n\n') if b.strip()]

    def is_bullet_line(ln: str) -> bool:
        ln = ln.strip()
        return (
            ln.startswith('- ') or ln.startswith('* ')
            or ln.startswith('•') or ln.startswith('–') or ln.startswith('—')
            or bool(re.match(r'^\d+\.\s', ln))
        )

    for b in blocks:
        lines = [ln.strip() for ln in b.split('\n') if ln.strip()]
        bulletish = sum(1 for ln in lines if is_bullet_line(ln))
        if bulletish >= max(2, len(lines)//2):
            for ln in lines:
                ln = re.sub(r'^\s*(?:[-*•–—]|\d+\.)\s*', '', ln)
                st.markdown(f"- {ln}")
        else:
            st.markdown(b)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
EVENT_MAP = {1: "browse", 2: "detail", 3: "add", 4: "remove", 5: "purchase"}
def get_last_intent(record, max_steps, session_mode="last", fallback="Unclassified"):
    """
    根據設定的 max_steps 與 session_mode 取出最後一個 intent
    """
    tl = record.get("intent_timeline_raw", []) or []
    if not tl:
        return fallback

    # 依照 mode 取前 N 或後 N
    if session_mode == "first":
        sub_tl = tl[:max_steps]
    else:  # last
        sub_tl = tl[-max_steps:]

    # 取最後一個有效值
    if len(sub_tl) == 0:
        return fallback
    return sub_tl[-1] if sub_tl[-1] is not None else fallback

def node_label_with_id(nm, lbl):
    if nm == "UTILITY":
        return "Utility: {:.2f}".format(total_utility)

    sidx = node_idx_to_session.get(node_idx[nm])
    if sidx is not None and show_session_ids:
        sid = records[sidx]['session_id_hash'][:6]

        # 只在 Intent 節點顯示 Session ID，其它節點不顯示
        if nm.startswith("DYN_INTENT_LAST_") or nm.startswith("GROUP_INTENT_"):
            return f"{lbl} ({sid})" if lbl else sid
        else:
            return lbl
    return lbl


def sequence_to_text(seq):
    return [EVENT_MAP.get(int(x), str(x)) for x in seq if x != 0]

def sort_shap_cols(cols):
    def key_fn(c):
        try:
            return int(c.split('_')[1])
        except:
            return 10**6
    return sorted(cols, key=key_fn)

# Dynamic intent rules (simplified)
def normalize_tokens(num_seq):
    tokens = []
    for x in num_seq:
        if x == 0:
            continue
        t = EVENT_MAP.get(int(x), str(x))
        tokens.append(t)
    return tokens

def check_intent1(trace):
    if "browse" not in trace: return None
    for i, e in enumerate(trace):
        if e == "browse":
            window = trace[i+1:i+6]
            if any(x in window for x in ["detail", "add"]): continue
            return True
    return False if "browse" in trace else None

def check_intent2(trace):
    if "browse" not in trace: return None
    seen_detail = any(e == "detail" for e in trace)
    return True if (not seen_detail and trace.count("browse") >= 2) else None

def check_intent3(trace):
    if "detail" not in trace: return None
    for i, e in enumerate(trace):
        if e == "detail" and any(x in trace[i+1:] for x in ["add","detail"]):
            return True
    return False

def check_intent4(trace):
    if "add" not in trace: return None
    for i, e in enumerate(trace):
        if e == "add" and any(x in trace[i+1:i+6] for x in ["detail"]):
            return True
    return False

def check_intent5(trace):
    if "remove" not in trace: return None
    for i, e in enumerate(trace):
        if e == "remove" and not any(x in trace[i+1:] for x in ["add","detail"]):
            return True
    return False

def check_intent6(trace):
    for i in range(len(trace)-2):
        if trace[i]=="detail" and trace[i+1]=="browse" and trace[i+2]=="detail":
            return True
    return None

rules = {
    "Intent1": check_intent1,  # Hesitant
    "Intent6": check_intent6,  # Intermittent
    "Intent2": check_intent2,  # Exploratory
    "Intent3": check_intent3,  # Engaged
    "Intent4": check_intent4,  # Comparative
    "Intent5": check_intent5,  # Uncertain
}
intent_name_by_key = {
    "Intent1": "Hesitant Buyer",
    "Intent2": "Exploratory Buyer",
    "Intent3": "Engaged Buyer",
    "Intent4": "Comparative Buyer",
    "Intent5": "Uncertain Buyer",
    "Intent6": "Intermittent Revisitor",
}
stage_order = {
    "Hesitant Buyer": 1,
    "Exploratory Buyer": 1,
    "Engaged Buyer": 2,
    "Intermittent Revisitor": 2.5,
    "Comparative Buyer": 3,
    "Uncertain Buyer": 4,
}


def eval_rules_on_prefix(prefix_tokens):
    return {name: func(prefix_tokens) for name, func in rules.items()}

def intent_timeline_for_trace(trace_tokens, bridge_na=True, strategy="late", K=2):
    per_step_true = []
    for t in range(1, len(trace_tokens)+1):
        prefix = trace_tokens[:t]
        res = eval_rules_on_prefix(prefix)
        true_set = [intent_name_by_key[k] for k, v in res.items() if v is True]
        per_step_true.append(true_set)

    if strategy == "late":
        raw = []
        for true_set in per_step_true:
            if not true_set: raw.append(None)
            else: raw.append(sorted(true_set, key=lambda x: stage_order[x], reverse=True)[0])
    else:
        from collections import defaultdict
        consec_true = defaultdict(int); current=None; raw=[]
        for true_set in per_step_true:
            for it in stage_order.keys():
                consec_true[it] = consec_true[it] + 1 if it in true_set else 0
            target = max(true_set, key=lambda x: stage_order[x]) if true_set else None
            if current is None: current = target
            else:
                if target is not None:
                    cur_rank = stage_order.get(current, -1)
                    tar_rank = stage_order[target]
                    if tar_rank != cur_rank and consec_true[target] >= K:
                        current = target
            raw.append(current)

    if bridge_na:
        bridged = []; last=None
        for x in raw:
            if x is None: bridged.append(last)
            else: bridged.append(x); last=x
    else:
        bridged = raw

    segments = []
    for x in bridged:
        if x is None: continue
        if not segments or segments[-1] != x: segments.append(x)
    return raw, bridged, segments

# ──────────────────────────────────────────────────────────────────────────────
# Build selected records
# ──────────────────────────────────────────────────────────────────────────────
all_shap_cols = sort_shap_cols([c for c in df.columns if c.upper().startswith('SHAP_')])
if intent_filter != "All" and 'Intent_type' in df.columns:
    filtered = df[df['Intent_type'] == intent_filter]
else:
    filtered = df
selected = filtered.head(num_sessions) if session_mode == 'first' else filtered.tail(num_sessions)

utility_dict = {(1,1):3, (1,0):-1, (0,1):-2.5, (0,0):1e-6}
outcome_color_map = {'TP':'#06d6a0','TN':'#118ab2','FP':'#FF6B6B','FN':'#ffd166'}
intent_color_map = {
    "Hesitant Buyer":         "#ffe066",
    "Comparative Buyer":      "#6c63ff",
    "Unclassified":           "#adb5bd",
    "Exploratory Buyer":      "#A3F7BF",
    "Intermittent Revisitor": "#9bf6ff",
    "Engaged Buyer":          "#ffb4a2",
    "Uncertain Buyer":        "#b983ff",
}

records = []
for _, row in selected.iterrows():
    # 1) 先取最後 max_steps
    if session_mode == 'first':
        seq_full = list(row['truncated_sequence'])[:max_steps]
    else:  # 'last'
        seq_full = list(row['truncated_sequence'])[-max_steps:]
    seq_len = len(seq_full)

    # 2) 對應抓同長度的 SHAP（還沒截 5）
    shap_vals_full = [float(row.get(col, 0.0)) for col in all_shap_cols]
    shap_slice_full = shap_vals_full[-seq_len:] if seq_len > 0 else []
    shap_slice_full = [v * shap_mult for v in shap_slice_full]

    # 3) 『遇到第一個 5（purchase）就截斷』（含 5）
    cut_idx = None
    for i, e in enumerate(seq_full):
        try:
            if int(e) == 5:
                cut_idx = i
                break
        except Exception:
            pass
    if cut_idx is not None:
        seq = seq_full[:cut_idx+1]
        shap_values = shap_slice_full[:cut_idx+1]
    else:
        seq = seq_full
        shap_values = shap_slice_full

    # 4) 空序列防呆（完全沒有步驟）
    #    - 不建立任何事件節點/連線（Sankey 部分會被跳過）
    #    - 仍保留在 records 內給 LLM 統計
    y_pred = int(row.get('y_pred', 0))
    y_true = int(row.get('purchase', 0))
    outcome = 'TP' if (y_pred, y_true) == (1, 1) else (
              'TN' if (y_pred, y_true) == (0, 0) else (
              'FP' if (y_pred, y_true) == (1, 0) else 'FN'))
    utility = utility_dict.get((y_pred, y_true), 0.0)
    static_intent = row.get('Intent_type', 'Unclassified')

    # 5) 用「截斷後」的序列做動態 intent
    tokens = normalize_tokens(seq)
    strat = "late" if dyn_strategy.startswith("late") else "hys"
    raw_tl, bridged_tl, segments = intent_timeline_for_trace(
        tokens, bridge_na=dyn_bridge, strategy=strat, K=dyn_K
    )

    records.append({
        'session_id_hash': row.get('session_id_hash', f's{len(records)}'),
        'sequence': seq,                 # ← 已經是「截斷後」的序列
        'shap': shap_values,             # ← 已經是「截斷後」對齊的 SHAP
        'outcome': outcome,
        'utility': utility,
        'static_intent': static_intent,
        'intent_timeline': bridged_tl,   # ← 用截斷後 tokens 推出的動態 intent
        'intent_timeline_raw': raw_tl,
        'y_pred': y_pred,
        'y_true': y_true,
    })


# ──────────────────────────────────────────────────────────────────────────────
# GIF generator — wide, discrete X axis (no decimals), always max_steps frames
# ──────────────────────────────────────────────────────────────────────────────
def generate_intent_gif(intent_list, max_steps, save_path, intents_order=INTENTS_ORDER, fps=2):
    """
    只依照實際 intent_list 長度繪製（不 padding）。
    - 若 intent_list 為空，輸出 1 幀佔位圖（顯示 No steps）。
    - X 軸為 1..T（T = min(max_steps, len(intent_list))）。
    """
    # 1) 實際要畫的步數（不補尾、不上限於 max_steps）
    actual_len = min(max_steps, len(intent_list)) if intent_list else 0

    # 2) 把 intent 轉成 y 軸（Unknown/NA=0）
    def to_y(v):
        return intents_order.index(v) + 1 if v in intents_order else 0

    if actual_len == 0:
        # 輸出 1 幀的佔位 GIF（空 timeline）
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.set_yticks([0] + list(range(1, len(intents_order)+1)))
        ax.set_yticklabels(["Unknown/NA"] + intents_order)
        ax.set_xlim(1, 1)
        ax.set_ylim(0, len(intents_order)+1)
        ax.set_xlabel("Step")
        ax.set_title("Customer Intent Timeline (dynamic) — No steps")
        ax.grid(True, axis="y", alpha=0.3)

        (line,) = ax.step([], [], where="post")
        (pts,) = ax.plot([], [], "o", markersize=4)
        step_text = ax.text(0.99, 0.92, "No steps", transform=ax.transAxes, ha="right", va="top")

        def update(_frame):
            return (line, pts, step_text)

        ani = FuncAnimation(fig, update, frames=1, interval=int(1000/fps), blit=True)
        ani.save(save_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return

    # 3) 正常有步數的情況
    trimmed = intent_list[:actual_len]
    y_full = [to_y(v) for v in trimmed]
    x_full = list(range(1, actual_len+1))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_yticks([0] + list(range(1, len(intents_order)+1)))
    ax.set_yticklabels(["Unknown/NA"] + intents_order)
    ax.set_xlim(1, actual_len)
    ax.set_ylim(0, len(intents_order)+1)
    ax.set_xlabel("Step")
    ax.set_title("Customer Intent Timeline (dynamic)")
    ax.grid(True, axis="y", alpha=0.3)

    ax.set_xticks(list(range(1, actual_len+1)))
    ax.set_xticklabels([f"Step {i}" for i in range(1, actual_len+1)], rotation=0)

    (line,) = ax.step([], [], where="post")
    (pts,) = ax.plot([], [], "o", markersize=4)
    step_text = ax.text(0.99, 0.92, "", transform=ax.transAxes, ha="right", va="top")

    def update(frame):
        x = x_full[:frame+1]
        y = y_full[:frame+1]
        line.set_data(x, y)
        pts.set_data(x, y)
        step_text.set_text(f"Step {frame+1}/{actual_len}")
        return (line, pts, step_text)

    ani = FuncAnimation(fig, update, frames=actual_len, interval=int(1000/fps), blit=True)
    ani.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Build Sankey (per-session) and map clicks → session id (links & nodes)
# Left column intent now uses the *last* dynamic intent (current final at Max Steps)
# ──────────────────────────────────────────────────────────────────────────────
nodes = []; node_labels=[]; node_colors=[]; node_idx={}
node_idx_to_session = {}   # any per-session node → session
link_customdata = []       # link → session

# 索引：node → Counter(session_id → 累積權重)
from collections import Counter as Cnt
node_sessions_map = defaultdict(Cnt)   # for resolving node clicks robustly
link_session_map = []                  # link_idx → session_id (same as link_customdata)


def ensure_node(name, label, color, session_index=None):
    if name not in node_idx:
        node_idx[name] = len(nodes)
        nodes.append(name); node_labels.append(label); node_colors.append(color)
        if session_index is not None:
            node_idx_to_session[node_idx[name]] = session_index
    return node_idx[name]

# Shared nodes
for out in ['TP','TN','FP','FN']:
    ensure_node(f'OUT_{out}', out, outcome_color_map[out])
ensure_node('UTILITY', 'Utility', "#457b9d")

sources=[]; targets=[]; values=[]; colors=[]; link_labels=[]
left_label_prefix = "DYN_INTENT_LAST_" if use_dynamic_as_left_column else "GROUP_INTENT_"
def left_node_name(intent_label): return f"{left_label_prefix}{intent_label}"

for sidx, record in enumerate(records):
    seq = record['sequence']; shap_seq = record['shap']
    filtered_steps = [(e, s) for e, s in zip(seq, shap_seq) if e != 0]
    if len(filtered_steps) == 0: continue

    if use_dynamic_as_left_column:
        left_intent = get_last_intent(record, max_steps, session_mode)
    else:
        left_intent = record['static_intent'] or 'Unclassified'



    left_node = f"{left_node_name(left_intent)}_s{sidx}" if separate_mode else left_node_name(left_intent)
    ensure_node(left_node, left_intent, intent_color_map.get(left_intent, "#ffe066"), session_index=(sidx if separate_mode else None))

    prev = None
    for step_idx, (event, shap) in enumerate(filtered_steps):
        if separate_mode:
            curr_node = f'{sidx}_step{step_idx}'
            ensure_node(curr_node, '', event_color_map.get(event, '#cccccc'), session_index=sidx)
        else:
            curr_node = f'group_{step_idx}_{event}'
            ensure_node(curr_node, '', event_color_map.get(event, '#cccccc'))

        if step_idx == 0:
            sources.append(node_idx[left_node]); targets.append(node_idx[curr_node])
            values.append(1); colors.append(intent_flow_color); link_labels.append(''); link_customdata.append(sidx)

        if prev is not None:
            sources.append(node_idx[prev]); targets.append(node_idx[curr_node])
            values.append(abs(shap)+0.1); colors.append('#888888' if shap>=0 else '#ffb6c1'); link_labels.append(''); link_customdata.append(sidx)
        prev = curr_node

    outcome_node = f'OUT_{record["outcome"]}'
    if prev is not None:
        last_shap = filtered_steps[-1][1]
        sources.append(node_idx[prev]); targets.append(node_idx[outcome_node])
        values.append(abs(last_shap)+0.1); colors.append('#888888' if last_shap>=0 else '#ffb6c1'); link_labels.append(''); link_customdata.append(sidx)
        sources.append(node_idx[outcome_node]); targets.append(node_idx['UTILITY'])
        values.append(abs(record['utility'])); colors.append('#27ae60' if record['utility']>0 else ('#e63946' if record['utility']<0 else '#bfc0c0')); link_labels.append(''); link_customdata.append(sidx)

# Stage layout
max_click_len = max((len([e for e in r['sequence'] if e!=0]) for r in records), default=1)
stage_left=0; stage_click_1=1; stage_predict=max_click_len+1; stage_utility=max_click_len+2; num_stages=stage_utility+1

node_stage=[None]*len(nodes)
for nm, idx0 in list(node_idx.items()):
    if nm.startswith("GROUP_INTENT_") or nm.startswith("DYN_INTENT_LAST_"): node_stage[idx0]=stage_left
for nm, idx0 in list(node_idx.items()):
    if nm.startswith("group_"):
        parts = nm.split("_")
        if len(parts)>=3:
            try: node_stage[idx0]=stage_click_1+int(parts[1])
            except: pass
    elif "_step" in nm:
        try: node_stage[idx0]=stage_click_1+int(nm.split("_step")[-1])
        except: pass
for out_lbl in ['TP','TN','FP','FN']:
    nm=f'OUT_{out_lbl}'
    if nm in node_idx: node_stage[node_idx[nm]]=stage_predict
node_stage[node_idx['UTILITY']]=stage_utility

node_x=[None]*len(nodes); den=max(1,(num_stages-1))
for i in range(len(nodes)):
    s=node_stage[i] if node_stage[i] is not None else stage_click_1
    node_x[i]=s/den

stage_titles=[]
for s in range(num_stages):
    if s==stage_left: stage_titles.append("Intent")
    elif s==stage_predict: stage_titles.append("Prediction Result")
    elif s==stage_utility: stage_titles.append("Utility")
    else: stage_titles.append(f"Click {s}")

stage_header_annotations=[]
if show_stage_headers:
    for s, title in enumerate(stage_titles):
        stage_header_annotations.append(dict(x=s/max(1,(num_stages-1)), y=1.08, xref="paper", yref="paper", showarrow=False,
                                            text=f"<b>{title}</b>", font=dict(size=14, color="#22334b"),
                                            xanchor="center", yanchor="bottom", bgcolor="rgba(255,255,255,0.7)"))

total_utility = sum([r['utility'] for r in records])

# 在建立 fig 之前，加這段：
node_customdata = [node_idx_to_session.get(i, -1) for i in range(len(nodes))]

fig = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        pad=15, thickness=20, line=dict(color="black", width=0.5),
        label=[node_label_with_id(nm, lbl) for nm,lbl in zip(nodes,node_labels)],
        color=node_colors, x=node_x,
        customdata=node_customdata,          # ✅ 這裡換成 sidx 或 -1（非 session 節點）
    ),
    link=dict(
        source=sources, target=targets, value=values, color=colors, label=link_labels,
        customdata=link_customdata           # ✅ link 上也有 sidx（你原本就有）
    )
))



fig.update_layout(width=1400, height=700, margin=dict(l=20,r=200,t=70,b=40), font_size=15, font_family="Arial", paper_bgcolor="white", clickmode="event+select")
if show_stage_headers and stage_header_annotations:
    fig.update_layout(annotations=stage_header_annotations)

# Render Sankey with click capture
st.subheader("SankeyX")
clicked = None
if HAS_PLOTLY_EVENTS and separate_mode:
    clicked = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=700,
        override_width="100%",
    )
else:
    st.plotly_chart(fig, use_container_width=True, height=700)

# ───────────────────────────────────────────────
# Robust: determine sidx from link or node click
# ───────────────────────────────────────────────
def resolve_clicked_session_idx(ev_dict):
    """
    支援三種來源：
    1) 點「連結」：ev['customdata'] 直接是 sidx（我們在 link.customdata 填了）
    2) 點「節點」：用 pointNumber 找到 node 名稱，再由：
       - node_idx_to_session 對照
       - 或 node 名稱規則解析：'<intent>_s{sidx}' / '{sidx}_step{k}'
    3) 取不到 → 回傳 None
    """
    if not ev_dict:
        return None

    # case 1: link with customdata
    sidx = ev_dict.get("customdata", None)
    if isinstance(sidx, int):
        return sidx

    # case 2: node click → use pointNumber → node index → name
    pn = ev_dict.get("pointNumber")
    if pn is None:
        pn = ev_dict.get("pointIndex")  # 某些環境是 pointIndex
    try:
        pn = int(pn) if pn is not None else None
    except Exception:
        pn = None

    if pn is not None and 0 <= pn < len(nodes):
        # 先用 node_idx_to_session 對照（我們建立 per-session node 時有放）
        if pn in node_idx_to_session:
            return node_idx_to_session[pn]

        # 再嘗試從節點名稱解析 sidx
        nm = nodes[pn]
        # 例如：'DYN_INTENT_LAST_Engaged Buyer_s12'
        m = re.search(r'_s(\d+)$', nm)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
        # 例如：'12_step3'
        m2 = re.match(r'^(\d+)_step\d+$', nm)
        if m2:
            try:
                return int(m2.group(1))
            except Exception:
                pass

    return None

# ───────────────────────────────────────────────
# Show dynamic intent GIF when a session is clicked
# ───────────────────────────────────────────────
# ───────────────────────────────────────────────
# Show dynamic intent GIF when a session is clicked
# ───────────────────────────────────────────────
def _to_int(x):
    try:
        return int(x)
    except Exception:
        return None

if clicked and len(clicked) > 0:
    ev = clicked[0]

    # 1) 優先從 customdata 直接拿 sidx（node/link 都會有，如果你前面有設定）
    sidx = _to_int(ev.get("customdata"))

    # 2) 如果拿不到，再用 pointNumber 回推 link 索引，但減一修正偏移
    if sidx is None or sidx < 0:
        pn = _to_int(ev.get("pointNumber")) or _to_int(ev.get("pointIndex"))
        if pn is not None:
            # 重點：這裡 -1；再做邊界裁切避免越界
            pn = max(0, min(len(link_customdata) - 1, pn - 1))
            sidx = _to_int(link_customdata[pn]) if 0 <= pn < len(link_customdata) else None

    if sidx is None or not (0 <= sidx < len(records)):
        st.info("請點擊某條 **session flow 的連結**（Intent→Click 或 Click→Click 的灰/粉色線）。")
    else:
        st.markdown(f"### Customer Intent Dynamics for Session {records[sidx]['session_id_hash'][:8]}")

        base_dir = r"C:\Users\ycw"
        os.makedirs(base_dir, exist_ok=True)
        save_path = os.path.join(base_dir, f"intent_timeline_session{sidx}_{uuid.uuid4().hex}.gif")

        try:
            generate_intent_gif(records[sidx]["intent_timeline"], max_steps, save_path)
            if os.path.exists(save_path):
                st.image(save_path, use_column_width=True)
                st.success(f"GIF saved at {save_path}")
            else:
                st.error(f"GIF 未生成或路徑不可寫：{save_path}")
        except Exception as e:
            st.exception(e)


# ──────────────────────────────────────────────────────────────────────────────
# Build LLM Context (filtered selection only)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────

# SHAP 欄位數供 prompt 敘述
N_SHAP = len(all_shap_cols)

# 總結數字
outcomes = [r['outcome'] for r in records]
counts = Counter(outcomes)
util_by_outcome = defaultdict(float)
for r in records:
    util_by_outcome[r['outcome']] += r['utility']

# Intent 分佈：若左欄使用動態 intent，就採用「最後動態 intent」，否則用 static intent
intent_counts = Counter([
    get_last_intent(r, max_steps, session_mode) if use_dynamic_as_left_column else r.get('static_intent', 'Unclassified')
    for r in records
])

summary_lines = [
    f"Total sessions: {len(records)}",
    f"Total utility: {total_utility:.2f}",
    "Outcomes:",
    f"  TP: {counts.get('TP', 0)} (utility {util_by_outcome.get('TP', 0):+.2f})",
    f"  FP: {counts.get('FP', 0)} (utility {util_by_outcome.get('FP', 0):+.2f})",
    f"  FN: {counts.get('FN', 0)} (utility {util_by_outcome.get('FN', 0):+.2f})",
    f"  TN: {counts.get('TN', 0)} (utility {util_by_outcome.get('TN', 0):+.2f})",
    "Intent distribution: " + ", ".join([f"{k}: {v}" for k, v in intent_counts.most_common()]) if intent_counts else "Intent distribution: (none)"
]
compact_summary = "\n".join(summary_lines)

# 如需提供原始 JSON（你的 UI 目前只有 Compact summary，預設不會用到這段）
raw_payload_rows = []
for r in records:
    raw_payload_rows.append({
        "dynamic_intent_last": get_last_intent(r, max_steps, session_mode),
        "intent_timeline": r.get("intent_timeline_raw", []),
        "outcome": r.get("outcome", ""),
        "utility": r.get("utility", 0.0),
        "sequence": r.get("sequence", []),
        "sequence_text": [
            { "code": int(x), "name": EVENT_MAP.get(int(x), str(x)) }
            for x in r.get("sequence", [])
        ],
        "shap": r.get("shap", []),
    })


raw_json_context = json.dumps({
    "legend": {"1":"browse","2":"detail","3":"add_to_cart","4":"remove_from_cart","5":"purchase"},
    "note": f"SHAP_1..SHAP_{N_SHAP} correspond to the first N={N_SHAP} steps; max_steps controls which steps are visualized.",
    "data": raw_payload_rows
}, ensure_ascii=False)

# 目前僅提供 Compact summary，固定給 compact_summary
llm_context = compact_summary
# 未來若加 "Raw JSON" 選項，可改為：
# llm_context = raw_json_context if send_mode.startswith("Raw JSON") else compact_summary
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Fixed Expert Prompt (JSON output, emphasize TP & FN)
# ──────────────────────────────────────────────────────────────────────────────
FIXED_PROMPT = f"""
You are an expert in e-commerce analytics and explainable AI. 
You are given clickstream session data, including:
- Customer intent category
- Clickstream sequence (numeric code where 1=browse, 2=detail, 3=add_to_cart, 4=remove_from_cart, 5=purchase)
- Model prediction probability and predicted label
- SHAP values for the first N steps (explaining each step's contribution to the prediction) — here N = {N_SHAP}
- Actual purchase outcome and derived utility
- Sankey diagram (not provided as an image here, but it matches the given data: 
  columns are steps, flow width represents session counts, and color indicates SHAP value sign/magnitude)

**Your tasks (focus more on purchase-related cases: True Positives and False Negatives):**
1. Summarize overall performance:
   - TP/FP/FN/TN counts and total utility
   - Model strengths and weaknesses based on the given data
2. Identify dominant customer behavior patterns:
   - Common clickstream sequences for high-utility outcomes
   - Sequences that frequently lead to false negatives or false positives
3. Interpret SHAP values:
   - Which actions (e.g., add_to_cart, detail) have the strongest positive or negative impact
   - How these relate to the Sankey diagram's colors and thickness
4. Explain how to read the Sankey diagram for this dataset
   - How sequence order, SHAP color, and flow width convey insight
5. Provide 2–3 actionable insights and action recommended:
   - Recommendations for improving prediction performance
   - Business strategies for high-utility patterns
6. Explain the customer intent with prediction results insight:
   - Which intent drives purchase
 
Make your explanation concise, structured, and supported by specific numbers from the data.

**Output format (IMPORTANT):**
Return a single JSON object inside a fenced code block with the following keys:
- "overall_performance": string
- "strengths": string
- "weaknesses": string
- "patterns": string
- "shap": string
- "sankey": string
- "insights": array of 2-5 short strings
- "intents": array of 2-5 short strings

Use explicit numbers found in the context whenever possible.
Example:
```json
{{"overall_performance":"...", "strengths":"...", "weaknesses":"...", "patterns":"...", "shap":"...", "sankey":"...", "insights":["...","..."], "Intents":["...","..."]}}
```
"""

final_prompt = f"""{FIXED_PROMPT}



=== CONTEXT (filtered selection) ===
{llm_context}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Generate Explanation — Tabs with paragraph blocks
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("AI Explanation (Tabs + Paragraph Blocks)")
st.caption("The model returns JSON; we split each section into paragraphs and render blocks.")

run_llm = st.button("Generate AI Explanation") or auto_generate
if run_llm:
    with st.spinner("Calling local LLM via Ollama..."):
        response = call_ollama(final_prompt, model=llm_model, temperature=llm_temperature)
    if not response:
        st.info("No response generated. Ensure Ollama is running and the model name is correct.")
    else:
        parsed = extract_json_block(response)
        if parsed is None:
            st.warning("Could not parse JSON from the model output. Showing raw text instead.")
            st.markdown(response)
        else:
            tabs = st.tabs(["Overall Performance", "Strengths", "Weaknesses", "Patterns", "SHAP", "Sankey", "Insights", "Intents"])
            with tabs[0]:
                render_paragraphs(parsed.get("overall_performance",""))
            with tabs[1]:
                render_paragraphs(parsed.get("strengths",""))
            with tabs[2]:
                render_paragraphs(parsed.get("weaknesses",""))
            with tabs[3]:
                render_paragraphs(parsed.get("patterns",""))
            with tabs[4]:
                render_paragraphs(parsed.get("shap",""))
            with tabs[5]:
                render_paragraphs(parsed.get("sankey",""))
            with tabs[6]:
                insights = parsed.get("insights", [])
                if isinstance(insights, list) and insights:
                    for i, tip in enumerate(insights, 1):
                        st.markdown(f"- **Insight {i}.** {tip}")
                else:
                    st.markdown("_No insights returned._")
            with tabs[7]:
                intents = parsed.get("intents", [])
                if isinstance(intents, list) and intents:
                    for i, tip in enumerate(intents, 1):
                        st.markdown(f"- **intents {i}.** {tip}")
                else:
                    st.markdown("_No intents returned._")