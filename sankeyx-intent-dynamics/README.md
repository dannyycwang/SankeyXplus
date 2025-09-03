# SankeyX — Intent Dynamics (Streamlit)

An interactive Streamlit app to visualize clickstream sessions as a **Sankey** flow with **dynamic customer intent timelines**, SHAP-weighted edges, prediction outcomes, and utility impact.

> This repo is prepped for a **public GitHub page** via `docs/` and includes deployment instructions (Streamlit Community, Hugging Face Spaces, or local).

---

## ✨ Features

- **SankeyX**: sessions rendered left→right as steps (intent → clicks → prediction → utility).
- **Dynamic Intent**: per-session intent timeline with **late** or **hysteresis** (inertia K) strategy.
- **Click-to-GIF**: click a session flow to auto-generate a GIF of its intent timeline.
- **SHAP integration**: per-step edge weight shows positive/negative contributions.
- **Utility accounting**: TP/TN/FP/FN mapped to configurable utility.
- **LLM summary (optional)**: local **Ollama** prompt for compact insights.

> App entry: `SankeyX_with_intent_dynamics.py`

---

## 📦 Quickstart (Local)

```bash
# 1) Create environment (recommended)
python -m venv .venv
source .venv/bin/activate    # (Windows: .venv\Scripts\activate)

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) start local Ollama for LLM
#    https://ollama.ai/download
#    then pull a model, e.g.:
#    ollama run mistral

# 4) Run
streamlit run SankeyX_with_intent_dynamics.py
```

Then open the local URL printed by Streamlit and **upload a CSV** (see schema below).

---

## 🔢 CSV Schema

Minimum required columns:

- `truncated_sequence` — list-like string of ints, e.g. `"[1,1,2,3,5]"` (0s are ignored)
- `purchase` — ground truth label (0/1)
- `y_pred` — predicted label (0/1)

Optional but recommended:

- `session_id_hash` — unique id per session
- `Intent_type` — static intent (fallback or comparison)
- `SHAP_1, SHAP_2, ...` — per-step SHAP values (aligned to earliest steps)
- Any other metadata you want to explore

A tiny example is provided at `sample_data/sample.csv`.

---

## 🧠 Dynamic Intent — Notes

- Strategies: **late** (greedy last true) and **hys** (late + inertia `K` consecutive confirmations).
- **Bridge NA** option: carry the last known intent across NA segments for a smoother timeline.
- The **leftmost column** can show either the **final dynamic intent** or the provided **static** intent.

---

## 🤖 LLM (Ollama) — Optional

The app can call a local LLM via **Ollama** (`http://localhost:11434`). In the sidebar, set:
- **Model**: `mistral`, `llama3.2`, `qwen2.5`, etc.
- **Temperature** and whether to auto-generate explanations.

If Ollama is **not running**, the app will simply show a friendly error and continue.

---

## 🌐 Deploy

### Option A — Streamlit Community Cloud
1. Push this repo to GitHub (public).
2. Go to https://share.streamlit.io/
3. Point to: `SankeyX_with_intent_dynamics.py`
4. Add secrets or hardware as needed (LLM optional).

### Option B — Hugging Face Spaces
1. Create a **Streamlit** Space.
2. Upload this repo.
3. Set `app_file` to `SankeyX_with_intent_dynamics.py`.
4. (Optional) Add Ollama on a separate machine and let the app call it.

### Option C — Local
Use `streamlit run ...` as shown above.

---

## 🧬 Requirements

See [`requirements.txt`](requirements.txt).

- Python ≥ 3.9 is recommended.
- If you want clickable interactions inside Streamlit, install `streamlit-plotly-events` (optional; already included).

---

## 📄 GitHub Pages

This repo includes a minimal **Docs site** under `docs/` for GitHub Pages. After pushing to GitHub:

1. Go to **Settings → Pages**.
2. **Build and deployment** → Source: **Deploy from a branch**.
3. Branch: `main` (or `master`), Folder: `/docs`.
4. Save. Your project page will be published at `https://<user>.github.io/<repo>/`.

The site contains a short project overview and a link back to the repo. You can further customize `docs/index.md`.

---

## 🧪 Sample Data

A minimal CSV is in `sample_data/sample.csv` to sanity-check rendering.

---

## ❗ Troubleshooting

- **No CSV / wrong columns**: The app will stop early and show a message.
- **GIF not saved**: Check the output path in the app log (Windows paths supported).
- **Ollama error**: Ensure `ollama serve` is available and your chosen model is pulled.
- **Plotly click events**: Requires `streamlit-plotly-events`. If unavailable, the app still renders (without click capture).

---

## 📚 Citation

If you use this tool in research, please cite:
> Y.-C. Wang, *SankeyX — Intent Dynamics: An Interactive Streamlit Visualization for Clickstream Prediction and Explainability*, 2025.  

---

## 🛡️ License

MIT — see [`LICENSE`](LICENSE).

---

_Last updated: 2025-09-02_


---


# 🧩 Modularization Guide

This repository now includes a **modular package** `sankeyx/` alongside the original monolithic app.

## Structure
```text
sankeyx-intent-dynamics/
├─ app.py                         # NEW: modular Streamlit entry (falls back to legacy when needed)
├─ SankeyX_with_intent_dynamics.py# legacy monolithic app (kept intact)
├─ sankeyx/
│  ├─ __init__.py
│  ├─ config.py                   # global settings (strategy, inertia K, utility matrix, etc.)
│  ├─ data_io.py                  # CSV load & schema validation
│  ├─ intent.py                   # dynamic intent timeline logic (late/hys, bridge NA) — stub
│  ├─ plot.py                     # plotly sankey construction — minimal demo
│  ├─ gif.py                      # timeline GIF exporter (matplotlib + Pillow)
│  └─ utils.py                    # helpers (e.g., parsing sequences)
├─ sample_data/
├─ docs/
├─ requirements.txt
└─ README.md
```

## Running (Modular app)
```bash
pip install -r requirements.txt
streamlit run app.py
```

If the modular path hits an error (until you fully migrate logic), it will **automatically fall back** to the legacy app.

## How to migrate the logic
- Move your real rule-evaluation + hysteresis + bridge-NA code into `sankeyx/intent.py` (replace the current placeholder).
- Refactor any inline CSV parsing/validation into `sankeyx/data_io.py`.
- Extract your Plotly Sankey building code into `sankeyx/plot.py`. Keep signatures clean, e.g. `build_sankey(nodes, links)`.
- Put GIF animation code into `sankeyx/gif.py`, wrapping it with a simple `export_timeline_gif(...)` API.
- If you use LLM summaries, move that logic into `sankeyx/llm.py` (Ollama helper provided).
- Keep Streamlit-specific UI in `app.py` (sidebar controls, file uploader, etc.).

## Minimal API examples

```python
# intent.py
timeline = compute_intent_timeline(sequence=[1,1,2,3,5], settings=DEFAULT_SETTINGS)

# plot.py
fig = build_sankey(nodes=["Intent","Click 1","Prediction","Utility"],
                   links={"source":[0],"target":[1],"value":[42]})

# gif.py
export_timeline_gif(timeline, "out.gif", fps=6)
```

## Notes
- `app.py` first tries the new modular flow on the uploaded CSV. If anything is missing, it will **gracefully fall back** to `SankeyX_with_intent_dynamics.py` so you never lose functionality.
- You can gradually move pieces into modules without breaking the app.
- Once migration is complete, you may remove the legacy file and the fallback branch.


---

## 📚 API Reference (Auto-extracted)

- `safe_eval()` → **other**
- `call_ollama()` → **llm**
- `extract_json_block()` → **data_io**
- `render_paragraphs()` → **other**
- `get_last_intent()` → **intent**
- `node_label_with_id()` → **intent**
- `sequence_to_text()` → **other**
- `sort_shap_cols()` → **other**
- `normalize_tokens()` → **other**
- `check_intent1()` → **intent**
- `check_intent2()` → **intent**
- `check_intent3()` → **intent**
- `check_intent4()` → **intent**
- `check_intent5()` → **intent**
- `check_intent6()` → **intent**
- `eval_rules_on_prefix()` → **other**
- `intent_timeline_for_trace()` → **intent**
- `generate_intent_gif()` → **intent**
- `ensure_node()` → **plot**
- `left_node_name()` → **intent**
- `resolve_clicked_session_idx()` → **intent**
- `_to_int()` → **other**
