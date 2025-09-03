
from typing import List, Optional
from .config import INTENT_LABELS, AppSettings

def compute_intent_timeline(sequence: List[int], settings: AppSettings) -> List[Optional[str]]:
    """Placeholder logic: returns a simple mock timeline based on step index.
    Replace this with your real rule-based evaluation and hysteresis/bridge logic.
    """
    if not sequence:
        return []
    # Very simple demo: cycle through a subset of intents by index
    intents = ["Hesitant", "Exploratory", "Engaged", "Comparative", "Uncertain"]
    out = []
    last = None
    for i, _ in enumerate(sequence):
        cand = intents[i % len(intents)]
        if settings.strategy == "hys":
            # trivialized inertia: require two repeats before switching
            if last is None or cand == last:
                out.append(cand)
                last = cand
            else:
                out.append(last)
        else:
            out.append(cand)
            last = cand
    # (Optional) Bridge NA would go here if you emit None for NA
    return out


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



def left_node_name(intent_label): return f"{left_label_prefix}{intent_label}"



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

