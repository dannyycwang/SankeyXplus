# Auto-extracted from legacy file on 2025-09-02



def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return val



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



def sequence_to_text(seq):
    return [EVENT_MAP.get(int(x), str(x)) for x in seq if x != 0]



def sort_shap_cols(cols):
    def key_fn(c):
        try:
            return int(c.split('_')[1])
        except:
            return 10**6
    return sorted(cols, key=key_fn)



def normalize_tokens(num_seq):
    tokens = []
    for x in num_seq:
        if x == 0:
            continue
        t = EVENT_MAP.get(int(x), str(x))
        tokens.append(t)
    return tokens



def eval_rules_on_prefix(prefix_tokens):
    return {name: func(prefix_tokens) for name, func in rules.items()}



def _to_int(x):
    try:
        return int(x)
    except Exception:
        return None

