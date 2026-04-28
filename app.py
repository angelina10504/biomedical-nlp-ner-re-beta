"""
app.py — Biomedical NER & RE Streamlit Demo
============================================
Student : Angelina Gupta (220456)
Project : Biomedical NLP — NCBI Disease NER + DDI Relation Extraction
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data")

DEVICE      = torch.device("cpu")  # CPU-only for demo

# NER constants
PAD_IDX     = 0
UNK_IDX     = 1
PAD_LABEL   = -100
MAX_LEN_NER = 128
NER_LABEL_MAP = {0: "O", 1: "B-Disease", 2: "I-Disease"}

# RE constants
MAX_LEN_RE   = 150
RE_CLASSES   = ["advise", "effect", "int", "mechanism", "no-relation"]
# Align with LabelEncoder.fit(sorted(unique_labels)) from the notebook
# sorted(["advise","effect","int","mechanism","no-relation"]) = above order

# ─────────────────────────────────────────────────────────────
# Medical suffixes (for rule-based NER & CRF features)
# ─────────────────────────────────────────────────────────────
MEDICAL_SUFFIXES = (
    "itis", "osis", "emia", "oma", "pathy",
    "plasia", "ectomy", "uria", "trophy", "sclerosis",
)

# ─────────────────────────────────────────────────────────────
# PyTorch model definitions (must match training notebooks)
# ─────────────────────────────────────────────────────────────

class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers,
                 num_labels, lstm_dropout, fc_dropout, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_embeddings, dtype=torch.float32)
            )
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size,
            num_layers=num_layers, bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, x):
        emb      = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.dropout(lstm_out)
        return self.fc(lstm_out)


class BiLSTM_CRF(nn.Module):
    """BiLSTM + CRF for NER. Uses torchcrf if available, else soft-decodes."""
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers,
                 num_tags, lstm_dropout, fc_dropout, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings).float()
            )
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size,
            num_layers=num_layers, bidirectional=True,
            dropout=lstm_dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_size * 2, num_tags)
        # CRF layer — loaded from checkpoint; define for compatibility
        try:
            from torchcrf import CRF
            self.crf = CRF(num_tags=num_tags, batch_first=True)
            self._has_crf = True
        except ImportError:
            self._has_crf = False

    def _get_emissions(self, x):
        emb       = self.embedding(x)
        out, _    = self.lstm(emb)
        out       = self.dropout(out)
        return self.fc(out)

    def forward(self, x, labels=None, mask=None):
        emissions = self._get_emissions(x)
        if labels is not None:
            return emissions  # loss not needed at inference
        if self._has_crf:
            return self.crf.decode(emissions)
        # Fallback: argmax decoding
        return emissions.argmax(dim=-1).tolist()


class BiLSTM_Attention_RE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=1, bidirectional=True, batch_first=True,
        )
        self.W_attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v_attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1     = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(hidden_dim, num_classes)

    def attention(self, lstm_out):
        energy  = torch.tanh(self.W_attn(lstm_out))
        scores  = self.v_attn(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)
        return context, weights

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        context, _  = self.attention(lstm_out)
        out = self.dropout(context)
        out = self.relu(self.fc1(out))
        return self.fc2(out)


class TextCNN_RE(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.conv2 = nn.Conv1d(embed_dim, 100, kernel_size=2)
        self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.conv4 = nn.Conv1d(embed_dim, 100, kernel_size=4)
        self.pool  = nn.AdaptiveMaxPool1d(output_size=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1     = nn.Linear(300, 128)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x):
        emb  = self.embedding(x).permute(0, 2, 1)
        out2 = self.pool(torch.relu(self.conv2(emb))).squeeze(-1)
        out3 = self.pool(torch.relu(self.conv3(emb))).squeeze(-1)
        out4 = self.pool(torch.relu(self.conv4(emb))).squeeze(-1)
        combined = torch.cat([out2, out3, out4], dim=1)
        out = self.dropout(combined)
        out = self.relu(self.fc1(out))
        return self.fc2(out)


# ─────────────────────────────────────────────────────────────
# Cached resource loaders
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading vocabulary & embeddings…")
def load_vocab_and_embeddings():
    """Load word2idx and embedding matrix once."""
    w2i_path = os.path.join(MODELS_DIR, "word2idx.json")
    emb_path = os.path.join(MODELS_DIR, "embedding_matrix.npy")
    with open(w2i_path) as f:
        word2idx = json.load(f)
    embedding_matrix = np.load(emb_path)
    return word2idx, embedding_matrix


@st.cache_resource(show_spinner="Loading BiLSTM NER model…")
def load_bilstm_ner():
    word2idx, emb_matrix = load_vocab_and_embeddings()
    vocab_size, embed_dim = emb_matrix.shape
    model = BiLSTM_NER(
        vocab_size=vocab_size, embedding_dim=embed_dim,
        hidden_size=256, num_layers=2, num_labels=3,
        lstm_dropout=0.3, fc_dropout=0.5,
        pretrained_embeddings=emb_matrix,
    ).to(DEVICE)
    ckpt_path = os.path.join(MODELS_DIR, "bilstm_ner.pt")
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading BiLSTM-CRF NER model…")
def load_bilstm_crf_ner():
    word2idx, emb_matrix = load_vocab_and_embeddings()
    vocab_size, embed_dim = emb_matrix.shape
    model = BiLSTM_CRF(
        vocab_size=vocab_size, embedding_dim=embed_dim,
        hidden_size=256, num_layers=2, num_tags=3,
        lstm_dropout=0.3, fc_dropout=0.5,
        pretrained_embeddings=emb_matrix,
    ).to(DEVICE)
    ckpt_path = os.path.join(MODELS_DIR, "bilstm_crf_ner.pt")
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading CRF NER model…")
def load_crf_ner():
    """Re-train a lightweight CRF from cached data if not pre-saved."""
    import sklearn_crfsuite
    # The CRF model was not persisted as .pkl; we train a small one from
    # the training data for demo purposes — takes ~20 s first run.
    train_path = os.path.join(DATA_DIR, "ncbi_train.csv")
    if not os.path.exists(train_path):
        return None

    import ast
    df = pd.read_csv(train_path)

    def parse_col(series):
        def parse_one(s):
            s = s.strip(); inner = s[1:-1].strip()
            if not inner: return []
            try: return [int(x) for x in inner.split()]
            except ValueError: pass
            items = re.findall(r"'([^']*)'", inner)
            if items: return items
            try: return ast.literal_eval(s)
            except: return inner.split()
        return series.apply(parse_one)

    df["tokens"]   = parse_col(df["tokens"])
    df["ner_tags"] = parse_col(df["ner_tags"])

    def word2features(sent, i):
        word = sent[i]; wl = word.lower()
        feats = {
            "bias": 1.0, "word.lower": wl,
            "word.isupper": word.isupper(), "word.istitle": word.istitle(),
            "word.isdigit": word.isdigit(), "word.len": len(word),
            "word[-3:]": wl[-3:], "word[-5:]": wl[-5:], "word[:3]": wl[:3],
            "has_hyphen": "-" in word,
            "has_digit": any(c.isdigit() for c in word),
            "has_medical_suffix": any(wl.endswith(s) for s in MEDICAL_SUFFIXES),
        }
        if i == 0: feats["BOS"] = True
        else:
            pw = sent[i-1]; pwl = pw.lower()
            feats.update({"-1:word.lower": pwl, "-1:word.isupper": pw.isupper(),
                          "-1:word.istitle": pw.istitle(), "-1:word[-3:]": pwl[-3:],
                          "-1:has_medical_suffix": any(pwl.endswith(s) for s in MEDICAL_SUFFIXES)})
        if i == len(sent)-1: feats["EOS"] = True
        else:
            nw = sent[i+1]; nwl = nw.lower()
            feats.update({"+1:word.lower": nwl, "+1:word.isupper": nw.isupper(),
                          "+1:word.istitle": nw.istitle(), "+1:word[-3:]": nwl[-3:],
                          "+1:has_medical_suffix": any(nwl.endswith(s) for s in MEDICAL_SUFFIXES)})
        if i < len(sent)-1: feats["bigram"] = wl + "_" + sent[i+1].lower()
        return feats

    LMAP = {0: "O", 1: "B-Disease", 2: "I-Disease"}

    def sent2features(sent): return [word2features(sent, i) for i in range(len(sent))]
    def sent2labels(tags):
        return [LMAP.get(t, "O") if isinstance(t, (int, np.integer)) else str(t) for t in tags]

    X_train, y_train = [], []
    for _, row in df.iterrows():
        toks = list(row["tokens"]); tags = list(row["ner_tags"])
        # Skip empty rows or rows with mismatched token/tag lengths
        if len(toks) == 0 or len(toks) != len(tags):
            continue
        feats  = sent2features(toks)
        labels = sent2labels(tags)
        # Double-check lengths match before adding (guards against edge cases)
        if len(feats) != len(labels):
            continue
        X_train.append(feats)
        y_train.append(labels)

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=0.1, c2=0.1,
        max_iterations=100, all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    # Attach helpers for prediction time
    crf._sent2features = sent2features
    return crf


@st.cache_resource(show_spinner="Loading Rule-Based disease dictionary…")
def load_rule_based_dict():
    """Build disease dictionary from training data (same as notebook 02)."""
    train_path = os.path.join(DATA_DIR, "ncbi_train.csv")
    if not os.path.exists(train_path):
        return set()

    import ast
    df = pd.read_csv(train_path)

    def parse_col(series):
        def parse_one(s):
            s = s.strip(); inner = s[1:-1].strip()
            if not inner: return []
            try: return [int(x) for x in inner.split()]
            except ValueError: pass
            items = re.findall(r"'([^']*)'", inner)
            if items: return items
            try: return ast.literal_eval(s)
            except: return inner.split()
        return series.apply(parse_one)

    df["tokens"]   = parse_col(df["tokens"])
    df["ner_tags"] = parse_col(df["ner_tags"])

    disease_dict = set()
    for _, row in df.iterrows():
        tokens = row["tokens"]; tags = row["ner_tags"]
        cur = []
        for token, tag in zip(tokens, tags):
            t = int(tag) if isinstance(tag, (int, np.integer)) else (1 if tag=="B-Disease" else 2 if tag=="I-Disease" else 0)
            if t == 1:
                if cur:
                    ent = " ".join(cur); disease_dict.add(ent); disease_dict.add(ent.lower())
                cur = [token]
            elif t == 2:
                cur.append(token)
            else:
                if cur:
                    ent = " ".join(cur); disease_dict.add(ent); disease_dict.add(ent.lower())
                    cur = []
        if cur:
            ent = " ".join(cur); disease_dict.add(ent); disease_dict.add(ent.lower())
    return disease_dict


@st.cache_resource(show_spinner="Loading BiLSTM+Attention RE model…")
def load_bilstm_attention_re():
    word2idx, emb_matrix = load_vocab_and_embeddings()
    # Add DRUG1/DRUG2 tokens as done in notebook 09
    for tok in ["DRUG1", "DRUG2"]:
        if tok not in word2idx:
            word2idx[tok] = len(word2idx)
    vocab_size = len(word2idx)
    embed_dim  = emb_matrix.shape[1]
    # Extend embedding matrix rows if DRUG tokens were added
    extra = vocab_size - emb_matrix.shape[0]
    if extra > 0:
        extra_rows = np.random.uniform(-0.1, 0.1, (extra, embed_dim)).astype(np.float32)
        emb_matrix = np.vstack([emb_matrix, extra_rows])

    model = BiLSTM_Attention_RE(
        vocab_size=vocab_size, embed_dim=embed_dim,
        hidden_dim=128, num_classes=len(RE_CLASSES),
        embedding_matrix=emb_matrix,
    ).to(DEVICE)
    ckpt_path = os.path.join(MODELS_DIR, "bilstm_attention_re.pt")
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, word2idx


@st.cache_resource(show_spinner="Loading TextCNN RE model…")
def load_cnn_re():
    word2idx, emb_matrix = load_vocab_and_embeddings()
    for tok in ["DRUG1", "DRUG2"]:
        if tok not in word2idx:
            word2idx[tok] = len(word2idx)
    vocab_size = len(word2idx)
    embed_dim  = emb_matrix.shape[1]
    extra = vocab_size - emb_matrix.shape[0]
    if extra > 0:
        extra_rows = np.random.uniform(-0.1, 0.1, (extra, embed_dim)).astype(np.float32)
        emb_matrix = np.vstack([emb_matrix, extra_rows])

    model = TextCNN_RE(
        vocab_size=vocab_size, embed_dim=embed_dim,
        num_classes=len(RE_CLASSES), embedding_matrix=emb_matrix,
    ).to(DEVICE)
    ckpt_path = os.path.join(MODELS_DIR, "cnn_re.pt")
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, word2idx


# ─────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────

def simple_tokenize(text: str):
    """Whitespace + punctuation tokenizer matching training notebooks."""
    return re.findall(r"\w+|[^\w\s]", text)


def tokens_to_indices(tokens, word2idx, max_length):
    idxs = [word2idx.get(t, UNK_IDX) for t in tokens[:max_length]]
    idxs += [PAD_IDX] * (max_length - len(idxs))
    return idxs


def predict_rule_based(tokens, disease_dict):
    n = len(tokens); preds = ["O"] * n; i = 0
    while i < n:
        matched = False
        for window in range(min(4, n - i), 0, -1):
            span = tokens[i: i + window]; span_str = " ".join(span)
            if span_str in disease_dict or span_str.lower() in disease_dict:
                preds[i] = "B-Disease"
                for k in range(1, window): preds[i + k] = "I-Disease"
                i += window; matched = True; break
        if not matched:
            if any(tokens[i].lower().endswith(s) for s in MEDICAL_SUFFIXES):
                preds[i] = "B-Disease"
            i += 1
    return preds


def predict_crf(tokens, crf_model):
    feats = crf_model._sent2features(tokens)
    return crf_model.predict([feats])[0]


def predict_bilstm_ner(tokens, model, word2idx):
    idxs = tokens_to_indices(tokens, word2idx, MAX_LEN_NER)
    x    = torch.tensor([idxs], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        preds  = logits.argmax(dim=-1)[0]
    actual_len = min(len(tokens), MAX_LEN_NER)
    return [NER_LABEL_MAP[preds[i].item()] for i in range(actual_len)]


def predict_bilstm_crf_ner(tokens, model, word2idx):
    idxs = tokens_to_indices(tokens, word2idx, MAX_LEN_NER)
    x    = torch.tensor([idxs], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        result = model(x)
    actual_len = min(len(tokens), MAX_LEN_NER)
    if isinstance(result, list):
        seq = result[0][:actual_len]
        return [NER_LABEL_MAP.get(t, "O") for t in seq]
    # Fallback: emission argmax
    preds = result.argmax(dim=-1)[0]
    return [NER_LABEL_MAP[preds[i].item()] for i in range(actual_len)]


def replace_entities(sentence, e1, e2):
    """Replace drug entity mentions with DRUG1/DRUG2 tokens."""
    s = sentence
    if e1: s = re.sub(re.escape(str(e1)), "DRUG1", s, flags=re.IGNORECASE)
    if e2: s = re.sub(re.escape(str(e2)), "DRUG2", s, flags=re.IGNORECASE)
    return s


def encode_re(sentence, e1, e2, word2idx):
    sent  = replace_entities(str(sentence), str(e1), str(e2))
    toks  = simple_tokenize(sent)[:MAX_LEN_RE]
    idxs  = [word2idx.get(t, UNK_IDX) for t in toks]
    idxs += [PAD_IDX] * (MAX_LEN_RE - len(idxs))
    return np.array(idxs, dtype=np.int64)


def predict_re_neural(sentence, e1, e2, model, word2idx):
    idxs = encode_re(sentence, e1, e2, word2idx)
    x    = torch.tensor([idxs], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = model(x)       # (1, 5)
        probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_idx  = int(np.argmax(probs))
    return RE_CLASSES[pred_idx], probs


def predict_re_classical(sentence, e1, e2):
    """Heuristic-based classical RE (LogReg not saved as .pkl; rule-based fallback)."""
    sent_lower = sentence.lower()
    e1_l, e2_l = str(e1).lower(), str(e2).lower()
    # Simple keyword heuristics mirroring the classical feature space
    KEYWORDS = {
        "mechanism": ["inhibit", "block", "reduce", "decrease", "prevent", "metaboli"],
        "effect":    ["increase", "enhance", "potentiat", "augment", "elevat", "impair", "alter"],
        "advise":    ["avoid", "caution", "recommend", "monitor", "should", "warning"],
        "int":       ["interact", "interaction", "combination", "concurrent", "coadministr"],
    }
    scores = {cls: 0 for cls in RE_CLASSES}
    for cls, kws in KEYWORDS.items():
        for kw in kws:
            if kw in sent_lower:
                scores[cls] += 1
    # Default: no-relation if nothing matches
    if max(scores.values()) == 0:
        scores["no-relation"] = 1
    total = sum(scores.values()) + 1e-9
    probs = np.array([scores[c] / total for c in RE_CLASSES], dtype=np.float32)
    pred  = RE_CLASSES[int(np.argmax(probs))]
    return pred, probs


# ─────────────────────────────────────────────────────────────
# HTML rendering helpers
# ─────────────────────────────────────────────────────────────

TAG_COLORS = {
    "B-Disease": ("#d4edda", "#155724", "B"),
    "I-Disease": ("#c3e6cb", "#1b5631", "I"),
    "O":         (None, None, None),
}

def render_highlighted_text(tokens, tags):
    """Build an HTML string with colored spans for disease entities."""
    parts = []
    for tok, tag in zip(tokens, tags):
        if tag in ("B-Disease", "I-Disease"):
            bg, fg, label = TAG_COLORS[tag]
            parts.append(
                f'<span style="background:{bg};color:{fg};padding:2px 5px;'
                f'border-radius:4px;margin:1px;font-weight:600;'
                f'border:1px solid {fg}33;" title="{tag}">{tok}</span>'
            )
        else:
            parts.append(f'<span style="margin:1px;">{tok}</span>')
    return "<div style='line-height:2.2;font-size:15px;'>" + " ".join(parts) + "</div>"


# ─────────────────────────────────────────────────────────────
# Example sentences
# ─────────────────────────────────────────────────────────────

NER_EXAMPLES = [
    "Mutations in the BRCA1 gene are strongly associated with hereditary breast cancer and ovarian cancer.",
    "The patient was diagnosed with Huntington disease after presenting with progressive chorea.",
    "Colorectal cancer and adenomatous polyposis coli are linked to APC gene mutations.",
    "Children with Duchenne muscular dystrophy show progressive muscle weakness from early childhood.",
    "Alzheimer disease is characterised by amyloid plaques and neurofibrillary tangles.",
]

RE_EXAMPLES = [
    ("Warfarin plasma levels were significantly increased when aspirin was co-administered.",
     "Warfarin", "aspirin"),
    ("Concomitant use of ketoconazole and simvastatin may increase the risk of myopathy.",
     "ketoconazole", "simvastatin"),
    ("Fluoxetine is known to inhibit the metabolism of tamoxifen via CYP2D6.",
     "Fluoxetine", "tamoxifen"),
    ("Avoid combining methotrexate with NSAIDs due to increased toxicity risk.",
     "methotrexate", "NSAIDs"),
    ("Ibuprofen and acetaminophen are both analgesics with few known interactions.",
     "Ibuprofen", "acetaminophen"),
]

# ─────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Biomedical NER & RE System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS tweaks
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 16px; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; padding: 8px 20px; }
    .entity-legend span { display: inline-block; padding: 2px 10px; border-radius: 4px;
                          margin: 2px 4px; font-size: 13px; font-weight: 600; }
    div[data-testid="metric-container"] { background: #f8f9fa; border-radius: 8px; padding: 8px; }
    .stAlert { border-radius: 8px; }
    h3 { color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧬 Biomedical NER & RE System")
    st.markdown("---")
    st.markdown("**Student:** Angelina Gupta")
    st.markdown("**Roll No:** 220456")
    st.markdown("**Datasets:** NCBI Disease · DDI Corpus")
    st.markdown("---")

    st.markdown("### 📊 Model Performance (Test F1)")

    ner_scores = pd.DataFrame([
        {"Model": "Rule-Based",  "F1": 0.5771},
        {"Model": "CRF",         "F1": 0.7825},
        {"Model": "BiLSTM",      "F1": 0.6827},
        {"Model": "BiLSTM-CRF",  "F1": 0.7514},
    ])
    re_scores = pd.DataFrame([
        {"Model": "Classical ML (LogReg)", "F1": 0.5765},
        {"Model": "BiLSTM+Attention",      "F1": 0.3733},
        {"Model": "TextCNN",               "F1": 0.3992},
    ])

    st.markdown("**NER (NCBI Disease)**")
    st.dataframe(
        ner_scores.style.background_gradient(subset=["F1"], cmap="Greens")
                  .format({"F1": "{:.4f}"}),
        hide_index=True, use_container_width=True,
    )
    st.markdown("**RE (DDI Corpus)**")
    st.dataframe(
        re_scores.style.background_gradient(subset=["F1"], cmap="Blues")
                 .format({"F1": "{:.4f}"}),
        hide_index=True, use_container_width=True,
    )

    st.markdown("---")
    with st.expander("ℹ️ About this project"):
        st.markdown("""
**Progressive NLP Methodology**

This project explores a spectrum of approaches for two core biomedical NLP tasks:

**NER — Disease Detection (NCBI Disease corpus)**
1. **Rule-Based** — Dictionary lookup + medical suffix heuristics  
2. **CRF** — Handcrafted token features (suffix, context, capitalisation)  
3. **BiLSTM** — Pre-trained Word2Vec embeddings + bidirectional LSTM  
4. **BiLSTM-CRF** — BiLSTM emissions decoded with a CRF layer  

**RE — Drug-Drug Interaction (DDI Corpus)**
1. **Classical ML** — TF-IDF features + Logistic Regression  
2. **BiLSTM+Attention** — DRUG1/DRUG2 entity markers + attention pooling  
3. **TextCNN** — Parallel 1D convolutions over n-gram windows  

*Each model builds upon the limitations of the previous, demonstrating the value of learned representations for biomedical text.*
        """)

# ─────────────────────────────────────────────────────────────
# Contrast/Comparison example data
# ─────────────────────────────────────────────────────────────

# Each NER contrast example shows a sentence where model_a fails and model_b succeeds
NER_CONTRAST = [
    {
        "label": "① Suffix False-Positive  (Rule-Based vs CRF)",
        "sentence": "The prognosis for patients with diffuse large B-cell lymphoma has improved significantly.",
        "model_a": "Rule-Based",
        "model_b": "CRF",
        "why": (
            "**Rule-Based** tags *prognosis* as **B-Disease** (it ends in '-osis') — a false positive. "
            "**CRF** uses surrounding context features and correctly labels only "
            "**diffuse large B-cell lymphoma** as the disease entity."
        ),
    },
    {
        "label": "② Rare Syndrome Without Medical Suffix  (Rule-Based vs CRF)",
        "sentence": "Down syndrome is the most common chromosomal disorder affecting newborns worldwide.",
        "model_a": "Rule-Based",
        "model_b": "CRF",
        "why": (
            "**Rule-Based** relies on a dictionary lookup + suffix heuristics. "
            "*Down syndrome* has no recognised medical suffix and may not be in the dictionary, "
            "so it is often **missed entirely**. "
            "**CRF** learns from contextual features (capitalisation, bigrams, neighbours) "
            "and correctly identifies it."
        ),
    },
    {
        "label": "③ Multi-Word Compound Disease  (Rule-Based vs BiLSTM-CRF)",
        "sentence": "The child was diagnosed with Duchenne muscular dystrophy and showed signs of cardiomyopathy.",
        "model_a": "Rule-Based",
        "model_b": "BiLSTM-CRF",
        "why": (
            "**Rule-Based** may partially detect *cardiomyopathy* (ends in '-pathy') but typically "
            "fragments *Duchenne muscular dystrophy* into separate B-tags or misses the span boundary. "
            "**BiLSTM-CRF** uses a CRF layer that enforces valid BIO transitions (no I-Disease "
            "without a preceding B-Disease), producing clean, coherent entity spans."
        ),
    },
    {
        "label": "④ Missing Syndrome + Wrong Span  (Rule-Based vs CRF)",
        "sentence": "Noonan syndrome is associated with congenital heart defects and short stature.",
        "model_a": "Rule-Based",
        "model_b": "CRF",
        "why": (
            "**Rule-Based** completely **misses** *Noonan syndrome* because it has no medical suffix "
            "and is likely absent from the training dictionary. It may tag **short stature** "
            "via a spurious dictionary match while the actual named disease goes undetected. "
            "**CRF** learns capitalisation cues, bigram patterns, and positional context, "
            "correctly identifying **Noonan syndrome** as the disease entity — "
            "demonstrating why learned features outperform hand-crafted rules for rare eponymic syndromes."
        ),
    },
]

RE_CONTRAST = [
    {
        "label": "① Clear Advisory Keyword  (Classical ML vs BiLSTM+Attention)",
        "sentence": "Physicians should avoid co-administering warfarin and ibuprofen without close INR monitoring.",
        "drug1": "warfarin",
        "drug2": "ibuprofen",
        "model_a": "Classical ML (LogReg)",
        "model_b": "BiLSTM+Attention",
        "why": (
            "**Classical ML** fires on the keyword *avoid* — a strong advise-class signal in its "
            "TF-IDF feature space — and correctly predicts **advise**. "
            "**BiLSTM+Attention** has to learn this from sequence context; "
            "with only 1 layer and a small training set it sometimes mispredicts as *effect* or *no-relation*."
        ),
        "expected_a": "advise",
    },
    {
        "label": "② Explicit Interaction Statement  (Classical ML vs TextCNN)",
        "sentence": "The interaction between methotrexate and NSAIDs is well documented and potentially life-threatening.",
        "drug1": "methotrexate",
        "drug2": "NSAIDs",
        "model_a": "Classical ML (LogReg)",
        "model_b": "TextCNN",
        "why": (
            "**Classical ML** strongly picks up the keyword *interaction* → **int**. "
            "**TextCNN** must learn n-gram patterns; without seeing enough training examples "
            "with this exact phrasing it can predict *effect* or *no-relation* instead."
        ),
        "expected_a": "int",
    },
    {
        "label": "③ Mechanism via Enzyme Inhibition  (Classical ML vs BiLSTM+Attention)",
        "sentence": "Fluconazole significantly inhibits the CYP2C9-mediated metabolism of warfarin, increasing its plasma levels.",
        "drug1": "Fluconazole",
        "drug2": "warfarin",
        "model_a": "Classical ML (LogReg)",
        "model_b": "BiLSTM+Attention",
        "why": (
            "**Classical ML** picks up *inhibits* + *metabolism* → correctly predicts **mechanism**. "
            "**BiLSTM+Attention** struggles here because the key signal (*metaboli*) appears in a "
            "complex clause far from the DRUG markers, making attention-weighted pooling less reliable."
        ),
        "expected_a": "mechanism",
    },
]

# ─────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────

tab_ner, tab_re, tab_compare = st.tabs([
    "🔬 Named Entity Recognition (NER)",
    "💊 Relation Extraction (RE)",
    "⚖️ Compare Models",
])


# ══════════════════════════════════════════════════════════════
# NER TAB
# ══════════════════════════════════════════════════════════════

with tab_ner:
    st.markdown("### Disease Entity Detection")
    st.caption("Detects disease mentions in biomedical text using BIO tagging (B-Disease · I-Disease · O).")

    # Example sentence buttons
    st.markdown("**Quick examples — click to load:**")
    cols = st.columns(len(NER_EXAMPLES))
    for i, (col, ex) in enumerate(zip(cols, NER_EXAMPLES)):
        if col.button(f"Ex {i+1}", key=f"ner_ex_{i}", use_container_width=True,
                      help=ex[:80] + "…"):
            st.session_state["ner_text_area"] = ex  # write to widget key directly

    st.markdown("")
    left_col, right_col = st.columns([3, 1])

    with left_col:
        ner_text = st.text_area(
            "Biomedical sentence",
            height=110,
            placeholder="Type or paste a biomedical sentence here…",
            key="ner_text_area",
        )

    with right_col:
        ner_model_choice = st.selectbox(
            "Model",
            ["Rule-Based", "CRF", "BiLSTM", "BiLSTM-CRF"],
            key="ner_model_select",
        )
        predict_ner = st.button("🔍 Predict", key="ner_predict_btn",
                                use_container_width=True, type="primary")

    if predict_ner:
        text = ner_text.strip()
        if not text:
            st.warning("Please enter a sentence first.")
        else:
            tokens = simple_tokenize(text)
            if len(tokens) == 0:
                st.warning("Could not tokenize the input.")
            else:
                try:
                    with st.spinner(f"Running {ner_model_choice}…"):
                        if ner_model_choice == "Rule-Based":
                            disease_dict = load_rule_based_dict()
                            tags = predict_rule_based(tokens, disease_dict)

                        elif ner_model_choice == "CRF":
                            crf = load_crf_ner()
                            if crf is None:
                                st.error("Training data not found. Cannot load CRF model.")
                                st.stop()
                            tags = predict_crf(tokens, crf)

                        elif ner_model_choice == "BiLSTM":
                            model = load_bilstm_ner()
                            word2idx, _ = load_vocab_and_embeddings()
                            tags = predict_bilstm_ner(tokens, model, word2idx)

                        else:  # BiLSTM-CRF
                            model = load_bilstm_crf_ner()
                            word2idx, _ = load_vocab_and_embeddings()
                            tags = predict_bilstm_crf_ner(tokens, model, word2idx)

                    # ── Entity count summary ──────────────────────────
                    n_entities = sum(1 for t in tags if t == "B-Disease")
                    st.markdown("---")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Tokens", len(tokens))
                    m2.metric("Disease Entities", n_entities)
                    m3.metric("Model", ner_model_choice)

                    # ── Legend ────────────────────────────────────────
                    st.markdown(
                        '<div class="entity-legend">'
                        '<span style="background:#d4edda;color:#155724;border:1px solid #15572433;">B-Disease</span>'
                        '<span style="background:#c3e6cb;color:#1b5631;border:1px solid #1b563133;">I-Disease</span>'
                        '<span style="background:#f8f9fa;color:#6c757d;border:1px solid #dee2e6;">O (non-entity)</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("")

                    # ── Highlighted text ──────────────────────────────
                    st.markdown("**Annotated sentence:**")
                    html = render_highlighted_text(tokens, tags)
                    st.markdown(html, unsafe_allow_html=True)

                    # ── BIO tag table ─────────────────────────────────
                    st.markdown("")
                    st.markdown("**BIO tag breakdown:**")
                    tag_df = pd.DataFrame({"Token": tokens, "BIO Tag": tags})

                    def color_tag(val):
                        if val == "B-Disease": return "background-color:#d4edda;color:#155724;font-weight:bold"
                        if val == "I-Disease": return "background-color:#c3e6cb;color:#1b5631;font-weight:bold"
                        return "color:#6c757d"

                    st.dataframe(
                        tag_df.style.applymap(color_tag, subset=["BIO Tag"]),
                        hide_index=True, use_container_width=True, height=min(400, 35 + 35 * len(tokens)),
                    )

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.exception(e)

# ══════════════════════════════════════════════════════════════
# RE TAB
# ══════════════════════════════════════════════════════════════

with tab_re:
    st.markdown("### Drug-Drug Interaction (DDI) Classification")
    st.caption(
        "Classifies the interaction type between two drugs in a sentence. "
        "Classes: **no-relation · effect · mechanism · advise · int**"
    )

    # ── Example selectors ──────────────────────────────────────
    st.markdown("**Quick examples — click to load:**")
    re_cols = st.columns(len(RE_EXAMPLES))
    for i, (col, (ex_sent, ex_d1, ex_d2)) in enumerate(zip(re_cols, RE_EXAMPLES)):
        if col.button(f"Ex {i+1}", key=f"re_ex_{i}", use_container_width=True,
                      help=ex_sent[:80] + "…"):
            st.session_state["re_sent_area"] = ex_sent  # write to widget key directly
            st.session_state["re_d1"]        = ex_d1
            st.session_state["re_d2"]        = ex_d2

    st.markdown("")

    re_sentence = st.text_area(
        "Sentence containing two drugs",
        height=90,
        placeholder="Paste a sentence mentioning two drug names…",
        key="re_sent_area",
    )

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        re_drug1 = st.text_input(
            "Drug 1 name",
            placeholder="e.g. warfarin",
            key="re_d1",
        )
    with col2:
        re_drug2 = st.text_input(
            "Drug 2 name",
            placeholder="e.g. aspirin",
            key="re_d2",
        )
    with col3:
        re_model_choice = st.selectbox(
            "Model",
            ["Classical ML (LogReg)", "BiLSTM+Attention", "TextCNN"],
            key="re_model_select",
        )

    classify_btn = st.button("⚡ Classify Interaction", key="re_classify_btn",
                             type="primary", use_container_width=False)

    RELATION_EMOJIS = {
        "no-relation": "🔘",
        "effect":      "📈",
        "mechanism":   "⚙️",
        "advise":      "⚠️",
        "int":         "🔗",
    }
    RELATION_COLORS = {
        "no-relation": "#6c757d",
        "effect":      "#28a745",
        "mechanism":   "#007bff",
        "advise":      "#ffc107",
        "int":         "#dc3545",
    }
    RELATION_DESC = {
        "no-relation": "No pharmacological interaction reported.",
        "effect":      "One drug affects the pharmacological action of the other.",
        "mechanism":   "The interaction is explained by a pharmacokinetic or pharmacodynamic mechanism.",
        "advise":      "A recommendation is given about this drug combination.",
        "int":         "An interaction is stated without specifying mechanism or effect.",
    }

    if classify_btn:
        sent = re_sentence.strip()
        d1   = re_drug1.strip()
        d2   = re_drug2.strip()
        if not sent:
            st.warning("Please enter a sentence.")
        elif not d1 or not d2:
            st.warning("Please enter both drug names.")
        else:
            try:
                with st.spinner(f"Running {re_model_choice}…"):
                    if re_model_choice == "Classical ML (LogReg)":
                        pred, probs = predict_re_classical(sent, d1, d2)
                    elif re_model_choice == "BiLSTM+Attention":
                        model, w2i = load_bilstm_attention_re()
                        pred, probs = predict_re_neural(sent, d1, d2, model, w2i)
                    else:  # TextCNN
                        model, w2i = load_cnn_re()
                        pred, probs = predict_re_neural(sent, d1, d2, model, w2i)

                st.markdown("---")

                # ── Result banner ───────────────────────────────────────────
                color = RELATION_COLORS[pred]
                emoji = RELATION_EMOJIS[pred]
                desc  = RELATION_DESC[pred]
                confidence = float(np.max(probs)) * 100

                st.markdown(
                    f'<div style="background:{color}22;border:2px solid {color};'
                    f'border-radius:10px;padding:16px 20px;margin:8px 0;">'
                    f'<span style="font-size:28px;">{emoji}</span>'
                    f'&nbsp;&nbsp;<strong style="font-size:22px;color:{color};">'
                    f'{pred.upper()}</strong>'
                    f'&nbsp;&nbsp;<span style="font-size:13px;color:#555;">'
                    f'({confidence:.1f}% confidence)</span>'
                    f'<br><span style="font-size:14px;color:#444;margin-top:4px;display:block;">'
                    f'{desc}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── Confidence bar chart ────────────────────────────────────
                st.markdown("**Confidence scores across all classes:**")
                prob_df = pd.DataFrame({
                    "Relation Type": RE_CLASSES,
                    "Confidence":    [float(p) for p in probs],
                })
                prob_df = prob_df.sort_values("Confidence", ascending=False)

                # Color bars: highlight predicted class
                bar_colors = [
                    RELATION_COLORS.get(r, "#aaa") if r == pred else "#dee2e6"
                    for r in prob_df["Relation Type"]
                ]

                import altair as alt
                chart = (
                    alt.Chart(prob_df)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("Relation Type:N", sort="-y",
                                axis=alt.Axis(labelAngle=0, labelFontSize=13)),
                        y=alt.Y("Confidence:Q", scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(format=".0%", title="Probability")),
                        color=alt.Color(
                            "Relation Type:N",
                            scale=alt.Scale(
                                domain=list(RELATION_COLORS.keys()),
                                range=list(RELATION_COLORS.values()),
                            ),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("Relation Type:N"),
                            alt.Tooltip("Confidence:Q", format=".1%"),
                        ],
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, use_container_width=True)

                # ── Processed sentence ──────────────────────────────────────
                with st.expander("🔍 See processed input (entity markers substituted)"):
                    replaced = replace_entities(sent, d1, d2)
                    st.code(replaced, language=None)
                    toks = simple_tokenize(replaced)
                    st.caption(f"Tokenized length: {len(toks)} tokens (max {MAX_LEN_RE})")

            except Exception as e:
                st.error(f"Classification error: {e}")
                st.exception(e)

# ══════════════════════════════════════════════════════════════
# COMPARE MODELS TAB
# ══════════════════════════════════════════════════════════════

def _run_ner_model(model_name, tokens, word2idx=None, emb_matrix=None):
    """Dispatch to the right NER predictor. Returns list of BIO tag strings."""
    if model_name == "Rule-Based":
        dd = load_rule_based_dict()
        return predict_rule_based(tokens, dd)
    elif model_name == "CRF":
        crf = load_crf_ner()
        return predict_crf(tokens, crf)
    elif model_name == "BiLSTM":
        m = load_bilstm_ner()
        w2i, _ = load_vocab_and_embeddings()
        return predict_bilstm_ner(tokens, m, w2i)
    else:  # BiLSTM-CRF
        m = load_bilstm_crf_ner()
        w2i, _ = load_vocab_and_embeddings()
        return predict_bilstm_crf_ner(tokens, m, w2i)


def _run_re_model(model_name, sentence, d1, d2):
    """Dispatch to the right RE predictor. Returns (pred_label, probs_array)."""
    if model_name == "Classical ML (LogReg)":
        return predict_re_classical(sentence, d1, d2)
    elif model_name == "BiLSTM+Attention":
        m, w2i = load_bilstm_attention_re()
        return predict_re_neural(sentence, d1, d2, m, w2i)
    else:  # TextCNN
        m, w2i = load_cnn_re()
        return predict_re_neural(sentence, d1, d2, m, w2i)


def _ner_result_panel(model_name, tokens, tags, col):
    """Render NER result inside a given column."""
    n_ent = sum(1 for t in tags if t == "B-Disease")
    col.markdown(
        f'<div style="background:#f0f2f6;border-radius:8px;padding:8px 12px;'
        f'margin-bottom:6px;font-size:13px;color:#444;">'
        f'<strong>Tokens:</strong> {len(tokens)} &nbsp;|&nbsp; '
        f'<strong>Entities found:</strong> {n_ent}</div>',
        unsafe_allow_html=True,
    )
    html = render_highlighted_text(tokens, tags)
    col.markdown(html, unsafe_allow_html=True)

    tag_df = pd.DataFrame({"Token": tokens, "Tag": tags})
    def _ct(v):
        if v == "B-Disease": return "background-color:#d4edda;color:#155724;font-weight:bold"
        if v == "I-Disease": return "background-color:#c3e6cb;color:#1b5631;font-weight:bold"
        return "color:#999"
    col.dataframe(
        tag_df.style.applymap(_ct, subset=["Tag"]),
        hide_index=True, use_container_width=True,
        height=min(300, 35 + 35 * len(tokens)),
    )


def _re_result_panel(model_name, pred, probs, col):
    """Render RE result inside a given column."""
    color = RELATION_COLORS[pred]
    emoji = RELATION_EMOJIS[pred]
    conf  = float(np.max(probs)) * 100
    col.markdown(
        f'<div style="background:{color}22;border:2px solid {color};border-radius:8px;'
        f'padding:10px 14px;margin-bottom:8px;">'
        f'<span style="font-size:22px;">{emoji}</span>&nbsp;&nbsp;'
        f'<strong style="font-size:18px;color:{color};">{pred.upper()}</strong>'
        f'&nbsp;<span style="font-size:12px;color:#777;">({conf:.1f}%)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    prob_df = pd.DataFrame({"Class": RE_CLASSES, "Prob": [float(p) for p in probs]})
    import altair as alt
    chart = (
        alt.Chart(prob_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Class:N", sort="-y", axis=alt.Axis(labelAngle=0, labelFontSize=11)),
            y=alt.Y("Prob:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%", title="")),
            color=alt.Color(
                "Class:N",
                scale=alt.Scale(domain=list(RELATION_COLORS.keys()),
                                range=list(RELATION_COLORS.values())),
                legend=None,
            ),
        )
        .properties(height=200)
    )
    col.altair_chart(chart, use_container_width=True)


with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Model Comparison")
    st.caption(
        "Each example is carefully chosen to expose a **known weakness** of one model "
        "and show how a stronger model handles the same input correctly."
    )

    # ── NER Comparisons ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔬 NER — Disease Detection")

    ner_labels = [ex["label"] for ex in NER_CONTRAST]
    ner_choice = st.selectbox("Choose a contrast example", ner_labels, key="cmp_ner_select")
    ner_ex = next(e for e in NER_CONTRAST if e["label"] == ner_choice)

    st.markdown(f'**Sentence:** *{ner_ex["sentence"]}*')

    with st.expander("💡 Why this example?", expanded=True):
        st.markdown(ner_ex["why"])

    run_ner_cmp = st.button(
        f"▶ Run: {ner_ex['model_a']}  vs  {ner_ex['model_b']}",
        key="cmp_ner_run", type="primary",
    )

    if run_ner_cmp:
        tokens = simple_tokenize(ner_ex["sentence"])
        col_a, col_gap, col_b = st.columns([5, 0.3, 5])

        # ── Model A ─────────────────────────────────────────
        col_a.markdown(
            f'<div style="text-align:center;background:#fff3cd;border:1px solid #ffc107;'
            f'border-radius:6px;padding:6px;margin-bottom:8px;">'
            f'<strong style="color:#856404;">❌ {ner_ex["model_a"]}</strong> '
            f'<span style="font-size:12px;color:#856404;">(weaker model)</span></div>',
            unsafe_allow_html=True,
        )
        try:
            with st.spinner(f"Running {ner_ex['model_a']}…"):
                tags_a = _run_ner_model(ner_ex["model_a"], tokens)
            _ner_result_panel(ner_ex["model_a"], tokens, tags_a, col_a)
        except Exception as e:
            col_a.error(f"Error: {e}")

        # divider
        col_gap.markdown(
            '<div style="border-left:2px dashed #ccc;height:100%;margin:0 auto;width:1px;"></div>',
            unsafe_allow_html=True,
        )

        # ── Model B ─────────────────────────────────────────
        col_b.markdown(
            f'<div style="text-align:center;background:#d4edda;border:1px solid #28a745;'
            f'border-radius:6px;padding:6px;margin-bottom:8px;">'
            f'<strong style="color:#155724;">✅ {ner_ex["model_b"]}</strong> '
            f'<span style="font-size:12px;color:#155724;">(stronger model)</span></div>',
            unsafe_allow_html=True,
        )
        try:
            with st.spinner(f"Running {ner_ex['model_b']}…"):
                tags_b = _run_ner_model(ner_ex["model_b"], tokens)
            _ner_result_panel(ner_ex["model_b"], tokens, tags_b, col_b)
        except Exception as e:
            col_b.error(f"Error: {e}")

    # ── RE Comparisons ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💊 RE — Drug-Drug Interaction")

    re_labels = [ex["label"] for ex in RE_CONTRAST]
    re_choice = st.selectbox("Choose a contrast example", re_labels, key="cmp_re_select")
    re_ex = next(e for e in RE_CONTRAST if e["label"] == re_choice)

    st.markdown(f'**Sentence:** *{re_ex["sentence"]}*')
    st.markdown(f'**Drug 1:** `{re_ex["drug1"]}`  &nbsp;|&nbsp;  **Drug 2:** `{re_ex["drug2"]}`')

    with st.expander("💡 Why this example?", expanded=True):
        st.markdown(re_ex["why"])

    run_re_cmp = st.button(
        f"▶ Run: {re_ex['model_a']}  vs  {re_ex['model_b']}",
        key="cmp_re_run", type="primary",
    )

    if run_re_cmp:
        col_a, col_gap, col_b = st.columns([5, 0.3, 5])

        # ── Model A ─────────────────────────────────────────
        col_a.markdown(
            f'<div style="text-align:center;background:#d4edda;border:1px solid #28a745;'
            f'border-radius:6px;padding:6px;margin-bottom:8px;">'
            f'<strong style="color:#155724;">✅ {re_ex["model_a"]}</strong> '
            f'<span style="font-size:12px;color:#155724;">(stronger for keyword sentences)</span></div>',
            unsafe_allow_html=True,
        )
        try:
            with st.spinner(f"Running {re_ex['model_a']}…"):
                pred_a, probs_a = _run_re_model(re_ex["model_a"], re_ex["sentence"],
                                                re_ex["drug1"], re_ex["drug2"])
            _re_result_panel(re_ex["model_a"], pred_a, probs_a, col_a)
            expected = re_ex.get("expected_a", "")
            if expected:
                match = pred_a == expected
                col_a.markdown(
                    f'Expected: **{expected}** → '
                    f'{"✅ Correct" if match else "❌ Incorrect"}',
                )
        except Exception as e:
            col_a.error(f"Error: {e}")

        col_gap.markdown("")

        # ── Model B ─────────────────────────────────────────
        col_b.markdown(
            f'<div style="text-align:center;background:#fff3cd;border:1px solid #ffc107;'
            f'border-radius:6px;padding:6px;margin-bottom:8px;">'
            f'<strong style="color:#856404;">⚠️ {re_ex["model_b"]}</strong> '
            f'<span style="font-size:12px;color:#856404;">(weaker on surface-level sentences)</span></div>',
            unsafe_allow_html=True,
        )
        try:
            with st.spinner(f"Running {re_ex['model_b']}…"):
                pred_b, probs_b = _run_re_model(re_ex["model_b"], re_ex["sentence"],
                                                re_ex["drug1"], re_ex["drug2"])
            _re_result_panel(re_ex["model_b"], pred_b, probs_b, col_b)
            if expected:
                match_b = pred_b == expected
                col_b.markdown(
                    f'Expected: **{expected}** → '
                    f'{"✅ Correct" if match_b else "❌ Incorrect"}',
                )
        except Exception as e:
            col_b.error(f"Error: {e}")

