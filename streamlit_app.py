"""
================================================================
  Recommendation System Using Neural Networks — Streamlit App
  Deep Learning Project
  Kartik Sehrawat | Reg. No: 241306124
  Dataset: MovieLens 100K
  Architecture: Neural Collaborative Filtering (NCF)
================================================================
"""

import os
import pickle
import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommendation System | Kartik Sehrawat",
    page_icon="RS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* Global */
.stApp { background: #f5f1eb; }
[data-testid="stSidebar"] { background: #fffdf8; border-right: 1px solid #e8e2d7; }
h1, h2, h3, h4 { font-family: 'DM Sans', sans-serif !important; color: #2d2a26 !important; }

/* Metric cards */
.metric-card {
    background: #fffdf8;
    border: 1px solid #e8e2d7;
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(45,42,38,.05);
    transition: transform .2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #c76a3d;
}
.metric-label { font-size: 0.78rem; color: #8c857b; margin-top: 2px; }

/* Student info */
.student-box {
    background: #fffdf8;
    border: 1px solid #e8e2d7;
    border-radius: 14px;
    padding: 1rem 1.3rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
}
.avatar {
    width: 48px; height: 48px;
    border-radius: 50%;
    background: linear-gradient(135deg, #c76a3d, #d4a24e);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; font-weight: 700; color: #fff;
}

/* Step card */
.step-card {
    background: #fffdf8;
    border: 1px solid #e8e2d7;
    border-radius: 14px;
    padding: 1.2rem;
    height: 100%;
}
.step-num {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: #c76a3d;
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.5rem;
}

/* Rec card */
.rec-card {
    background: #fffdf8;
    border: 1px solid #e8e2d7;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.rank-badge {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.82rem;
}
.rank-1 { background: rgba(212,162,78,.2); color: #d4a24e; }
.rank-2 { background: rgba(140,133,123,.12); color: #8c857b; }
.rank-3 { background: rgba(199,106,61,.12); color: #c76a3d; }
.rank-n { background: rgba(42,140,130,.1); color: #2a8c82; }

.score-bar {
    height: 5px;
    border-radius: 3px;
    background: #e8e2d7;
    overflow: hidden;
}
.score-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #2a8c82, #d4a24e);
}

/* Watched item */
.watched-item {
    padding: 0.3rem 0;
    font-size: 0.85rem;
    color: #2d2a26;
    border-bottom: 1px solid #f0ebe3;
}

/* Architecture node */
.arch-node {
    display: inline-block;
    border-radius: 8px;
    padding: 0.35rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.15rem;
}
.n-input  { border: 1.5px solid #7b5ea7; color: #7b5ea7; background: rgba(123,94,167,.06); }
.n-embed  { border: 1.5px solid #2a8c82; color: #2a8c82; background: rgba(42,140,130,.06); }
.n-op     { border: 1.5px solid #c76a3d; color: #c76a3d; background: rgba(199,106,61,.06); }
.n-output { border: 1.5px solid #d4a24e; color: #d4a24e; background: rgba(212,162,78,.08); }

.why-box {
    background: #fffdf8;
    border-left: 4px solid #2a8c82;
    border-radius: 0 14px 14px 0;
    padding: 1rem 1.3rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #5a554d;
}
</style>
""", unsafe_allow_html=True)


# ── Data Loading & Caching ───────────────────────────────────
@st.cache_resource
def load_model_and_data():
    """Load trained model, encodings, and movie data."""
    import requests

    # Download dataset if needed
    data_dir = "ml-100k"
    if not os.path.exists(data_dir):
        st.info("Downloading MovieLens 100K dataset...")
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        r = requests.get(url, timeout=60)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(".")

    # Load movies
    movies_df = pd.read_csv(
        os.path.join(data_dir, "u.item"),
        sep="|", encoding="latin-1",
        usecols=[0, 1], names=["item_id", "title"]
    )

    # Load model & encodings
    model_path = "model.keras"
    enc_path   = "encodings.pkl"

    if not os.path.exists(model_path) or not os.path.exists(enc_path):
        return None, None, movies_df

    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)

    with open(enc_path, "rb") as f:
        encodings = pickle.load(f)

    return model, encodings, movies_df


def get_recommendations(model, encodings, movies_df, user_id, n=10):
    """Generate top-N recommendations for a user."""
    user2idx = encodings["user2idx"]
    item2idx = encodings["item2idx"]
    n_items  = encodings["n_items"]
    user_item_set = encodings["user_item_set"]

    user_idx = user2idx[user_id]
    idx2item = {v: k for k, v in item2idx.items()}

    # Watched movies
    seen_idxs = [i for (u, i) in user_item_set if u == user_idx]
    watched = []
    for i in seen_idxs[:10]:
        orig = idx2item.get(i)
        row = movies_df[movies_df["item_id"] == orig]
        title = row["title"].values[0] if len(row) > 0 else f"Movie #{orig}"
        watched.append(title)

    # Predictions
    seen_set = set(seen_idxs)
    candidates = [i for i in range(n_items) if i not in seen_set]

    users_arr = np.array([user_idx] * len(candidates))
    items_arr = np.array(candidates)
    preds = model.predict([users_arr, items_arr], batch_size=512, verbose=0).flatten()

    top_indices = np.argsort(preds)[::-1][:n]
    recs = []
    for idx in top_indices:
        item_idx = candidates[idx]
        score = float(preds[idx])
        orig = idx2item.get(item_idx)
        row = movies_df[movies_df["item_id"] == orig]
        title = row["title"].values[0] if len(row) > 0 else f"Movie #{orig}"
        recs.append({"title": title, "score": round(score * 100, 1)})

    return watched, len(seen_idxs), recs


# ── Load Assets ──────────────────────────────────────────────
model, encodings, movies_df = load_model_and_data()
model_loaded = model is not None


# =====================================================
#  SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("""
    <div class="student-box">
        <div class="avatar">KS</div>
        <div>
            <div style="font-weight:700;font-size:0.95rem;">Kartik Sehrawat</div>
            <div style="font-size:0.75rem;color:#8c857b;">Reg. No: <b style="color:#c76a3d;">241306124</b></div>
            <div style="font-size:0.7rem;color:#8c857b;">Deep Learning & Neural Networks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### About This Project")
    st.markdown("""
    A **Neural Collaborative Filtering** model that learns from 100,000 movie ratings 
    to predict personalized recommendations. Built with TensorFlow/Keras.
    """)

    st.markdown("---")
    st.markdown("#### Tech Stack")
    st.code("""
Model:     NCF (NeuMF)
Framework: TensorFlow 2.x
Dataset:   MovieLens 100K
Training:  8 epochs
Loss:      Binary Cross-Entropy
Optimizer: Adam (lr=0.001)
Embedding: 32 dimensions
MLP:       64 → 32 → 16 → 8
    """, language="yaml")

    if model_loaded:
        st.markdown("---")
        st.markdown("#### Model Performance")
        hr   = encodings.get("hr", 0)
        ndcg = encodings.get("ndcg", 0)
        st.metric("Hit Rate @ 10", f"{hr*100:.1f}%")
        st.metric("NDCG @ 10", f"{ndcg:.4f}")


# =====================================================
#  MAIN CONTENT
# =====================================================

# ── Header ───────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:0.5rem;">
    <span style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#8c857b;">
    Deep Learning / Neural Networks / Project
    </span>
</div>
""", unsafe_allow_html=True)

st.title("Movie Recommendation System Using Neural Networks")
st.markdown("""
A deep learning model that learns user preferences from thousands of movie ratings 
and predicts what films a person will enjoy — using **neural network embeddings** 
instead of traditional similarity measures.
""")

st.markdown("---")


# ── Metrics Row ──────────────────────────────────────
if model_loaded:
    hr   = encodings.get("hr", 0)
    ndcg = encodings.get("ndcg", 0)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{hr*100:.1f}%</div><div class="metric-label">Hit Rate @ 10</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{ndcg:.4f}</div><div class="metric-label">NDCG @ 10</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card"><div class="metric-val">100K</div><div class="metric-label">MovieLens Ratings</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card"><div class="metric-val">943</div><div class="metric-label">Users in Dataset</div></div>""", unsafe_allow_html=True)

st.markdown("")


# ── Section 1: How It Works ──────────────────────────
st.header("How the Model Works")

c1, c2, c3, c4 = st.columns(4)
steps = [
    ("1", "Collect Data", "Load the **MovieLens 100K** dataset — 100K real ratings from 943 users on 1,682 movies."),
    ("2", "Create Embeddings", "Each user & movie gets a **vector of 32 numbers**. Similar users get similar vectors."),
    ("3", "Train the Network", "Train on positive (watched) & negative (not watched) pairs using **binary cross-entropy** loss."),
    ("4", "Predict & Rank", "Score all unseen movies, return the **Top-N highest** as personalized recommendations.")
]
for col, (num, title, desc) in zip([c1, c2, c3, c4], steps):
    with col:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">{num}</div>
            <h4 style="font-size:0.9rem;margin:0 0 0.3rem;">{title}</h4>
            <p style="font-size:0.78rem;color:#8c857b;line-height:1.6;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# Why Deep Learning
st.markdown("""
<div class="why-box">
    <b style="color:#2a8c82;">Why is this Deep Learning, not just Matrix Factorization?</b><br>
    Traditional recommenders use a simple <b>dot product</b> between vectors — it only captures linear patterns.
    Our model adds a <b>Multi-Layer Perceptron (MLP)</b> with 4 hidden layers (64→32→16→8 neurons)
    on top of the embeddings, allowing it to learn <b>complex, non-linear user-item interactions</b>.
</div>
""", unsafe_allow_html=True)

st.markdown("")


# ── Section 2: Architecture ──────────────────────────
st.header("NCF Model Architecture")

st.markdown("""
<div style="background:#fffdf8;border:1px solid #e8e2d7;border-radius:14px;padding:1.3rem 1.5rem;margin-bottom:1rem;">
    <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#8c857b;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.4rem;">GMF Branch — Generalized Matrix Factorization</div>
    <div style="text-align:center;margin-bottom:0.3rem;">
        <span class="arch-node n-input">User ID</span> →
        <span class="arch-node n-embed">User Emb (32)</span> ×
        <span class="arch-node n-embed">Item Emb (32)</span> →
        <span class="arch-node n-op">Element-wise Product</span>
    </div>
    <div style="text-align:center;color:#8c857b;font-size:0.8rem;margin:0.4rem 0;">↘ &nbsp;Concat&nbsp; ↙</div>
    <div style="text-align:center;margin-bottom:0.5rem;">
        <span class="arch-node n-output">NeuMF → Dense(1) → Sigmoid → Prediction (0-1)</span>
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#8c857b;text-transform:uppercase;letter-spacing:1px;margin:0.6rem 0 0.4rem;">MLP Branch — Multi-Layer Perceptron</div>
    <div style="text-align:center;">
        <span class="arch-node n-input">Item ID</span> →
        <span class="arch-node n-embed">User Emb (32)</span> ⊕
        <span class="arch-node n-embed">Item Emb (32)</span> →
        <span class="arch-node n-op">Dense 64→32→16→8</span> →
        <span class="arch-node n-op">MLP Output</span>
    </div>
    <p style="text-align:center;font-size:0.73rem;color:#8c857b;margin-top:0.8rem;font-style:italic;">
        Both branches concatenated into NeuMF → final Dense + Sigmoid. Based on: He et al., "Neural Collaborative Filtering" (WWW 2017)
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("")


# ── Section 3: Live Demo ─────────────────────────────
st.header("Live Demo — Try It Yourself")

if not model_loaded:
    st.error("⚠ Model not trained yet. Run `python recommendation_model.py` first, then restart.")
    st.stop()

col_ctrl, col_result = st.columns([1, 2.5])

with col_ctrl:
    st.markdown("##### Pick a User")
    st.caption("The dataset has 943 anonymous users. Each has a different watch history.")

    user_id = st.selectbox(
        "Select User",
        options=list(range(1, 51)) + [100, 200, 500, 943],
        format_func=lambda x: f"User #{x}",
        index=0,
        label_visibility="collapsed"
    )

    top_n = st.radio(
        "How many recommendations?",
        options=[5, 10, 15, 20],
        index=1,
        horizontal=True
    )

    run_btn = st.button("Generate Recommendations", use_container_width=True, type="primary")

with col_result:
    if run_btn:
        with st.spinner("Running neural network..."):
            watched, total_watched, recs = get_recommendations(model, encodings, movies_df, user_id, top_n)

        # User banner
        st.markdown(f"""
        <div style="background:#fffdf8;border:1px solid #e8e2d7;border-radius:14px;padding:0.8rem 1.2rem;margin-bottom:1rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:0.7rem;">
                <div class="avatar" style="width:38px;height:38px;font-size:0.85rem;">U{user_id}</div>
                <div>
                    <div style="font-weight:700;">User #{user_id}</div>
                    <div style="font-size:0.73rem;color:#8c857b;">Watched <b>{total_watched}</b> movies</div>
                </div>
            </div>
            <div style="font-size:0.75rem;color:#8c857b;font-style:italic;">Model scored all unwatched movies, showing top {top_n}</div>
        </div>
        """, unsafe_allow_html=True)

        # Two columns: watched + recommendations
        w_col, r_col = st.columns([1, 1.7])

        with w_col:
            st.markdown(f"##### Already Watched ({total_watched})")
            for movie in watched:
                st.markdown(f"""<div class="watched-item">● {movie}</div>""", unsafe_allow_html=True)
            if total_watched > 10:
                st.caption(f"...and {total_watched - 10} more movies")

        with r_col:
            st.markdown(f"##### Predicted Recommendations (Top {top_n})")
            for i, r in enumerate(recs):
                rank_cls = "rank-1" if i==0 else "rank-2" if i==1 else "rank-3" if i==2 else "rank-n"
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rank-badge {rank_cls}">{i+1}</div>
                    <div style="flex:1;">
                        <div style="font-weight:600;font-size:0.88rem;">{r['title']}</div>
                    </div>
                    <div style="width:80px;text-align:right;">
                        <div style="font-family:'Space Mono',monospace;font-size:0.78rem;font-weight:700;color:#2a8c82;">{r['score']}%</div>
                        <div class="score-bar"><div class="score-fill" style="width:{r['score']}%"></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#fffdf8;border:2px dashed #e8e2d7;border-radius:16px;padding:3rem 2rem;text-align:center;">
            <div style="font-size:2.2rem;margin-bottom:0.8rem;">🎬</div>
            <p style="color:#8c857b;font-size:0.9rem;">
                Select a user from the left panel and click<br><b style="color:#2d2a26;">Generate Recommendations</b>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.75rem;color:#8c857b;padding:0.5rem 0;">
    Deep Learning Project · <b style="color:#2d2a26;">Kartik Sehrawat</b> · 
    Reg. No: <b style="color:#c76a3d;">241306124</b> · 
    Neural Collaborative Filtering · MovieLens 100K
</div>
""", unsafe_allow_html=True)
