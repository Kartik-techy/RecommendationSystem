"""
======================================================
  Recommendation System Using Neural Networks (NCF)
  Deep Learning Subject Project
  Author: Kartik Sehrawat
  Dataset: MovieLens 100K
======================================================

Architecture: Neural Collaborative Filtering (NCF)
  - Generalized Matrix Factorization (GMF) branch
  - Multi-Layer Perceptron (MLP) branch
  - NeuMF = Concat(GMF, MLP) -> Sigmoid output
"""

import os
import pickle
import requests
import zipfile
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────
EMBEDDING_DIM   = 32          # Embedding size for GMF branch
MLP_LAYERS      = [64, 32, 16, 8]  # MLP hidden layer sizes
BATCH_SIZE      = 256
EPOCHS          = 20
LEARNING_RATE   = 0.001
NUM_NEGATIVES   = 4           # Negative samples per positive interaction
TOP_K           = 10          # Hit Rate / NDCG evaluation @K
SEED            = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────
#  1. DATA LOADING
# ─────────────────────────────────────────────────────────
def load_movielens():
    data_dir = "ml-100k"
    if not os.path.exists(data_dir):
        print("Downloading MovieLens 100K dataset...")
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        r = requests.get(url, timeout=60)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(".")
        print("Download complete.")

    df = pd.read_csv(
        os.path.join(data_dir, "u.data"),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    # Load movie titles
    movies = pd.read_csv(
        os.path.join(data_dir, "u.item"),
        sep="|",
        encoding="latin-1",
        usecols=[0, 1],
        names=["item_id", "title"]
    )
    return df, movies


def preprocess(df):
    """Encode user/item IDs to contiguous indices."""
    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_ids)}

    df = df.copy()
    df["user"]  = df["user_id"].map(user2idx)
    df["item"]  = df["item_id"].map(item2idx)
    df["label"] = 1  # All observed interactions are positive

    n_users = len(user2idx)
    n_items = len(item2idx)

    print(f"  Users: {n_users} | Items: {n_items} | Interactions: {len(df)}")
    return df, user2idx, item2idx, n_users, n_items


# ─────────────────────────────────────────────────────────
#  2. NEGATIVE SAMPLING
# ─────────────────────────────────────────────────────────
def negative_sample(df, n_users, n_items, num_neg=4):
    """For each (user, item) positive, sample `num_neg` unseen items."""
    user_item_set = set(zip(df["user"], df["item"]))

    users, items, labels = [], [], []
    for u, i in zip(df["user"], df["item"]):
        users.append(u)
        items.append(i)
        labels.append(1)
        for _ in range(num_neg):
            neg = np.random.randint(0, n_items)
            while (u, neg) in user_item_set:
                neg = np.random.randint(0, n_items)
            users.append(u)
            items.append(neg)
            labels.append(0)

    sample_df = pd.DataFrame({"user": users, "item": items, "label": labels})
    sample_df = sample_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return sample_df


# ─────────────────────────────────────────────────────────
#  3. NCF MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────
def build_ncf(n_users, n_items, embedding_dim=32, mlp_layers=None):
    """
    Neural Collaborative Filtering (He et al., 2017)
    Combines:
      - GMF: element-wise product of user & item embeddings
      - MLP: concatenated embeddings passed through dense layers
    """
    if mlp_layers is None:
        mlp_layers = [64, 32, 16, 8]

    # ── Inputs ──────────────────────────────────────────
    user_input = layers.Input(shape=(1,), name="user_input")
    item_input = layers.Input(shape=(1,), name="item_input")

    # ── GMF Branch ──────────────────────────────────────
    gmf_user_emb = layers.Embedding(n_users, embedding_dim, name="gmf_user_emb")(user_input)
    gmf_item_emb = layers.Embedding(n_items, embedding_dim, name="gmf_item_emb")(item_input)
    gmf_user_flat = layers.Flatten(name="gmf_user_flat")(gmf_user_emb)
    gmf_item_flat = layers.Flatten(name="gmf_item_flat")(gmf_item_emb)
    gmf_out = layers.Multiply(name="gmf_multiply")([gmf_user_flat, gmf_item_flat])

    # ── MLP Branch ──────────────────────────────────────
    mlp_user_emb = layers.Embedding(n_users, mlp_layers[0] // 2, name="mlp_user_emb")(user_input)
    mlp_item_emb = layers.Embedding(n_items, mlp_layers[0] // 2, name="mlp_item_emb")(item_input)
    mlp_user_flat = layers.Flatten(name="mlp_user_flat")(mlp_user_emb)
    mlp_item_flat = layers.Flatten(name="mlp_item_flat")(mlp_item_emb)
    mlp_concat = layers.Concatenate(name="mlp_concat")([mlp_user_flat, mlp_item_flat])

    x = mlp_concat
    for i, units in enumerate(mlp_layers):
        x = layers.Dense(units, activation="relu", name=f"mlp_dense_{i}")(x)
        x = layers.Dropout(0.2, name=f"mlp_dropout_{i}")(x)
    mlp_out = x

    # ── NeuMF Fusion ────────────────────────────────────
    concat = layers.Concatenate(name="neumf_concat")([gmf_out, mlp_out])
    output = layers.Dense(1, activation="sigmoid", name="output")(concat)

    model = Model(inputs=[user_input, item_input], outputs=output, name="NeuralCF")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─────────────────────────────────────────────────────────
#  4. EVALUATION — Hit Rate@K & NDCG@K
# ─────────────────────────────────────────────────────────
def evaluate_model(model, test_df, n_items, user_item_set, top_k=10):
    """
    Leave-one-out evaluation:
      - For each user, take their last interaction as the positive item.
      - Sample 99 random negatives.
      - Score all 100 items and check if positive lands in top_k.
    """
    hits, ndcgs = [], []
    users_evaluated = 0

    grouped = test_df.groupby("user")
    for user_idx, group in tqdm(grouped, desc="Evaluating"):
        pos_items = group[group["label"] == 1]["item"].tolist()
        if not pos_items:
            continue
        pos_item = pos_items[0]

        # 99 negatives
        negatives = []
        while len(negatives) < 99:
            neg = np.random.randint(0, n_items)
            if neg != pos_item and (user_idx, neg) not in user_item_set:
                negatives.append(neg)

        eval_items = [pos_item] + negatives
        eval_users = [user_idx] * 100

        preds = model.predict(
            [np.array(eval_users), np.array(eval_items)],
            batch_size=100,
            verbose=0
        ).flatten()

        # Rank items by predicted score (descending)
        top_k_items = np.argsort(preds)[::-1][:top_k]
        top_k_item_indices = [eval_items[i] for i in top_k_items]

        # Hit Rate
        hit = 1 if pos_item in top_k_item_indices else 0
        hits.append(hit)

        # NDCG
        if hit:
            rank = top_k_item_indices.index(pos_item) + 1
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            ndcgs.append(0.0)

        users_evaluated += 1

    hr   = np.mean(hits)
    ndcg = np.mean(ndcgs)
    print(f"\n  Hit Rate@{top_k}  : {hr:.4f}")
    print(f"  NDCG@{top_k}      : {ndcg:.4f}")
    print(f"  Users evaluated  : {users_evaluated}")
    return hr, ndcg


# ─────────────────────────────────────────────────────────
#  5. TRAINING PLOTS
# ─────────────────────────────────────────────────────────
def plot_training(history, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Neural Collaborative Filtering — Training Curves", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(history.history["loss"],     label="Train Loss", color="#4C9BE8", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss",   color="#E8834C", linewidth=2, linestyle="--")
    axes[0].set_title("Binary Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history.history["accuracy"],     label="Train Accuracy", color="#4CE8A0", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy",   color="#E84C6E", linewidth=2, linestyle="--")
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Training curves saved → {save_path}")
    plt.close()


def plot_metrics_bar(hr, ndcg, save_path="evaluation_metrics.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [f"Hit Rate@{TOP_K}", f"NDCG@{TOP_K}"],
        [hr, ndcg],
        color=["#4C9BE8", "#E8834C"],
        width=0.4,
        edgecolor="white"
    )
    ax.set_ylim(0, 1)
    ax.set_title("Recommendation Quality Metrics", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    for bar, val in zip(bars, [hr, ndcg]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Evaluation metrics saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────
#  6. GENERATE RECOMMENDATIONS FOR A USER
# ─────────────────────────────────────────────────────────
def get_top_n_recommendations(model, user_idx, n_items, user_item_set, movies_df, item2idx, n=10):
    """Return top-N movie recommendations for a user."""
    idx2item = {v: k for k, v in item2idx.items()}

    # Items the user has NOT interacted with
    seen_items = {i for (u, i) in user_item_set if u == user_idx}
    candidate_items = [i for i in range(n_items) if i not in seen_items]

    users_arr = np.array([user_idx] * len(candidate_items))
    items_arr = np.array(candidate_items)
    preds = model.predict([users_arr, items_arr], batch_size=512, verbose=0).flatten()

    top_indices = np.argsort(preds)[::-1][:n]
    top_item_idxs = [candidate_items[i] for i in top_indices]
    top_scores    = [preds[i] for i in top_indices]

    recommendations = []
    for item_idx, score in zip(top_item_idxs, top_scores):
        original_item_id = idx2item[item_idx]
        title_row = movies_df[movies_df["item_id"] == original_item_id]
        title = title_row["title"].values[0] if len(title_row) > 0 else f"Item {original_item_id}"
        recommendations.append({"title": title, "score": float(score), "item_id": int(original_item_id)})

    return recommendations


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Recommendation System Using Neural Networks")
    print("  Deep Learning Project — Kartik Sehrawat")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading MovieLens 100K Dataset...")
    df, movies_df = load_movielens()
    df, user2idx, item2idx, n_users, n_items = preprocess(df)

    user_item_set = set(zip(df["user"], df["item"]))

    # 2. Train/test split
    print("\n[2/6] Splitting Dataset...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
    print(f"  Train size: {len(train_df)} | Test size: {len(test_df)}")

    # 3. Negative sampling (training only)
    print("\n[3/6] Generating Negative Samples...")
    train_sample = negative_sample(train_df, n_users, n_items, num_neg=NUM_NEGATIVES)
    print(f"  Total training samples (with negatives): {len(train_sample)}")

    # 4. Build model
    print("\n[4/6] Building NCF Model...")
    model = build_ncf(n_users, n_items, EMBEDDING_DIM, MLP_LAYERS)
    model.summary()

    # 5. Train
    print("\n[5/6] Training Model...")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(
        [train_sample["user"].values, train_sample["item"].values],
        train_sample["label"].values,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    plot_training(history)

    # 6. Evaluate
    print("\n[6/6] Evaluating Model...")
    hr, ndcg = evaluate_model(model, test_df, n_items, user_item_set, TOP_K)
    plot_metrics_bar(hr, ndcg)

    # Save model and encodings
    model.save("model.keras")
    with open("encodings.pkl", "wb") as f:
        pickle.dump({
            "user2idx": user2idx,
            "item2idx": item2idx,
            "n_users": n_users,
            "n_items": n_items,
            "user_item_set": user_item_set,
            "hr": hr,
            "ndcg": ndcg,
            "top_k": TOP_K
        }, f)
    print("\n  Model saved → model.keras")
    print("  Encodings saved → encodings.pkl")

    # Sample recommendations
    print("\n── Sample Recommendations for User 0 ──────────────────")
    recs = get_top_n_recommendations(model, 0, n_items, user_item_set, movies_df, item2idx, n=10)
    for rank, r in enumerate(recs, 1):
        print(f"  {rank:2d}. {r['title']:<50s}  score: {r['score']:.4f}")

    print("\n✅ Done! Run `python app.py` to launch the web interface.")
    print("=" * 60)
