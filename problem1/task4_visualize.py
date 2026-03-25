"""
Task 4 - Visualization

Projects word embeddings into 2D using PCA and t-SNE and produces
cluster plots comparing Scratch CBOW, Scratch Skip-gram, and Gensim CBOW.

Gensim embeddings are extracted once as plain NumPy arrays — no gensim
similarity API is used after that point.

Depends on: models/  (produced by task2_train.py)
Saves to  : outputs/  (embedding_pca.png, embedding_tsne.png,
                        scratch_pca.png, loss_curves.png)
"""

import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import Word2Vec as GensimWord2Vec
from vocab import Vocabulary
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
OUT_DIR   = BASE_DIR / "outputs"

# Words grouped by theme — used to colour-code the scatter plots.
# Only words present in a model's vocabulary will appear in that model's plot.
WORD_GROUPS = {
    "research"  : ["research", "phd", "thesis", "publication",
                   "laboratory", "innovation", "discovery", "project",
                   "supervisor", "dissertation", "proposal"],
    "academics" : ["exam", "grade", "semester", "cgpa", "credit",
                   "syllabus", "evaluation", "assignment", "examination",
                   "marks", "attendance", "coursework"],
    "people"    : ["student", "faculty", "professor", "instructor", "advisor",
                   "coordinator", "committee", "dean", "chairman"],
    "programmes": ["btech", "mtech", "msc", "dual", "degree", "undergraduate",
                   "postgraduate", "programme", "curriculum", "ug", "pg"],
    "admin"     : ["registration", "admission", "leave",
                   "senate", "department", "circular", "regulation"],
}

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


def load_scratch(pkl_path, vocab_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return data["embeddings"], vocab.word2idx, vocab.idx2word


def load_gensim_as_numpy(model_path):
    gm    = GensimWord2Vec.load(str(model_path))
    words = list(gm.wv.key_to_index.keys())
    return (
        np.array(gm.wv.vectors, dtype=np.float32),
        {w: i for i, w in enumerate(words)},
        {i: w for i, w in enumerate(words)},
    )


def collect_group_vectors(embeddings, word2idx):
    vecs, words, group_tags = [], [], []
    for group, word_list in WORD_GROUPS.items():
        for w in word_list:
            if w in word2idx:
                vecs.append(embeddings[word2idx[w]])
                words.append(w)
                group_tags.append(group)
    return (np.array(vecs), words, group_tags) if vecs else (np.array([]), [], [])


def project_pca(vecs):
    return PCA(n_components=2, random_state=42).fit_transform(normalize(vecs))


def project_tsne(vecs):
    perp = min(20, len(vecs) - 1)
    return TSNE(n_components=2, perplexity=perp, random_state=42,
                n_iter=1000, learning_rate="auto", init="pca").fit_transform(normalize(vecs))


def scatter_plot(coords, words, groups, group_names, title, ax):
    color_map = {g: COLORS[i % len(COLORS)] for i, g in enumerate(group_names)}
    for g in group_names:
        idxs = [i for i, grp in enumerate(groups) if grp == g]
        ax.scatter(coords[idxs, 0], coords[idxs, 1],
                   c=color_map[g], label=g, s=60, alpha=0.85, zorder=3)
    for i, word in enumerate(words):
        ax.annotate(word, (coords[i, 0], coords[i, 1]),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points", alpha=0.9)
    # annotate collapsed CBOW plots so the report reader understands
    if "CBOW" in title and len(set(zip(coords[:,0].round(2), coords[:,1].round(2)))) < 5:
        ax.text(0.5, 0.02, "Note: CBOW vectors collapsed (small corpus limitation)",
                ha="center", va="bottom", transform=ax.transAxes,
                fontsize=7, color="grey", style="italic")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(loc="best", fontsize=7, framealpha=0.7)
    ax.set_xlabel("Component 1", fontsize=8)
    ax.set_ylabel("Component 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.3)


def comparison_plot(model_dict, method, save_path):
    group_names = list(WORD_GROUPS.keys())
    labels      = list(model_dict.keys())
    fig, axes   = plt.subplots(1, len(labels), figsize=(8 * len(labels), 8))
    if len(labels) == 1:
        axes = [axes]

    fig.suptitle(f"Word Embedding Clusters — {method.upper()}",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, label in zip(axes, labels):
        emb, w2i, i2w = model_dict[label]
        vecs, words, groups = collect_group_vectors(emb, w2i)
        if len(vecs) < 5:
            ax.text(0.5, 0.5, "Not enough vocab overlap",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        coords = project_pca(vecs) if method == "pca" else project_tsne(vecs)
        scatter_plot(coords, words, groups, group_names,
                     title=f"{label} — {method.upper()}", ax=ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved -> {save_path}")


def loss_curve_plot(scratch_paths, save_path):
    colors = {"Scratch CBOW": "#3498db", "Scratch Skip-gram": "#e74c3c"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, pkl_path in scratch_paths.items():
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        losses = data.get("losses", [])
        if losses:
            ax.plot(range(1, len(losses) + 1), losses,
                    marker="o", label=label,
                    color=colors.get(label, "grey"), linewidth=2)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Average Loss", fontsize=10)
    ax.set_title("Scratch Word2Vec — Training Loss", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved -> {save_path}")


def main():
    vocab_path        = MODEL_DIR / "vocabulary.pkl"
    scratch_cbow_path = MODEL_DIR / "scratch_cbow_best.pkl"
    scratch_sg_path   = MODEL_DIR / "scratch_skipgram_best.pkl"
    gensim_cbow_path  = MODEL_DIR / "gensim_cbow_best.model"
    gensim_sg_path    = MODEL_DIR / "gensim_sg_best.model"

    if not scratch_cbow_path.exists() or not scratch_sg_path.exists():
        log.error("Scratch best models not found. Run task2_train.py first.")
        return

    scratch_cbow = load_scratch(scratch_cbow_path, vocab_path)
    scratch_sg   = load_scratch(scratch_sg_path,   vocab_path)

    scratch_models = {
        "Scratch CBOW"     : scratch_cbow,
        "Scratch Skip-gram": scratch_sg,
    }

    all_models = dict(scratch_models)
    if gensim_cbow_path.exists():
        all_models["Gensim CBOW (ref)"]      = load_gensim_as_numpy(gensim_cbow_path)
    if gensim_sg_path.exists():
        all_models["Gensim Skip-gram (ref)"] = load_gensim_as_numpy(gensim_sg_path)

    loss_curve_plot(
        {"Scratch CBOW": scratch_cbow_path, "Scratch Skip-gram": scratch_sg_path},
        OUT_DIR / "loss_curves.png",
    )

    log.info("Generating PCA plots (scratch models) ...")
    comparison_plot(scratch_models, "pca", OUT_DIR / "scratch_pca.png")

    log.info("Generating PCA comparison plot (all models) ...")
    comparison_plot(all_models, "pca", OUT_DIR / "embedding_pca.png")

    log.info("Generating t-SNE comparison plot ...")
    comparison_plot(all_models, "tsne", OUT_DIR / "embedding_tsne.png")

    log.info("All visualisations saved in outputs/")


if __name__ == "__main__":
    main()