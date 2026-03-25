"""
Task 3 - Semantic Analysis

Computes nearest neighbours and analogy results using our own cosine
similarity implementation. Gensim embeddings are extracted as plain
NumPy arrays so the same cosine code runs on all four models — gensim's
.most_similar() API is never called.

Depends on: models/  (produced by task2_train.py)
Saves to  : outputs/semantic_analysis.txt
            outputs/semantic_analysis.json
"""

import json
import pickle
import logging
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec as GensimWord2Vec

# Vocabulary must be imported here so pickle can reconstruct the object.
# If omitted, loading vocabulary.pkl raises:
#   AttributeError: Can't get attribute 'Vocabulary' on <module '__main__'>
from vocab import Vocabulary

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
OUT_DIR   = BASE_DIR / "outputs"

SCRATCH_CBOW_PATH = MODEL_DIR / "scratch_cbow_best.pkl"
SCRATCH_SG_PATH   = MODEL_DIR / "scratch_skipgram_best.pkl"
GENSIM_CBOW_PATH  = MODEL_DIR / "gensim_cbow_best.model"
GENSIM_SG_PATH    = MODEL_DIR / "gensim_sg_best.model"
VOCAB_PATH        = MODEL_DIR / "vocabulary.pkl"

PROBE_WORDS = ["research", "student", "phd", "exam", "faculty"]

ANALOGIES = [
    ("ug",         "btech",       "pg"),          # UG:BTech :: PG:?
    ("lecture",    "professor",   "laboratory"),  # lecture:professor :: lab:?
    ("semester",   "examination", "year"),        # semester:exam :: year:?
    ("mtech",      "thesis",      "btech"),       # MTech:thesis :: BTech:?
    ("research",   "phd",         "teaching"),    # research:PhD :: teaching:?
    ("professor",  "department",  "student"),     # professor:dept :: student:?
    ("admission",  "undergraduate","registration"),# admission:UG :: registration:?
]


# English stopwords to exclude from nearest-neighbour results.
# Even after training with stopwords removed, gensim models may still
# have them in vocabulary; filtering here keeps results meaningful.
import nltk as _nltk
_nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as _sw_corpus
_RESULT_STOPWORDS = set(_sw_corpus.words("english"))


def nearest_neighbors(embeddings, word2idx, idx2word, query, topn=5):
    """
    Vectorised cosine similarity search.
    Normalising once and using a single matrix multiply is ~100x faster
    than computing cosine(query, word) in a Python loop.
    Stopwords and numeric tokens are excluded from results.
    """
    if query not in word2idx:
        return []
    idx    = word2idx[query]
    target = embeddings[idx]
    norms  = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    scores = (embeddings / norms) @ (target / (np.linalg.norm(target) + 1e-10))
    ranked = np.argsort(-scores)
    results = []
    for i in ranked:
        if i == idx:
            continue
        word = idx2word[i]
        # skip stopwords and pure-numeric tokens in results
        if word in _RESULT_STOPWORDS or word.isdigit():
            continue
        results.append((word, float(scores[i])))
        if len(results) >= topn:
            break
    return results


def analogy_3cosadd(embeddings, word2idx, idx2word, a, b, c, topn=5):
    """
    3CosAdd: a:b :: c:?
    target = emb(b) - emb(a) + emb(c), then find nearest neighbours
    excluding the three input words themselves.
    """
    for w in (a, b, c):
        if w not in word2idx:
            log.warning(f"'{w}' not in vocabulary, skipping analogy.")
            return []
    target  = (embeddings[word2idx[b]]
              - embeddings[word2idx[a]]
              + embeddings[word2idx[c]])
    exclude = {word2idx[w] for w in (a, b, c)}
    norms   = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    scores  = (embeddings / norms) @ (target / (np.linalg.norm(target) + 1e-10))
    ranked  = np.argsort(-scores)
    results = []
    for i in ranked:
        if i in exclude:
            continue
        results.append((idx2word[i], float(scores[i])))
        if len(results) >= topn:
            break
    return results


def load_scratch(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    return data["embeddings"], vocab.word2idx, vocab.idx2word


def load_gensim_as_numpy(model_path):
    """Extract the embedding matrix from a gensim model as a plain NumPy array."""
    gm    = GensimWord2Vec.load(str(model_path))
    words = list(gm.wv.key_to_index.keys())
    return (
        np.array(gm.wv.vectors, dtype=np.float32),
        {w: i for i, w in enumerate(words)},
        {i: w for i, w in enumerate(words)},
    )


def main():
    for p in [SCRATCH_CBOW_PATH, SCRATCH_SG_PATH, VOCAB_PATH]:
        if not p.exists():
            log.error(f"Required file not found: {p}. Run task2_train.py first.")
            return

    models = {
        "Scratch CBOW"      : load_scratch(SCRATCH_CBOW_PATH),
        "Scratch Skip-gram" : load_scratch(SCRATCH_SG_PATH),
    }
    if GENSIM_CBOW_PATH.exists():
        models["Gensim CBOW (ref)"]      = load_gensim_as_numpy(GENSIM_CBOW_PATH)
    if GENSIM_SG_PATH.exists():
        models["Gensim Skip-gram (ref)"] = load_gensim_as_numpy(GENSIM_SG_PATH)

    report, data = [], {}

    report.append("=" * 65)
    report.append("NEAREST NEIGHBOURS  (top-5, cosine similarity)")
    report.append("=" * 65)

    for word in PROBE_WORDS:
        report.append(f"\n  Query word: '{word}'")
        data[f"neighbors_{word}"] = {}
        for label, (emb, w2i, i2w) in models.items():
            nbrs = nearest_neighbors(emb, w2i, i2w, word)
            report.append(f"    {label}:")
            if nbrs:
                for w, s in nbrs:
                    report.append(f"      {w:<28} {s:.4f}")
            else:
                report.append("      (word not in this model's vocabulary)")
            data[f"neighbors_{word}"][label] = [(w, round(s, 4)) for w, s in nbrs]

    report.append("\n" + "=" * 65)
    report.append("ANALOGY EXPERIMENTS  (a : b :: c : ?  via 3CosAdd)")
    report.append("=" * 65)

    for a, b, c in ANALOGIES:
        report.append(f"\n  {a.upper()} : {b.upper()} :: {c.upper()} : ?")
        key = f"analogy_{a}_{b}_{c}"
        data[key] = {}
        for label, (emb, w2i, i2w) in models.items():
            results = analogy_3cosadd(emb, w2i, i2w, a, b, c)
            report.append(f"    {label}:")
            if results:
                for w, s in results:
                    report.append(f"      {w:<28} {s:.4f}")
                report.append(f"      -> top answer: '{results[0][0]}'")
            else:
                report.append("      (one or more words OOV)")
            data[key][label] = [(w, round(s, 4)) for w, s in results]

    text = "\n".join(report)
    print(text)

    with open(OUT_DIR / "semantic_analysis.txt", "w", encoding="utf-8") as f:
        f.write(text)
    with open(OUT_DIR / "semantic_analysis.json", "w") as f:
        json.dump(data, f, indent=2)

    log.info("Semantic analysis saved.")


if __name__ == "__main__":
    main()