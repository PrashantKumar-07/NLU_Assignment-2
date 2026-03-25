"""
Task 2 - Model Training

Trains Word2Vec from scratch (pure NumPy) as the primary implementation,
then trains gensim on the same hyperparameter settings for comparison.

Scratch implementation covers 3 representative hyperparameter combos to
keep training time manageable on a laptop. Gensim runs the full 12-combo
grid since it finishes in seconds.

Depends on: corpus/cleaned_corpus.txt  (produced by task1_corpus.py)
Saves to  : models/   outputs/training_results.json
"""

import json
import time
import logging
import pickle
from pathlib import Path
from collections import Counter
from itertools import product

import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec as GensimWord2Vec

# Vocabulary lives in its own module so pickle can find the class
# when task3/task4 load vocabulary.pkl from a different __main__ context.
from vocab import Vocabulary

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "corpus"
MODEL_DIR  = BASE_DIR / "models"
OUT_DIR    = BASE_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

CLEANED_CORPUS_FILE = CORPUS_DIR / "corpus.txt"

# Short academic terms that must survive the len > 2 token filter.
KEEP_SHORT = {"ug", "pg", "phd", "msc", "mba", "iit", "ncc", "nss", "nso", "cgpa", "sgpa"}


def load_sentences(path):
    from nltk.corpus import stopwords
    # removing stopwords before training prevents high-frequency function words
    # ("the", "and", "for") from dominating cosine similarities and making
    # every query word's nearest neighbours useless
    _STOPWORDS = set(stopwords.words("english")) - KEEP_SHORT

    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = word_tokenize(line)
                tokens = [
                    t for t in tokens
                    if t.isalpha()
                    and (len(t) > 2 or t in KEEP_SHORT)
                    and t not in _STOPWORDS
                    and not t.isdigit()
                ]
                if tokens:
                    sentences.append(tokens)
    log.info(f"Loaded {len(sentences)} sentences from corpus.")
    return sentences


class Word2VecScratch:
    """
    Word2Vec in pure NumPy supporting both CBOW and Skip-gram with negative sampling.

    Key implementation choices:
    - Linear LR decay (lr_start → lr_min) prevents the high constant LR from
      causing gradient oscillations in later epochs, which collapses all vectors.
    - Small init scale (0.01) keeps initial dot products near zero so sigmoid
      doesn't saturate immediately and stall learning from the first step.
    - W_in only as final embedding — averaging with W_out hurts quality on small
      corpora where W_out hasn't converged enough.
    - Subsampling of frequent words (matching the original paper's formula)
      reduces ~20% of training time with no meaningful quality loss.
    """

    SUBSAMPLE_T = 1e-4

    def __init__(self, vocab_size, embed_dim=100, mode="cbow",
                 window=5, neg_samples=5, learning_rate=0.025):
        self.V        = vocab_size
        self.D        = embed_dim
        self.mode     = mode
        self.window   = window
        self.K        = neg_samples
        self.lr_start = learning_rate
        self.lr_min   = 0.0001
        self.lr       = learning_rate

        scale      = 0.01
        self.W_in  = np.random.uniform(-scale, scale, (vocab_size, embed_dim))
        self.W_out = np.zeros((vocab_size, embed_dim))
        self.losses = []

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def _draw_negatives(self, exclude, n):
        negs = []
        while len(negs) < n:
            idx = np.random.randint(0, self.V)
            if idx not in exclude:
                negs.append(idx)
        return negs

    def _subsample(self, encoded, word_freq, total_tokens):
        """Probabilistically drop frequent words, matching the Word2Vec paper formula."""
        subsampled = []
        for sent in encoded:
            kept = []
            for idx in sent:
                if idx not in word_freq:
                    kept.append(idx)
                    continue
                f         = word_freq[idx] / total_tokens
                keep_prob = min(1.0, (np.sqrt(f / self.SUBSAMPLE_T) + 1) * (self.SUBSAMPLE_T / f))
                if np.random.random() < keep_prob:
                    kept.append(idx)
            if kept:
                subsampled.append(kept)
        return subsampled

    def _cbow_step(self, context_ids, target_id):
        """One CBOW update: average context vectors → predict target word."""
        if not context_ids:
            return 0.0

        h      = self.W_in[context_ids].mean(axis=0)
        loss   = 0.0
        grad_h = np.zeros(self.D)

        # positive sample
        pos   = self.W_out[target_id]
        p     = self._sigmoid(h @ pos)
        err   = p - 1.0
        loss -= np.log(p + 1e-10)
        grad_h += err * pos
        self.W_out[target_id] -= self.lr * err * h

        # negative samples
        for nid in self._draw_negatives({target_id}, self.K):
            neg   = self.W_out[nid]
            p     = self._sigmoid(h @ neg)
            loss -= np.log(1.0 - p + 1e-10)
            grad_h += p * neg
            self.W_out[nid] -= self.lr * p * h

        grad_per_ctx = grad_h / len(context_ids)
        for cid in context_ids:
            self.W_in[cid] -= self.lr * grad_per_ctx
        return loss

    def _skipgram_step(self, center_id, context_id):
        """One Skip-gram update: predict each context word from the center."""
        center = self.W_in[center_id]
        loss   = 0.0
        grad_c = np.zeros(self.D)

        pos   = self.W_out[context_id]
        p     = self._sigmoid(center @ pos)
        err   = p - 1.0
        loss -= np.log(p + 1e-10)
        grad_c += err * pos
        self.W_out[context_id] -= self.lr * err * center

        for nid in self._draw_negatives({center_id, context_id}, self.K):
            neg   = self.W_out[nid]
            p     = self._sigmoid(center @ neg)
            loss -= np.log(1.0 - p + 1e-10)
            grad_c += p * neg
            self.W_out[nid] -= self.lr * p * center

        self.W_in[center_id] -= self.lr * grad_c
        return loss

    def train(self, encoded, vocab_freq=None, total_tokens=None, epochs=5):
        working = self._subsample(encoded, vocab_freq, total_tokens) \
                  if vocab_freq and total_tokens else encoded

        total_steps = epochs * max(sum(len(s) for s in working), 1)
        step = 0

        for epoch in range(epochs):
            total_loss = 0.0
            n_updates  = 0
            t0         = time.time()

            for sent in working:
                if len(sent) < 2:
                    continue
                for i, center in enumerate(sent):
                    w   = np.random.randint(1, self.window + 1)
                    ctx = [sent[j] for j in range(max(0, i - w), min(len(sent), i + w + 1)) if j != i]

                    if self.mode == "cbow":
                        total_loss += self._cbow_step(ctx, center)
                        n_updates  += 1
                    else:
                        for cid in ctx:
                            total_loss += self._skipgram_step(center, cid)
                            n_updates  += 1

                # decay LR linearly after each sentence so late-epoch updates
                # are small refinements rather than large destabilising steps
                step    += 1
                progress = step / total_steps
                self.lr  = max(self.lr_min, self.lr_start * (1.0 - progress))

            avg = total_loss / max(n_updates, 1)
            self.losses.append(avg)
            log.info(f"  [scratch-{self.mode}] epoch {epoch+1}/{epochs}  loss={avg:.4f}  time={time.time()-t0:.1f}s")

        self.embeddings = np.array(self.W_in)

    def most_similar(self, word, vocab, topn=5):
        """Cosine similarity nearest-neighbour search — no external library used."""
        if word not in vocab.word2idx:
            return []
        idx    = vocab.word2idx[word]
        target = self.embeddings[idx]
        norms  = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        scores = (self.embeddings / norms) @ (target / (np.linalg.norm(target) + 1e-10))
        ranked = np.argsort(-scores)
        results = []
        for i in ranked:
            if i == idx:
                continue
            results.append((vocab.idx2word[i], float(scores[i])))
            if len(results) >= topn:
                break
        return results

    def analogy(self, a, b, c, vocab, topn=5):
        """3CosAdd: a:b :: c:?  →  target = emb(b) - emb(a) + emb(c)"""
        for w in (a, b, c):
            if w not in vocab.word2idx:
                log.warning(f"'{w}' not in scratch vocabulary.")
                return []
        v       = (self.embeddings[vocab.word2idx[b]]
                 - self.embeddings[vocab.word2idx[a]]
                 + self.embeddings[vocab.word2idx[c]])
        exclude = {vocab.word2idx[w] for w in (a, b, c)}
        norms   = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        scores  = (self.embeddings / norms) @ (v / (np.linalg.norm(v) + 1e-10))
        ranked  = np.argsort(-scores)
        results = []
        for i in ranked:
            if i in exclude:
                continue
            results.append((vocab.idx2word[i], float(scores[i])))
            if len(results) >= topn:
                break
        return results

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "W_in": self.W_in, "W_out": self.W_out,
                "losses": self.losses, "mode": self.mode,
                "D": self.D, "window": self.window, "K": self.K,
            }, f)
        log.info(f"Scratch model saved -> {path}")


def train_gensim_model(sentences, sg, vector_size, window, negative, epochs=10):
    """sg=0 → CBOW, sg=1 → Skip-gram. Used as reference only."""
    return GensimWord2Vec(
        sentences=sentences, sg=sg, vector_size=vector_size,
        window=window, negative=negative, min_count=3,
        workers=4, epochs=epochs, seed=42,
    )


# 3 representative combos for scratch (low / standard / high).
# Running all 12 in pure NumPy would take several hours on a laptop.
SCRATCH_GRID = [
    {"dim": 50,  "window": 3, "neg": 5},
    {"dim": 100, "window": 5, "neg": 5},
    {"dim": 200, "window": 5, "neg": 10},
]

EMBED_DIMS          = [50, 100, 200]
WINDOWS             = [3, 5]
NEG_SAMPLES         = [5, 10]
SCRATCH_EPOCHS_GRID = 10
SCRATCH_EPOCHS_BEST = 15
GENSIM_EPOCHS       = 10


def run_experiments(sentences, vocab):
    freq       = Counter(w for sent in sentences for w in sent)
    total_toks = sum(freq.values())
    vocab_freq = {vocab.word2idx[w]: c for w, c in freq.items() if w in vocab.word2idx}
    encoded    = vocab.encode(sentences)
    results    = []

    log.info("=" * 65)
    log.info("SCRATCH EXPERIMENTS")
    log.info("=" * 65)

    for cfg in SCRATCH_GRID:
        dim, win, neg = cfg["dim"], cfg["window"], cfg["neg"]
        tag = f"_d{dim}_w{win}_n{neg}"

        for mode, sg_flag in [("cbow", 0), ("skipgram", 1)]:
            model = Word2VecScratch(vocab_size=vocab.size, embed_dim=dim,
                                    mode=mode, window=win, neg_samples=neg)
            t0 = time.time()
            model.train(encoded, vocab_freq, total_toks, epochs=SCRATCH_EPOCHS_GRID)
            elapsed = round(time.time() - t0, 2)
            model.save(MODEL_DIR / f"scratch_{mode}{tag}.pkl")
            results.append({"impl": "scratch", "mode": mode, "dim": dim,
                             "window": win, "neg": neg, "train_time_s": elapsed,
                             "final_loss": round(model.losses[-1], 4)})
            log.info(f"  scratch {mode:<8} dim={dim} win={win} neg={neg} -> {elapsed:.1f}s")

    log.info("\n" + "=" * 65)
    log.info("GENSIM REFERENCE  (full 12-combo grid)")
    log.info("=" * 65)

    for dim, win, neg in product(EMBED_DIMS, WINDOWS, NEG_SAMPLES):
        tag = f"_d{dim}_w{win}_n{neg}"
        for mode, sg_flag in [("cbow", 0), ("skipgram", 1)]:
            t0 = time.time()
            gm = train_gensim_model(sentences, sg=sg_flag, vector_size=dim,
                                    window=win, negative=neg, epochs=GENSIM_EPOCHS)
            elapsed = round(time.time() - t0, 2)
            gm.save(str(MODEL_DIR / f"gensim_{mode}{tag}.model"))
            results.append({"impl": "gensim (reference)", "mode": mode, "dim": dim,
                             "window": win, "neg": neg, "train_time_s": elapsed,
                             "vocab_size": len(gm.wv)})

    log.info("\nTraining best-config models (dim=100, win=5, neg=5) for task3/task4 ...")
    for mode, sg_flag in [("cbow", 0), ("skipgram", 1)]:
        best = Word2VecScratch(vocab_size=vocab.size, embed_dim=100,
                               mode=mode, window=5, neg_samples=5)
        best.train(encoded, vocab_freq, total_toks, epochs=SCRATCH_EPOCHS_BEST)
        best.save(MODEL_DIR / f"scratch_{mode}_best.pkl")

        gm_best = train_gensim_model(sentences, sg=sg_flag, vector_size=100,
                                     window=5, negative=5, epochs=GENSIM_EPOCHS)
        gm_best.save(str(MODEL_DIR / f"gensim_{mode}_best.model"))

    return results


def main():
    if not CLEANED_CORPUS_FILE.exists():
        log.error("Cleaned corpus not found. Run task1_corpus.py first.")
        return

    sentences = load_sentences(CLEANED_CORPUS_FILE)
    vocab     = Vocabulary(sentences)
    log.info(f"Vocabulary: {vocab.size} words")

    with open(MODEL_DIR / "vocabulary.pkl", "wb") as f:
        pickle.dump(vocab, f)
    log.info("Vocabulary saved.")

    results = run_experiments(sentences, vocab)

    with open(OUT_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # deduplicate in case of interrupted/repeated runs
    seen, unique = set(), []
    for r in results:
        key = (r["impl"], r["mode"], r["dim"], r["window"], r["neg"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print("\n" + "=" * 72)
    print(f"{'IMPL':<22} {'MODE':<10} {'DIM':<6} {'WIN':<5} {'NEG':<5} {'TIME(s)':<10} {'LOSS/VOCAB'}")
    print("=" * 72)
    for r in unique:
        extra = r.get("final_loss") or r.get("vocab_size", "")
        print(f"{r['impl']:<22} {r['mode']:<10} {r['dim']:<6} "
              f"{r['window']:<5} {r['neg']:<5} {r['train_time_s']:<10} {extra}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()