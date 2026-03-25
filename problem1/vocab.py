"""
Shared vocabulary module used by task2_train.py, task3_analysis.py, and task4_visualize.py.

Keeping Vocabulary in its own file is required for pickle to work correctly.
If the class were defined inside task2_train.py, unpickling it in another
script would raise:
    AttributeError: Can't get attribute 'Vocabulary' on <module '__main__'>
"""

from collections import Counter


class Vocabulary:
    """Word <-> integer index mapping with min-frequency filtering."""

    # Short academic acronyms that must survive the min_count filter.
    # These are meaningful for analogy experiments but rare in a small corpus.
    KEEP_ALWAYS = {
        "ug", "pg", "phd", "btech", "mtech", "msc", "mba",
        "cgpa", "sgpa", "iit", "ncc", "nss", "nso",
    }

    def __init__(self, sentences, min_count=3):
        freq = Counter(w for sent in sentences for w in sent)
        kept = [w for w, c in freq.items()
                if (c >= min_count or w in self.KEEP_ALWAYS)
                and not w.isdigit()]   # numeric tokens add no semantic signal

        self.word2idx = {w: i for i, w in enumerate(kept)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.size     = len(kept)
        self.freq     = {w: freq[w] for w in kept}

    def encode(self, sentences):
        encoded = []
        for sent in sentences:
            ids = [self.word2idx[w] for w in sent if w in self.word2idx]
            if ids:
                encoded.append(ids)
        return encoded