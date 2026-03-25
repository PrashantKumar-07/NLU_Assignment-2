"""
Shared character vocabulary used by train.py, generate.py, and evaluate.py.
Kept in its own module so pickle can reconstruct CharVocab regardless of
which script is currently running as __main__.
"""

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"


class CharVocab:
    """Character-level vocabulary with special tokens."""

    def __init__(self, names):
        chars = sorted(set("".join(names)))
        self.tokens   = [PAD, BOS, EOS] + chars
        self.char2idx = {c: i for i, c in enumerate(self.tokens)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.size     = len(self.tokens)
        self.pad_idx  = self.char2idx[PAD]
        self.bos_idx  = self.char2idx[BOS]
        self.eos_idx  = self.char2idx[EOS]

    def encode(self, name):
        return (
            [self.bos_idx]
            + [self.char2idx[c] for c in name if c in self.char2idx]
            + [self.eos_idx]
        )

    def decode(self, indices):
        chars = []
        for i in indices:
            c = self.idx2char.get(i, "")
            if c in (PAD, BOS):
                continue
            if c == EOS:
                break
            chars.append(c)
        return "".join(chars)