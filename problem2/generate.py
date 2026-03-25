import json
import pickle
import logging
import random
from pathlib import Path

import torch

from models import VanillaRNN, BidirectionalLSTM, AttentionRNN
from char_vocab import CharVocab

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
CKPT_DIR = BASE_DIR / "checkpoints"
OUT_DIR  = BASE_DIR / "outputs"

MODEL_HYPERPARAMS = {
    "vanilla_rnn"   : {"embed_dim": 64, "hidden_size": 256, "num_layers": 1, "dropout": 0.3},
    "blstm"         : {"embed_dim": 64, "hidden_size": 64,  "num_layers": 2, "dropout": 0.5},
    "attention_rnn" : {"embed_dim": 64, "hidden_size": 64,  "num_layers": 1, "dropout": 0.5},
}

MAX_LEN    = 30
N_GENERATE = 200
MIN_LEN    = 6
device     = torch.device("cpu")


def top_k_sample(logits, temperature=0.9, top_k=10, recent_ids=None, rep_penalty=1.8):
    logits = logits.clone()
    if recent_ids:
        for idx in set(recent_ids):
            logits[idx] = logits[idx] / rep_penalty if logits[idx] > 0 else logits[idx] * rep_penalty
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.size(-1)))[0][-1]
        logits    = logits.masked_fill(logits < threshold, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if torch.isnan(probs).any() or probs.sum() == 0:
        return logits.argmax().item()
    return torch.multinomial(probs, num_samples=1).item()


def _suppress_eos_if_short(logits, generated_ids, vocab, min_chars=4):
    real = sum(1 for i in generated_ids
               if i not in (vocab.bos_idx, vocab.eos_idx, vocab.pad_idx))
    if real < min_chars:
        logits = logits.clone()
        logits[vocab.eos_idx] = float("-inf")
    return logits


def generate_name(model, vocab, temperature=0.9, top_k=10):
    model.eval()
    with torch.no_grad():
        x             = torch.tensor([[vocab.bos_idx]], dtype=torch.long)
        hidden        = model.init_hidden(1, device)
        generated_ids = []
        for _ in range(MAX_LEN):
            logits, hidden = model(x, hidden)
            raw      = _suppress_eos_if_short(logits[0, -1, :], generated_ids, vocab)
            next_idx = top_k_sample(raw, temperature, top_k,
                                    recent_ids=generated_ids[-8:], rep_penalty=1.5)
            if next_idx == vocab.eos_idx:
                break
            if next_idx in (vocab.pad_idx, vocab.bos_idx):
                continue
            generated_ids.append(next_idx)
            x = torch.tensor([[next_idx]], dtype=torch.long)
    return "".join(vocab.idx2char.get(i, "") for i in generated_ids).strip().title()


def generate_from_blstm(model, vocab, temperature=0.82, top_k=12):
    model.eval()
    with torch.no_grad():
        generated_ids = [vocab.bos_idx]
        for _ in range(MAX_LEN):
            x        = torch.tensor([generated_ids], dtype=torch.long)
            logits, _ = model(x, None)
            raw      = _suppress_eos_if_short(logits[0, -1, :], generated_ids, vocab,
                                              min_chars=8)
            next_idx = top_k_sample(raw, temperature, top_k,
                                    recent_ids=generated_ids[-6:], rep_penalty=1.3)
            if next_idx == vocab.eos_idx:
                break
            if next_idx in (vocab.pad_idx, vocab.bos_idx):
                continue
            generated_ids.append(next_idx)
    return "".join(vocab.idx2char.get(i, "") for i in generated_ids[1:]).strip().title()


def generate_from_attention(model, vocab, temperature=0.85, top_k=10):
    model.eval()
    with torch.no_grad():
        generated_ids = [vocab.bos_idx]
        for _ in range(MAX_LEN):
            x        = torch.tensor([generated_ids], dtype=torch.long)
            logits, _ = model(x, None)
            raw      = _suppress_eos_if_short(logits[0, -1, :], generated_ids, vocab)
            next_idx = top_k_sample(raw, temperature, top_k,
                                    recent_ids=generated_ids[-8:], rep_penalty=2.0)
            if next_idx == vocab.eos_idx:
                break
            if next_idx in (vocab.pad_idx, vocab.bos_idx):
                continue
            generated_ids.append(next_idx)
    return "".join(vocab.idx2char.get(i, "") for i in generated_ids[1:]).strip().title()


def main():
    with open(OUT_DIR / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    V = vocab.size
    p = MODEL_HYPERPARAMS

    model_configs = {
        "vanilla_rnn"   : (VanillaRNN(V, p["vanilla_rnn"]["embed_dim"],
                                      p["vanilla_rnn"]["hidden_size"],
                                      p["vanilla_rnn"]["dropout"]),
                           generate_name),
        "blstm"         : (BidirectionalLSTM(V, p["blstm"]["embed_dim"],
                                              p["blstm"]["hidden_size"],
                                              p["blstm"]["num_layers"],
                                              p["blstm"]["dropout"]),
                           generate_from_blstm),
        "attention_rnn" : (AttentionRNN(V, p["attention_rnn"]["embed_dim"],
                                        p["attention_rnn"]["hidden_size"],
                                        p["attention_rnn"]["num_layers"],
                                        p["attention_rnn"]["dropout"]),
                           generate_from_attention),
    }

    all_generated = {}

    for model_name, (model, gen_fn) in model_configs.items():
        ckpt = CKPT_DIR / f"{model_name}_best.pt"
        if not ckpt.exists():
            log.warning(f"Checkpoint not found: {ckpt}. Skipping.")
            continue

        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)

        names, attempts = [], 0
        while len(names) < N_GENERATE and attempts < N_GENERATE * 10:
            name = gen_fn(model, vocab, temperature=random.uniform(0.75, 1.0))
            if len(name) >= MIN_LEN:
                names.append(name)
            attempts += 1

        all_generated[model_name] = names
        log.info(f"Generated {len(names)} names from {model_name}")

        with open(OUT_DIR / f"{model_name}_names.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(names) + "\n")

    with open(OUT_DIR / "generated_names.json", "w", encoding="utf-8") as f:
        json.dump(all_generated, f, indent=2, ensure_ascii=False)

    log.info(f"All generated names saved -> {OUT_DIR}")

    for model_name, names in all_generated.items():
        print(f"\n  {model_name} (first 10):")
        for n in names[:10]:
            print(f"    {n}")


if __name__ == "__main__":
    main()