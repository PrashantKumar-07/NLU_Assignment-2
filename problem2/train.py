import json
import time
import random
import logging
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models import VanillaRNN, BidirectionalLSTM, AttentionRNN, count_parameters
from char_vocab import CharVocab

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
NAMES_FILE = BASE_DIR / "TrainingNames.txt"
CKPT_DIR   = BASE_DIR / "checkpoints"
OUT_DIR    = BASE_DIR / "outputs"
CKPT_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

MODEL_HYPERPARAMS = {
    "vanilla_rnn": {
        "embed_dim": 64, "hidden_size": 256, "num_layers": 1,
        "dropout": 0.3,  "lr": 0.001, "weight_decay": 0.0, "epochs": 50,
    },
    "blstm": {
        "embed_dim": 64, "hidden_size": 64, "num_layers": 2,
        "dropout": 0.5,  "lr": 0.001, "weight_decay": 1e-4, "epochs": 25,
    },
    "attention_rnn": {
        "embed_dim": 64, "hidden_size": 64, "num_layers": 1,
        "dropout": 0.5,  "lr": 0.001, "weight_decay": 5e-4, "epochs": 25,
    },
}

HYPERPARAMS = {"batch_size": 64, "clip": 1.0, "seed": 42, "early_stop": 3}

torch.manual_seed(HYPERPARAMS["seed"])
random.seed(HYPERPARAMS["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")


def load_names(path):
    with open(path, "r", encoding="utf-8") as f:
        names = [l.strip().lower() for l in f if l.strip()]
    log.info(f"Loaded {len(names)} names from {path}")
    return names


class NameDataset(Dataset):
    def __init__(self, names, vocab):
        self.sequences = [vocab.encode(n) for n in names]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return (
            torch.tensor(seq[:-1], dtype=torch.long),
            torch.tensor(seq[1:],  dtype=torch.long),
        )


def collate_fn(batch, pad_idx):
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)
    inp_pad = torch.stack([
        torch.cat([x, torch.full((max_len - x.size(0),), pad_idx)]) for x in inputs
    ])
    tgt_pad = torch.stack([
        torch.cat([y, torch.full((max_len - y.size(0),), pad_idx)]) for y in targets
    ])
    return inp_pad, tgt_pad


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, n_batches = 0.0, 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad()
        logits, _ = model(inp)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), HYPERPARAMS["clip"])
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


def train_model(model, model_name, loader, epochs, lr, weight_decay=0.0):
    criterion    = nn.CrossEntropyLoss(ignore_index=0)
    optimizer    = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler    = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_loss    = float("inf")
    patience_cnt = 0
    t_start      = time.time()
    losses       = []
    log.info(f"Training {model_name}  ({count_parameters(model):,} params)")

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion)
        scheduler.step(loss)
        losses.append(loss)

        if loss < best_loss - 1e-4:
            best_loss    = loss
            patience_cnt = 0
            torch.save(model.state_dict(), CKPT_DIR / f"{model_name}_best.pt")
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"  [{model_name}] epoch {epoch:3d}/{epochs}  "
                     f"loss={loss:.4f}  best={best_loss:.4f}  "
                     f"elapsed={time.time()-t_start:.1f}s")

        if patience_cnt >= HYPERPARAMS["early_stop"]:
            log.info(f"  Early stopping at epoch {epoch}")
            break

    log.info(f"{model_name} done. Best loss: {best_loss:.4f}")
    return losses


def main():
    names = load_names(NAMES_FILE)
    vocab = CharVocab(names)
    log.info(f"Vocabulary size: {vocab.size} characters")

    with open(OUT_DIR / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    dataset = NameDataset(names, vocab)
    loader  = DataLoader(
        dataset,
        batch_size=HYPERPARAMS["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx),
    )

    V = vocab.size
    p = MODEL_HYPERPARAMS

    models_to_train = {
        "vanilla_rnn"   : VanillaRNN(V, p["vanilla_rnn"]["embed_dim"],
                                     p["vanilla_rnn"]["hidden_size"],
                                     p["vanilla_rnn"]["dropout"]).to(device),
        "blstm"         : BidirectionalLSTM(V, p["blstm"]["embed_dim"],
                                            p["blstm"]["hidden_size"],
                                            p["blstm"]["num_layers"],
                                            p["blstm"]["dropout"]).to(device),
        "attention_rnn" : AttentionRNN(V, p["attention_rnn"]["embed_dim"],
                                       p["attention_rnn"]["hidden_size"],
                                       p["attention_rnn"]["num_layers"],
                                       p["attention_rnn"]["dropout"]).to(device),
    }

    print("\n" + "=" * 50)
    print("MODEL PARAMETER COUNTS")
    print("=" * 50)
    for name, model in models_to_train.items():
        print(f"  {name:<22} {count_parameters(model):>10,} params")
    print("=" * 50 + "\n")

    all_losses = {}
    for name, model in models_to_train.items():
        hp = p[name]
        all_losses[name] = train_model(
            model, name, loader,
            epochs=hp["epochs"], lr=hp["lr"],
            weight_decay=hp.get("weight_decay", 0.0),
        )

    results = {
        "hyperparams" : HYPERPARAMS,
        "param_counts": {n: count_parameters(m) for n, m in models_to_train.items()},
        "losses"      : all_losses,
    }
    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Training history saved -> {OUT_DIR / 'training_history.json'}")


if __name__ == "__main__":
    main()