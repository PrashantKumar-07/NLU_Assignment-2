import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
OUT_DIR    = BASE_DIR / "outputs"
NAMES_FILE = BASE_DIR / "TrainingNames.txt"


def novelty_rate(generated, training_set):
    if not generated:
        return 0.0
    return sum(1 for n in generated if n.lower() not in training_set) / len(generated)


def diversity(generated):
    if not generated:
        return 0.0
    return len(set(generated)) / len(generated)


def avg_length(generated):
    if not generated:
        return 0.0
    return sum(len(n) for n in generated) / len(generated)


def analyse_failures(names):
    too_short  = [n for n in names if len(n) < 3]
    too_long   = [n for n in names if len(n) > 30]
    repetitive = [n for n in names if len(set(n.lower().replace(" ", ""))) < 3]
    return {
        "too_short_count"  : len(too_short),
        "too_long_count"   : len(too_long),
        "repetitive_count" : len(repetitive),
    }


def bar_comparison(metrics, save_path):
    model_names = list(metrics.keys())
    novelty = [metrics[m]["novelty_rate"] * 100 for m in model_names]
    div     = [metrics[m]["diversity"]    * 100 for m in model_names]

    x      = np.arange(len(model_names))
    width  = 0.35
    labels = [m.replace("_", "\n") for m in model_names]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, novelty, width, label="Novelty Rate (%)",
           color="#3498db", alpha=0.85)
    ax.bar(x + width / 2, div,     width, label="Diversity (%)",
           color="#2ecc71", alpha=0.85)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Novelty Rate vs Diversity", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Bar chart saved -> {save_path}")


def length_histogram(all_names, save_path):
    colors = {"vanilla_rnn": "#e74c3c", "blstm": "#3498db", "attention_rnn": "#2ecc71"}
    fig, ax = plt.subplots(figsize=(9, 4))
    for model_name, names in all_names.items():
        ax.hist([len(n) for n in names], bins=20, alpha=0.55,
                label=model_name.replace("_", " ").title(),
                color=colors.get(model_name, "grey"))
    ax.set_xlabel("Name Length (characters)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Distribution of Generated Name Lengths", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Length histogram saved -> {save_path}")


def training_loss_plot(history_path, save_path):
    if not history_path.exists():
        log.warning("training_history.json not found, skipping loss plot.")
        return
    with open(history_path) as f:
        history = json.load(f)

    colors = {"vanilla_rnn": "#e74c3c", "blstm": "#3498db", "attention_rnn": "#2ecc71"}
    labels = {"vanilla_rnn": "Vanilla RNN", "blstm": "BLSTM", "attention_rnn": "Attention RNN"}

    fig, ax = plt.subplots(figsize=(9, 4))
    for model_name, loss_list in history.get("losses", {}).items():
        ax.plot(range(1, len(loss_list) + 1), loss_list,
                label=labels.get(model_name, model_name),
                color=colors.get(model_name, "grey"), linewidth=2)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax.set_title("Training Loss", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Loss plot saved -> {save_path}")


def main():
    gen_path = OUT_DIR / "generated_names.json"
    if not gen_path.exists():
        log.error("generated_names.json not found. Run generate.py first.")
        return

    with open(gen_path) as f:
        all_generated = json.load(f)

    with open(NAMES_FILE, encoding="utf-8") as f:
        training_set = {l.strip().lower() for l in f if l.strip()}

    metrics = {}
    report_lines = ["=" * 60, "QUANTITATIVE EVALUATION", "=" * 60]

    for model_name, names in all_generated.items():
        nr     = novelty_rate(names, training_set)
        div    = diversity(names)
        al     = avg_length(names)
        fmodes = analyse_failures(names)

        metrics[model_name] = {
            "total_generated" : len(names),
            "novelty_rate"    : round(nr, 4),
            "diversity"       : round(div, 4),
            "avg_length"      : round(al, 2),
            "failure_modes"   : fmodes,
        }

        report_lines += [
            f"\n  Model: {model_name}",
            f"    Total generated : {len(names)}",
            f"    Novelty Rate    : {nr*100:.1f}%",
            f"    Diversity       : {div*100:.1f}%",
            f"    Avg name length : {al:.1f} chars",
            f"    Too short (<3)  : {fmodes['too_short_count']}",
            f"    Too long  (>30) : {fmodes['too_long_count']}",
            f"    Repetitive      : {fmodes['repetitive_count']}",
            f"\n    Samples:",
        ]
        for n in names[:15]:
            report_lines.append(f"      {n}")

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(OUT_DIR / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(OUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    bar_comparison(metrics, OUT_DIR / "metrics_comparison.png")
    length_histogram(all_generated, OUT_DIR / "length_histogram.png")
    training_loss_plot(OUT_DIR / "training_history.json", OUT_DIR / "training_loss.png")

    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()