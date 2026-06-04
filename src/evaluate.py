import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm
from predict import predict_with_uncertainty, load_model

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
FIGURES_DIR = "./results/figures"
N_EVAL      = 300


def evaluate_dataset(model, tokenizer):
    dataset = load_dataset("dair-ai/emotion", split="test")
    indices = np.random.choice(len(dataset), min(N_EVAL, len(dataset)), replace=False)

    true_labels, pred_labels, confidences, uncertainties = [], [], [], []

    print(f"Running MC Dropout evaluation on {len(indices)} samples...")
    for idx in tqdm(indices):
        example = dataset[int(idx)]
        result  = predict_with_uncertainty(example["text"], model, tokenizer)

        true_labels.append(example["label"])
        pred_labels.append(LABEL_NAMES.index(result["predicted_emotion"]))
        confidences.append(result["confidence_pct"] / 100)
        uncertainties.append(result["uncertainty_pct"] / 100)

    return true_labels, pred_labels, confidences, uncertainties


def plot_confusion_matrix(true_labels, pred_labels):
    cm  = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)

    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(LABEL_NAMES, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black", fontsize=10)

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Uncertainty-Aware Emotion Classifier", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print("  Saved: confusion_matrix.png")


def plot_uncertainty_distribution(true_labels, pred_labels, uncertainties):
    correct   = [t == p for t, p in zip(true_labels, pred_labels)]
    corr_unc  = [u for u, c in zip(uncertainties, correct) if c]
    wrong_unc = [u for u, c in zip(uncertainties, correct) if not c]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(corr_unc,  bins=25, alpha=0.65, color="steelblue", label=f"Correct ({len(corr_unc)})")
    ax.hist(wrong_unc, bins=25, alpha=0.65, color="tomato",    label=f"Incorrect ({len(wrong_unc)})")
    ax.set_xlabel("Uncertainty Score (Normalized Entropy)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Uncertainty Distribution: Correct vs Incorrect Predictions", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/uncertainty_distribution.png", dpi=150)
    plt.close()
    print("  Saved: uncertainty_distribution.png")


def plot_reliability_diagram(true_labels, pred_labels, confidences, n_bins=10):
    bins    = np.linspace(0, 1, n_bins + 1)
    correct = [t == p for t, p in zip(true_labels, pred_labels)]

    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = [bins[i] <= c < bins[i + 1] for c in confidences]
        if any(mask):
            idxs = [j for j, m in enumerate(mask) if m]
            bin_accs.append(np.mean([correct[j]     for j in idxs]))
            bin_confs.append(np.mean([confidences[j] for j in idxs]))
            bin_counts.append(len(idxs))
        else:
            bin_accs.append(0)
            bin_confs.append((bins[i] + bins[i + 1]) / 2)
            bin_counts.append(0)

    ece = sum(
        cnt * abs(acc - conf)
        for acc, conf, cnt in zip(bin_accs, bin_confs, bin_counts)
    ) / len(true_labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
    ax.bar(bin_confs, bin_accs, width=0.07, alpha=0.75, color="steelblue",
           edgecolor="white", label="Model")
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Reliability Diagram  |  ECE = {ece:.4f}", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/reliability_diagram.png", dpi=150)
    plt.close()
    print(f"  Saved: reliability_diagram.png  (ECE = {ece:.4f})")
    return ece


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model()

    true_labels, pred_labels, confidences, uncertainties = evaluate_dataset(model, tokenizer)

    acc = accuracy_score(true_labels, pred_labels)
    f1  = f1_score(true_labels, pred_labels, average="weighted")

    print(f"\nAccuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nPer-class Report:")
    print(classification_report(true_labels, pred_labels, target_names=LABEL_NAMES))

    print("\nGenerating plots...")
    plot_confusion_matrix(true_labels, pred_labels)
    plot_uncertainty_distribution(true_labels, pred_labels, uncertainties)
    ece = plot_reliability_diagram(true_labels, pred_labels, confidences)

    metrics = {
        "accuracy": round(acc, 4),
        "weighted_f1": round(f1, 4),
        "ece": round(ece, 4),
        "n_evaluated": len(true_labels)
    }
    with open("./results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nAll done! Check results/figures/ for your plots.")


if __name__ == "__main__":
    main()