"""Figures for report section 3.3 drawn from the saved Kaggle results (no new experiments).

    python src/make_model_figures.py

Writes results/figures/mlp_training_vs_validation_loss.png and results/figures/confusion_matrices_student_test.png.
Only the MLP has an epoch-wise history, so it is the only model with a loss plot; nothing is drawn for the other models.
"""
import os, ast
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
M = os.path.join(ROOT, "results", "models", "model_comparison")
OUT = os.path.join(ROOT, "results", "figures")


def mlp_loss():
    h = pd.read_csv(os.path.join(M, "MLP_grouped_early_stopping.csv"))
    c = pd.read_csv(os.path.join(M, "MLP_loss_curve.csv"))
    best = int(h.validation_log_loss.idxmin()) + 1
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.epoch, h.training_loss, marker="o", ms=3, label="training loss (fitting trips)")
    ax[0].plot(h.epoch, h.validation_log_loss, marker="o", ms=3, label="validation log loss (held-out trips)")
    ax[0].axvline(best, color="grey", ls="--", lw=1)
    ax[0].annotate(f"best epoch = {best}", (best, h.validation_log_loss.min()), xytext=(best - 8, h.validation_log_loss.min() + 0.04),
                   arrowprops=dict(arrowstyle="->", color="grey"))
    ax[0].set(xlabel="epoch", ylabel="loss", title="MLP monitoring run: inner holdout of whole trips")
    ax[0].legend()
    ax[1].plot(np.arange(1, len(c) + 1), c.training_loss, marker="o", ms=3, color="tab:green")
    ax[1].set(xlabel="epoch", ylabel="training loss", title=f"MLP final fit on all 40 trips ({len(c)} epochs chosen above)")
    fig.suptitle("MLP loss curves (the only model with an epoch-wise history; the two panels come from two different fits)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mlp_training_vs_validation_loss.png"), dpi=140)
    plt.close(fig)


def confusion():
    m = pd.read_csv(os.path.join(M, "metrics.csv"))
    t = m[m.partition == "student_test"].set_index("model")
    order = ["LogReg", "LightGBM", "RandomForest", "MLP", "KNN"]
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.6))
    for a, k in zip(ax, order):
        cm = np.array(ast.literal_eval(t.loc[k, "confusion_matrix"]))
        a.imshow(np.log10(cm + 1), cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=9,
                       color="white" if np.log10(cm[i, j] + 1) > 3.6 else "black")
        a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["wrong road", "right road"], yticklabels=["wrong road", "right road"],
              xlabel="predicted", title=k)
        if k == "LogReg":
            a.set_ylabel("actual")
    fig.suptitle("Candidate-level confusion matrices on the 10 student-test trips (threshold 0.5; colour on a log scale)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confusion_matrices_student_test.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    mlp_loss(); confusion()
    print("wrote mlp_training_vs_validation_loss.png and confusion_matrices_student_test.png")
