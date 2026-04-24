"""Plotting helpers: confusion matrix and training curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_history(
    history_csv: str | Path,
    output_path: str | Path,
) -> None:
    import pandas as pd

    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "loss" in df and "val_loss" in df:
        axes[0].plot(df["epoch"], df["loss"], label="train")
        axes[0].plot(df["epoch"], df["val_loss"], label="val")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].legend()
        axes[0].set_title("Loss")

    if "accuracy" in df and "val_accuracy" in df:
        axes[1].plot(df["epoch"], df["accuracy"], label="train")
        axes[1].plot(df["epoch"], df["val_accuracy"], label="val")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("accuracy")
        axes[1].legend()
        axes[1].set_title("Accuracy")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
