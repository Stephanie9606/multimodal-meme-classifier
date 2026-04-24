"""Evaluate a trained model on the held-out test split.

Usage:
    python scripts/evaluate.py \\
        --config configs/fusion/late_multi.yaml \\
        --checkpoint outputs/late_multi/best.weights.h5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.config import load_config
from src.data.dataset import (
    build_multimodal_dataset,
    build_text_dataset,
    build_image_generator,
    load_dataframe,
    split_dataframe,
)
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualize import plot_confusion_matrix
from src.models.factory import build_model


def _build_test_ds(cfg: dict):
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    df, encoder = load_dataframe(
        data_cfg["tsv_path"], binary_classes=data_cfg.get("binary_classes")
    )
    _train_df, _val_df, test_df = split_dataframe(
        df,
        test_size=float(data_cfg.get("test_size", 0.2)),
        val_size=float(data_cfg.get("val_size", 0.1)),
        seed=int(data_cfg.get("seed", 15)),
    )
    batch_size = int(data_cfg.get("batch_size", 32))
    mtype = model_cfg["type"]

    if mtype == "text":
        return build_text_dataset(
            test_df,
            num_classes=int(model_cfg["num_classes"]),
            pretrained=model_cfg.get("pretrained", "bert-base-uncased"),
            max_length=int(model_cfg.get("max_length", 128)),
            batch_size=batch_size,
            shuffle=False,
        ), test_df, encoder

    if mtype == "image":
        class_mode = (
            "binary" if int(model_cfg["num_classes"]) == 1 else "categorical"
        )
        return build_image_generator(
            test_df,
            image_size=int(model_cfg.get("image_size", 224)),
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False,
        ), test_df, encoder

    if mtype in {"early_fusion", "late_fusion"}:
        return build_multimodal_dataset(
            test_df,
            num_classes=int(model_cfg["num_classes"]),
            pretrained=model_cfg.get("pretrained", "bert-base-uncased"),
            max_length=int(model_cfg.get("max_length", 128)),
            image_size=int(model_cfg.get("image_size", 224)),
            batch_size=batch_size,
            shuffle=False,
        ), test_df, encoder

    raise ValueError(f"Unknown model type: {mtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a meme classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "outputs" / Path(args.config).stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg["model"])
    model.load_weights(args.checkpoint)

    test_ds, test_df, encoder = _build_test_ds(cfg)

    y_pred = model.predict(test_ds)
    y_true = test_df["MemeLabel"].to_numpy()

    class_names = (
        list(encoder.classes_)
        if encoder is not None
        else cfg["data"].get("binary_classes")
    )
    metrics = compute_metrics(y_true, y_pred, class_names=class_names)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]),
        class_names=class_names,
        output_path=output_dir / "confusion_matrix.png",
        title=f"Confusion — {Path(args.config).stem}",
    )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
