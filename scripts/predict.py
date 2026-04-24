"""Predict a single meme (image + caption) with a trained model.

Usage:
    python scripts/predict.py \\
        --config configs/fusion/late_multi.yaml \\
        --checkpoint outputs/late_multi/best.weights.h5 \\
        --image sample.jpg \\
        --text "the caption here"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import tensorflow as tf

from src.config import load_config
from src.data.preprocessing import preprocess_image, tokenize_texts
from src.models.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-example inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", default=None, help="Caption text")
    parser.add_argument("--image", default=None, help="Path to meme image")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mtype = cfg["model"]["type"]

    model = build_model(cfg["model"])
    model.load_weights(args.checkpoint)

    inputs: list = []

    if mtype in {"text", "early_fusion", "late_fusion"}:
        if args.text is None:
            raise SystemExit("--text is required for this model type.")
        enc = tokenize_texts(
            [args.text],
            pretrained=cfg["model"].get("pretrained", "bert-base-uncased"),
            max_length=int(cfg["model"].get("max_length", 128)),
        )
        inputs.extend([enc["input_ids"], enc["attention_mask"]])

    if mtype in {"image", "early_fusion", "late_fusion"}:
        if args.image is None:
            raise SystemExit("--image is required for this model type.")
        img = preprocess_image(
            args.image, image_size=int(cfg["model"].get("image_size", 224))
        )
        inputs.append(tf.convert_to_tensor(img[None, ...]))

    probs = model.predict(inputs if len(inputs) > 1 else inputs[0])[0]

    if int(cfg["model"]["num_classes"]) == 1:
        print(f"Score (class 1 prob): {float(probs):.4f}")
    else:
        top = int(np.argmax(probs))
        print(f"Predicted class index: {top}")
        print(f"Per-class probabilities: {probs.tolist()}")


if __name__ == "__main__":
    main()
