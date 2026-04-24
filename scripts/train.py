"""Train a model from a YAML config.

Usage:
    python scripts/train.py --config configs/fusion/late_multi.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.data.dataset import (
    build_multimodal_dataset,
    build_text_dataset,
    build_image_generator,
    load_dataframe,
    split_dataframe,
)
from src.models.factory import build_model
from src.training.trainer import Trainer


def _build_datasets(cfg: dict):
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    df, _encoder = load_dataframe(
        data_cfg["tsv_path"], binary_classes=data_cfg.get("binary_classes")
    )
    train_df, val_df, _test_df = split_dataframe(
        df,
        test_size=float(data_cfg.get("test_size", 0.2)),
        val_size=float(data_cfg.get("val_size", 0.1)),
        seed=int(data_cfg.get("seed", 15)),
    )

    mtype = model_cfg["type"]
    batch_size = int(data_cfg.get("batch_size", 32))

    if mtype == "text":
        ds_kwargs = dict(
            num_classes=int(model_cfg["num_classes"]),
            pretrained=model_cfg.get("pretrained", "bert-base-uncased"),
            max_length=int(model_cfg.get("max_length", 128)),
            batch_size=batch_size,
        )
        train_ds = build_text_dataset(train_df, shuffle=True, **ds_kwargs)
        val_ds = build_text_dataset(val_df, shuffle=False, **ds_kwargs)
        return train_ds, val_ds

    if mtype == "image":
        class_mode = "binary" if int(model_cfg["num_classes"]) == 1 else "categorical"
        train_ds = build_image_generator(
            train_df,
            image_size=int(model_cfg.get("image_size", 224)),
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=True,
        )
        val_ds = build_image_generator(
            val_df,
            image_size=int(model_cfg.get("image_size", 224)),
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False,
        )
        return train_ds, val_ds

    if mtype in {"early_fusion", "late_fusion"}:
        ds_kwargs = dict(
            num_classes=int(model_cfg["num_classes"]),
            pretrained=model_cfg.get("pretrained", "bert-base-uncased"),
            max_length=int(model_cfg.get("max_length", 128)),
            image_size=int(model_cfg.get("image_size", 224)),
            batch_size=batch_size,
        )
        train_ds = build_multimodal_dataset(train_df, shuffle=True, **ds_kwargs)
        val_ds = build_multimodal_dataset(val_df, shuffle=False, **ds_kwargs)
        return train_ds, val_ds

    raise ValueError(f"Unknown model type: {mtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a meme classifier.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write checkpoints and history.csv. "
        "Defaults to outputs/<config-stem>/.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "outputs" / Path(args.config).stem
    )

    model = build_model(cfg["model"])
    model.summary()

    train_ds, val_ds = _build_datasets(cfg)
    trainer = Trainer(cfg["training"], output_dir=output_dir)
    trainer.fit(
        model,
        train_ds,
        val_ds,
        num_classes=int(cfg["model"]["num_classes"]),
    )
    print(f"Training complete. Checkpoints at {output_dir}")


if __name__ == "__main__":
    main()
