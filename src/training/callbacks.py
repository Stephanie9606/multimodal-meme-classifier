"""Default callback set: checkpoint + early stopping + CSV logger."""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def default_callbacks(output_dir: Path, training_cfg: dict):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best.weights.h5"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "history.csv")),
    ]

    patience = training_cfg.get("early_stopping_patience")
    if patience:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(patience),
                restore_best_weights=True,
            )
        )

    return callbacks
