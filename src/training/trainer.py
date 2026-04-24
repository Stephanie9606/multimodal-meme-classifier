"""Training orchestration: compile + fit + callbacks."""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from src.training.callbacks import default_callbacks


class Trainer:
    """Thin wrapper around ``model.compile`` + ``model.fit``.

    The whole run is configured via the ``training`` section of the YAML
    config; see ``configs/*/*.yaml`` for examples.
    """

    def __init__(self, training_cfg: dict, output_dir: str | Path):
        self.cfg = training_cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _loss(self, num_classes: int) -> str:
        return "binary_crossentropy" if num_classes == 1 else "categorical_crossentropy"

    def _optimizer(self) -> tf.keras.optimizers.Optimizer:
        name = self.cfg.get("optimizer", "adam").lower()
        lr = float(self.cfg["learning_rate"])
        if name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=lr)
        if name == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=lr)
        raise ValueError(f"Unsupported optimizer {name!r}")

    def compile(self, model: tf.keras.Model, num_classes: int) -> None:
        model.compile(
            optimizer=self._optimizer(),
            loss=self._loss(num_classes),
            metrics=["accuracy"],
        )

    def fit(
        self,
        model: tf.keras.Model,
        train_ds,
        val_ds,
        num_classes: int,
    ) -> tf.keras.callbacks.History:
        self.compile(model, num_classes)
        return model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(self.cfg.get("epochs", 10)),
            callbacks=default_callbacks(self.output_dir, self.cfg),
            verbose=int(self.cfg.get("verbose", 2)),
        )
