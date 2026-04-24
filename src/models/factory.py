"""Single entry point that turns a config dict into a Keras model."""
from __future__ import annotations

from tensorflow.keras.models import Model

from src.models.fusion import build_early_fusion, build_late_fusion
from src.models.image import build_image_classifier
from src.models.text import build_text_classifier


def build_model(model_cfg: dict) -> Model:
    """Dispatch on ``model_cfg['type']`` ∈ {text, image, early_fusion, late_fusion}.

    The remaining keys in ``model_cfg`` are forwarded verbatim to the chosen
    builder, except for ``type`` itself.
    """
    cfg = dict(model_cfg)
    model_type = cfg.pop("type")

    builders = {
        "text": build_text_classifier,
        "image": build_image_classifier,
        "early_fusion": build_early_fusion,
        "late_fusion": build_late_fusion,
    }
    if model_type not in builders:
        raise ValueError(
            f"Unknown model type {model_type!r}. Must be one of {list(builders)}."
        )
    return builders[model_type](**cfg)
