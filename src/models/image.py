"""VGG16-based image classifier."""
from __future__ import annotations

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model


def _vgg_backbone(fine_tuned: bool, image_size: int) -> Model:
    """Return a VGG16 feature extractor.

    * ``fine_tuned=False`` returns full VGG16 including the 1000-way head
      (used when the head is discarded via the imagenet logits pooling).
    * ``fine_tuned=True`` swaps the last layer so the 4096-d fc7 features
      are exposed — matches ``fine_tune_vgg_memes_TF`` in the paper.
    """
    base = VGG16(
        weights="imagenet",
        include_top=True,
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False
    if fine_tuned:
        return Model(base.input, base.layers[-2].output, name="vgg16_fc7")
    return base


def build_image_classifier(
    num_classes: int,
    image_size: int = 224,
    hidden_units: int = 512,
    dropout: float = 0.5,
    fine_tuned: bool = True,
    name: str = "image_classifier",
) -> Model:
    """VGG16 (frozen) + Flatten + Dense(relu) + Dropout + Dense head."""
    backbone = _vgg_backbone(fine_tuned=fine_tuned, image_size=image_size)

    inputs = Input(shape=(image_size, image_size, 3), name="image")
    x = backbone(inputs, training=False)
    x = Flatten(name="flatten")(x)
    x = Dense(hidden_units, activation="relu", name="fc_hidden")(x)
    x = Dropout(dropout, name="dropout")(x)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    logits = Dense(num_classes, activation=activation, name="classifier")(x)

    return Model(inputs=inputs, outputs=logits, name=name)


def build_image_feature_extractor(
    image_size: int = 224,
    fine_tuned: bool = True,
    name: str = "image_feature_extractor",
) -> Model:
    """Return a VGG16 that outputs fc7 (or imagenet logits) embeddings only.

    Used as the image branch of the multimodal fusion models.
    """
    backbone = _vgg_backbone(fine_tuned=fine_tuned, image_size=image_size)
    inputs = Input(shape=(image_size, image_size, 3), name="image")
    features = backbone(inputs, training=False)
    features = Flatten(name="flatten")(features)
    return Model(inputs=inputs, outputs=features, name=name)
