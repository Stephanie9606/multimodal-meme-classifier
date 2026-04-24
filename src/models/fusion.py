"""Early- and late-fusion multimodal classifiers (text + image)."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Average,
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Input,
    LSTM,
)
from tensorflow.keras.models import Model
from transformers import TFBertModel

from src.models.image import build_image_feature_extractor


def _text_branch(
    pretrained: str,
    max_length: int,
    trainable_bert: bool,
    lstm_units: int = 128,
):
    """BERT → BiLSTM → 256-d text feature, per the paper's fusion design."""
    bert = TFBertModel.from_pretrained(pretrained, name="bert")
    bert.trainable = trainable_bert

    input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(
        shape=(max_length,), dtype=tf.int32, name="attention_mask"
    )
    embeddings = bert(input_ids, attention_mask=attention_mask)[0]
    text_feat = Bidirectional(LSTM(lstm_units), name="text_bilstm")(embeddings)
    return input_ids, attention_mask, text_feat


def build_early_fusion(
    num_classes: int,
    pretrained: str = "bert-base-uncased",
    max_length: int = 128,
    image_size: int = 224,
    hidden_units: int = 512,
    dropout: float = 0.5,
    trainable_bert: bool = True,
    fine_tuned_vgg: bool = True,
    name: str = "early_fusion",
) -> Model:
    """Concatenate text + image embeddings → shared classifier head."""
    input_ids, attention_mask, text_feat = _text_branch(
        pretrained, max_length, trainable_bert
    )

    image_extractor = build_image_feature_extractor(
        image_size=image_size, fine_tuned=fine_tuned_vgg
    )
    image_input = Input(shape=(image_size, image_size, 3), name="image")
    image_feat = image_extractor(image_input)
    image_feat = Dense(max_length, activation="relu", name="image_project")(image_feat)

    fused = Concatenate(name="fusion_concat")([text_feat, image_feat])
    x = Dense(hidden_units, activation="relu", name="fusion_hidden")(fused)
    x = Dropout(dropout, name="fusion_dropout")(x)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    logits = Dense(num_classes, activation=activation, name="classifier")(x)

    return Model(
        inputs=[input_ids, attention_mask, image_input],
        outputs=logits,
        name=name,
    )


def build_late_fusion(
    num_classes: int,
    pretrained: str = "bert-base-uncased",
    max_length: int = 128,
    image_size: int = 224,
    hidden_units: int = 512,
    dropout: float = 0.5,
    trainable_bert: bool = True,
    fine_tuned_vgg: bool = True,
    name: str = "late_fusion",
) -> Model:
    """Each modality predicts independently; probabilities are averaged."""
    input_ids, attention_mask, text_feat = _text_branch(
        pretrained, max_length, trainable_bert
    )
    activation = "sigmoid" if num_classes == 1 else "softmax"

    text_logits = Dense(
        num_classes, activation=activation, name="text_head"
    )(text_feat)

    image_extractor = build_image_feature_extractor(
        image_size=image_size, fine_tuned=fine_tuned_vgg
    )
    image_input = Input(shape=(image_size, image_size, 3), name="image")
    image_feat = image_extractor(image_input)
    x = Dense(hidden_units, activation="relu", name="image_hidden")(image_feat)
    x = Dropout(dropout, name="image_dropout")(x)
    image_logits = Dense(
        num_classes, activation=activation, name="image_head"
    )(x)

    fused_logits = Average(name="late_fusion_avg")([text_logits, image_logits])

    return Model(
        inputs=[input_ids, attention_mask, image_input],
        outputs=fused_logits,
        name=name,
    )
