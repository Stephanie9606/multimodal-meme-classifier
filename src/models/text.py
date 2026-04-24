"""BERT-based text classifier."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalMaxPool1D, Input
from tensorflow.keras.models import Model
from transformers import TFBertModel


def build_text_classifier(
    num_classes: int,
    pretrained: str = "bert-base-uncased",
    max_length: int = 128,
    trainable_bert: bool = True,
    name: str = "text_classifier",
) -> Model:
    """BERT encoder + GlobalMaxPool + Dense head.

    Mirrors the original paper architecture: the BERT last-hidden-state is
    reduced over the sequence axis with a GlobalMaxPool1D, then fed into a
    single dense classification head.
    """
    bert = TFBertModel.from_pretrained(pretrained, name="bert")
    bert.trainable = trainable_bert

    input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

    embeddings = bert(input_ids, attention_mask=attention_mask)[0]
    pooled = GlobalMaxPool1D(name="seq_pool")(embeddings)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    logits = Dense(num_classes, activation=activation, name="classifier")(pooled)

    return Model(inputs=[input_ids, attention_mask], outputs=logits, name=name)
