"""Text tokenization and image loading utilities."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from transformers import AutoTokenizer


@lru_cache(maxsize=2)
def get_tokenizer(pretrained: str = "bert-base-uncased"):
    return AutoTokenizer.from_pretrained(pretrained)


def tokenize_texts(
    texts: list[str],
    pretrained: str = "bert-base-uncased",
    max_length: int = 128,
) -> dict:
    """Return dict with ``input_ids`` and ``attention_mask`` as tf tensors."""
    tokenizer = get_tokenizer(pretrained)
    encoded = tokenizer(
        text=texts,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=True,
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def preprocess_image(path: str, image_size: int = 224) -> np.ndarray:
    """Load a single image from disk and return a VGG16-ready array."""
    img = tf.keras.preprocessing.image.load_img(
        path, target_size=(image_size, image_size)
    )
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = vgg16.preprocess_input(arr)
    arr = arr / 255.0
    return arr
